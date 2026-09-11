# -*- coding: utf-8 -*-
"""P3 基线组运行器（采样门禁与全量共用本脚本，仅 --start/--end 不同）。

基线（5 个中的 4 个非深度学习；DLinear 走 neuralforecast，见 p3_dlinear.py）：
  persistence    ŷ[t+h] = y[t]
  seasonal_naive ŷ[t+h] = y[t+h-24]（昨日同时）
  ma7_same_hour  ŷ[t+h] = mean(y[t+h-24k], k=1..7)（同小时前 7 日均值）
  xgboost        MultiOutput（直接 24 输出）+ 显式 hour/dow 特征
评估：主口径扁平化整体 MAE；辅助 RMSE/sMAPE/R²；horizon 均值口径并报。
双路径核对（硬约束 18）：pandas vs SQLite SQL 计算核心指标，必须一致。
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import (H, L, SEED, build_features, build_targets, load_hourly,
                       metrics_pandas, time_split, window_origins)

OUTDIR = "outputs/p3"
os.makedirs(OUTDIR, exist_ok=True)

N_CV_FOLDS = 3


# ---------- 基线 ----------

def predict_persistence(y, origins):
    return np.repeat(y[origins][:, None], H, axis=1)


def predict_seasonal_naive(y, origins):
    return np.stack([y[origins + k - 24] for k in range(1, H + 1)], axis=1)


def predict_ma7_same_hour(y, origins):
    return np.stack([np.nanmean([y[origins + k - 24 * j] for j in range(1, 8)], axis=0)
                     for k in range(1, H + 1)], axis=1)


def predict_xgboost(X_tr, T_tr, X_va, T_va, X_te):
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor
    base = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=SEED, n_jobs=-1, tree_method="hist")
    model = MultiOutputRegressor(base)
    model.fit(X_tr, T_tr)
    return model.predict(X_va), model.predict(X_te), model


# ---------- SQL 双路径核对（硬约束 18） ----------

def metrics_sql(y_true, y_pred, tag):
    """把预测写入 SQLite，用 SQL 计算 pooled 指标，与 pandas 对照。"""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    hh = np.tile(np.arange(1, H + 1), len(yt) // H)
    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.execute("CREATE TABLE pred (h INT, yt REAL, yp REAL)")
    cur.executemany("INSERT INTO pred VALUES (?,?,?)", zip(hh, yt, yp))
    row = cur.execute("""
        SELECT AVG(ABS(yp-yt)),
               SQRT(AVG((yp-yt)*(yp-yt))),
               AVG(2.0*ABS(yp-yt)/NULLIF(ABS(yt)+ABS(yp),0)),
               1 - SUM((yp-yt)*(yp-yt)) / SUM((yt-(SELECT AVG(yt) FROM pred))*(yt-(SELECT AVG(yt) FROM pred)))
        FROM pred""").fetchone()
    sql_pooled = {"MAE": row[0], "RMSE": row[1], "sMAPE": row[2], "R2": row[4 - 1]}
    h_rows = cur.execute("""
        SELECT h, AVG(ABS(yp-yt)) FROM pred GROUP BY h ORDER BY h""").fetchall()
    sql_horizon_mae = float(np.mean([r[1] for r in h_rows]))
    con.close()
    return {"pooled": sql_pooled, "horizon_MAE": sql_horizon_mae}


def dual_path_check(name, y_true, y_pred, tag, max_rel_diff=1e-6):
    pd_res = metrics_pandas(y_true, y_pred)
    sql_res = metrics_sql(y_true, y_pred, tag)
    diffs = []
    for k in ["MAE", "RMSE", "sMAPE", "R2"]:
        a, b = pd_res["pooled"][k], sql_res["pooled"][k]
        rel = abs(a - b) / max(abs(a), 1e-12)
        diffs.append((k, a, b, rel))
    a, b = pd_res["horizon"]["MAE"], sql_res["horizon_MAE"]
    diffs.append(("hMAE", a, b, abs(a - b) / max(abs(a), 1e-12)))
    ok = all(d[3] < max_rel_diff for d in diffs)
    line = " | ".join(f"{k}: pandas={a:.6f} sql={b:.6f} rel={r:.2e}" for k, a, b, r in diffs)
    print(f"  [dual-path {'OK ' if ok else 'MISMATCH'}] {name}: {line}")
    return ok, pd_res


# ---------- rolling origin CV（硬约束 12：只在 test 切分点之前滚动） ----------

def rolling_origin_cv(h, origins, cut_test):
    """3-fold rolling origin：切点全部在 test 之前。eligible=test 前的 origins 四等分，
    fold i：train < 第 i 个内部分界，val = [第 i 个分界, 第 i+1 个分界)（至多 720）。"""
    times = h.index[origins]
    eligible_idx = np.where(times < cut_test)[0]
    chunks = np.array_split(eligible_idx, N_CV_FOLDS + 1)
    bounds = [times[ch[0]] for ch in chunks[1:]]  # 3 个内部切点（pandas Timestamp 标量）
    X = build_features(h, origins)
    T = build_targets(h, origins)
    results = []
    for i in range(N_CV_FOLDS):
        fold_end = bounds[i]
        next_bound = bounds[i + 1] if i + 1 < len(bounds) else cut_test
        tr_idx = np.where(times < fold_end)[0]
        va_mask = (times >= fold_end) & (times < next_bound)
        va_idx = np.where(va_mask)[0][:720]
        if len(tr_idx) < 500 or len(va_idx) < 100:
            continue
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
        base = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=SEED, n_jobs=-1, tree_method="hist")
        model = MultiOutputRegressor(base)
        model.fit(X.iloc[tr_idx], T.iloc[tr_idx])
        pred = model.predict(X.iloc[va_idx])
        res = metrics_pandas(T.iloc[va_idx], pred)
        results.append(res)
        print(f"  CV fold {i+1}: train origins < {fold_end}, val [{fold_end}, {next_bound}), "
              f"n={len(va_idx)}, pooled MAE={res['pooled']['MAE']:.4f}")
    if results:
        mae = float(np.mean([r["pooled"]["MAE"] for r in results]))
        sd = float(np.std([r["pooled"]["MAE"] for r in results]))
        print(f"  CV {len(results)} folds mean pooled MAE = {mae:.4f} +- {sd:.4f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="门禁子集开始日期（含）；全量不传")
    ap.add_argument("--end", default=None, help="门禁子集结束日期（含）")
    ap.add_argument("--tag", default="full", help="输出文件后缀")
    ap.add_argument("--skip-cv", action="store_true", help="跳过 CV（门禁模式可跳）")
    ap.add_argument("--only-cv", action="store_true", help="只跑 rolling origin CV（基线结果已存时）")
    args = ap.parse_args()

    h = load_hourly(args.start, args.end)
    origins = window_origins(h)
    y = h["gap_kw"].to_numpy()
    print(f"[{args.tag}] 小时行={len(h)}, 可构造窗口 origins={len(origins)}")
    if len(origins) < 300:
        print("窗口过少，无法评估")
        sys.exit(1)

    tr, va, te, cut_val, cut_test = time_split(h.index[origins])
    print(f"切分点: val >= {cut_val.date()}, test >= {cut_test.date()}  "
          f"(train={len(tr)}, val={len(va)}, test={len(te)})")

    X = build_features(h, origins)
    T = build_targets(h, origins)

    if args.only_cv and args.start is None:
        _, _, _, cut_val, cut_test = time_split(h.index[origins])
        print("rolling origin CV（仅 test 切分点之前）...")
        rolling_origin_cv(h, origins, cut_test)
        return

    preds = {
        "persistence": predict_persistence(y, origins),
        "seasonal_naive": predict_seasonal_naive(y, origins),
        "ma7_same_hour": predict_ma7_same_hour(y, origins),
    }
    print("训练 XGBoost（直接 24 输出 MultiOutput）...")
    pred_va_xgb, pred_te_xgb, xgb_model = predict_xgboost(
        X.iloc[tr], T.iloc[tr], X.iloc[va], T.iloc[va], X.iloc[te])
    preds_xgb_full = np.full((len(origins), H), np.nan)
    preds_xgb_full[va] = pred_va_xgb
    preds_xgb_full[te] = pred_te_xgb
    preds["xgboost"] = preds_xgb_full

    rows, all_ok = [], True
    detail = {}
    for name, P in preds.items():
        eval_idx = te if name == "xgboost" else np.r_[va, te]
        yt, yp = T.iloc[eval_idx].to_numpy(), P[eval_idx]
        ok, res = dual_path_check(name, yt, yp, args.tag)
        all_ok &= ok
        rows.append({
            "model": name, "n_eval": len(eval_idx),
            "MAE_pooled": round(res["pooled"]["MAE"], 4),
            "MAE_horizon_mean": round(res["horizon"]["MAE"], 4),
            "RMSE_pooled": round(res["pooled"]["RMSE"], 4),
            "sMAPE_pooled": round(res["pooled"]["sMAPE"], 4),
            "R2_pooled": round(res["pooled"]["R2"], 4),
        })
        detail[name] = res
        print(f"  {name:16s} MAE={res['pooled']['MAE']:.4f} RMSE={res['pooled']['RMSE']:.4f} "
              f"sMAPE={res['pooled']['sMAPE']:.4f} R2={res['pooled']['R2']:.4f}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(f"{OUTDIR}/baseline_results_{args.tag}.csv", index=False, encoding="utf-8-sig")
    np.save(f"{OUTDIR}/xgb_pred_test_{args.tag}.npy", pred_te_xgb)
    T.iloc[te].to_csv(f"{OUTDIR}/y_test_{args.tag}.csv", encoding="utf-8-sig")

    if not args.skip_cv and args.start is None:
        print("rolling origin CV（仅 test 切分点之前）...")
        rolling_origin_cv(h, origins, cut_test)

    print(f"\n双路径核对: {'全部一致' if all_ok else '存在不一致！'}")
    print(res_df.to_string(index=False))
    print(f"结果已存: {OUTDIR}/baseline_results_{args.tag}.csv")
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
