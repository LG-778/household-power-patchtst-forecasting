# -*- coding: utf-8 -*-
"""滚动 168h 块评估（统一协议，防「单月训练窗」陷阱）。

协议：test/val origins 按 168h 分块；每块只用数据 < 块起点 重新训练，
预测该块全部 origins。无状态基线（persistence/seasonal/ma7）直接用全序列。
模型对比表只使用所有模型共同覆盖的 origins。

泄漏审查：块内任意 origin t 的训练数据严格 < 块起点 ≤ t，无越界；
跨段块按段拆开处理（段内小时连续）。
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import H, L, SEED, build_features, build_targets, load_hourly, time_split, window_origins

warnings.filterwarnings("ignore")
os.makedirs("outputs/p5", exist_ok=True)

BLOCK = 168  # 滚动块长（小时）= lookback，语义「每周重训」


def blocks_of(times):
    """DatetimeIndex → (block_start, origins 子索引) 列表。

    用 Timedelta 整除分块，单位无关——parquet 读回的索引是 datetime64[us]，
    view('int64') 得到的是微秒，若按纳秒常量整除会全部归 0（曾致每段合并成一块、
    模型 stale，见 decisions.md）。
    """
    times = pd.DatetimeIndex(times)
    key = ((times - times[0]) // pd.Timedelta(hours=BLOCK)).to_numpy()
    out = []
    for k in np.unique(key):
        sel = np.where(key == k)[0]
        out.append((times[sel[0]], sel))
    return out


def run_xgboost_block(h, origins, block_start, sel, n_estimators=300):
    y = h["gap_kw"].to_numpy()
    times = h.index[origins]
    tr = np.where(times < block_start)[0]
    X = build_features(h, origins)
    T = build_targets(h, origins)
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor
    base = XGBRegressor(n_estimators=n_estimators, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=SEED, n_jobs=-1, tree_method="hist")
    model = MultiOutputRegressor(base)
    model.fit(X.iloc[tr], T.iloc[tr])
    return model.predict(X.iloc[sel])


def run_nf_block(h, origins, block_start, sel, model_factory, seg_cache):
    """neuralforecast 模型在单个（段, 块）上：外部 fit + use_fitted cross_validation。"""
    from neuralforecast import NeuralForecast
    times = h.index[origins]
    block_times = times[sel]
    t0, t1 = block_times[0], block_times[-1]
    s = seg_cache
    train_df = s[s.index < t0]
    full_df = s[s.index <= t1 + pd.Timedelta(hours=H)]
    n_win = len(sel)
    assert len(full_df) - len(train_df) == n_win + H, "块窗口数与数据切片不符"
    if len(train_df) < L:
        print(f"  block {t0}: 训练历史仅 {len(train_df)}h < {L}h，DL 不覆盖（数据限制）")
        return None
    ydf_tr = pd.DataFrame({"unique_id": "hhs", "ds": train_df.index, "y": train_df.to_numpy()})
    ydf_full = pd.DataFrame({"unique_id": "hhs", "ds": full_df.index, "y": full_df.to_numpy()})
    # val_size 自适应：训练段紧时收缩 early-stop 窗口，极端短历史时放弃 early stop（val_size=0）
    room = len(train_df) - L
    val_size = max(0, min(24 * 30, room // 4)) if room >= 24 else 0
    model = model_factory(early_stop=val_size > 0)
    nf = NeuralForecast(models=[model], freq="h")
    nf.fit(df=ydf_tr, val_size=val_size)
    cv = nf.cross_validation(df=ydf_full, h=H, n_windows=n_win, step_size=1,
                             refit=False, use_fitted=True)
    name = type(model).__name__
    cv = cv.sort_values(["cutoff", "ds"])
    pred = np.stack(cv.groupby("cutoff", sort=True)[name].apply(np.asarray).to_numpy())
    cutoffs = pd.DatetimeIndex(cv["cutoff"].sort_values().unique())
    assert (cutoffs == block_times).all(), f"{name} cutoff 与 origins 未对齐"
    return pred


def segments_of(h):
    valid = h["gap_kw"].notna().to_numpy()
    d = np.diff(valid.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if valid[0]:
        starts = [0] + starts
    if valid[-1]:
        ends = ends + [len(valid)]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= L + H + 24]


def rolling_eval(h, origins, eval_idx, model_kind, model=None):
    """返回 DataFrame(index=origin_time, columns=h1..h24)。model_kind: xgboost/dlinear/patchtst。"""
    times = h.index[origins]
    eval_times = times[eval_idx]
    preds = {}
    t_start = time.time()
    if model_kind == "xgboost":
        for bstart, sel in blocks_of(eval_times):
            gsel = eval_idx[sel]
            preds[bstart] = (gsel, run_xgboost_block(h, origins, bstart, gsel))
    else:
        # 按段切 eval origins，再分块
        seg_bounds = segments_of(h)
        for s, e in seg_bounds:
            seg_times = h.index[s:e]
            in_seg = np.where(np.isin(times[eval_idx], seg_times))[0]
            if len(in_seg) == 0:
                continue
            seg_series = h["gap_kw"].iloc[s:e]
            for bstart, sel in blocks_of(eval_times[in_seg]):
                gsel = eval_idx[in_seg[sel]]
                r = run_nf_block(h, origins, bstart, gsel, model, seg_series)
                if r is not None:
                    preds[bstart] = (gsel, r)
                print(f"  block {bstart}: n={len(gsel)} ({time.time()-t_start:.0f}s)")
    all_sel, all_pred = [], []
    for gsel, p in preds.values():
        all_sel.append(gsel)
        all_pred.append(p)
    all_sel = np.concatenate(all_sel)
    all_pred = np.vstack(all_pred)
    order = np.argsort(all_sel)
    return pd.DataFrame(all_pred[order], index=times[all_sel[order]])


def stateless_preds(h, origins, eval_idx):
    y = h["gap_kw"].to_numpy()
    o = origins[eval_idx]
    return {
        "persistence": np.repeat(y[o][:, None], H, axis=1),
        "seasonal_naive": np.stack([y[o + k - 24] for k in range(1, H + 1)], axis=1),
        "ma7_same_hour": np.stack([np.nanmean([y[o + k - 24 * j] for j in range(1, 8)], axis=0)
                                   for k in range(1, H + 1)], axis=1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["xgboost", "dlinear", "patchtst"])
    ap.add_argument("--mode", required=True, choices=["val", "test"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--patch-len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    import torch
    if args.model in ("dlinear", "patchtst"):
        if not torch.cuda.is_available():
            print("FATAL: CUDA 不可用，按规约禁止静默回退 CPU。")
            sys.exit(3)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    h = load_hourly()
    origins = window_origins(h)
    tr, va, te, cut_val, cut_test = time_split(h.index[origins])
    eval_idx = va if args.mode == "val" else te
    print(f"[{args.run_id}] {args.model}/{args.mode}: eval_origins={len(eval_idx)}")

    if args.model == "xgboost":
        pred = rolling_eval(h, origins, eval_idx, "xgboost")
    else:
        from neuralforecast.losses.pytorch import MSE
        from neuralforecast.models import DLinear, PatchTST
        os.environ["RUN_ID"] = args.run_id
        os.environ["DEVICE"] = "cuda"
        if args.model == "dlinear":
            def mk(early_stop=True):
                return DLinear(h=H, input_size=L, loss=MSE(), max_steps=300,
                               early_stop_patience_steps=300 if early_stop else 0,
                               random_seed=SEED, enable_progress_bar=False)
        else:
            from p4_patchtst import EpochLogger
            def mk(early_stop=True):
                return PatchTST(h=H, input_size=L, patch_len=args.patch_len, stride=args.stride, revin=True,
                                n_heads=16, encoder_layers=3, dropout=args.dropout,
                                fc_dropout=args.dropout, loss=MSE(), valid_loss=MSE(),
                                learning_rate=args.lr,
                                batch_size=256, max_steps=args.max_steps,
                                early_stop_patience_steps=300 if early_stop else 0,
                                scaler_type="identity",
                                random_seed=SEED, enable_progress_bar=False,
                                callbacks=[EpochLogger(args.run_id, "cuda")])
        pred = rolling_eval(h, origins, eval_idx, args.model, mk)

    np.save(f"outputs/p5/{args.run_id}_{args.mode}_pred.npy",
            {"origins": pred.index, "pred": pred.to_numpy()}, allow_pickle=True)
    print(f"[{args.run_id}] saved outputs/p5/{args.run_id}_{args.mode}_pred.npy n={len(pred)}")
