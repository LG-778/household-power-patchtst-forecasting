# -*- coding: utf-8 -*-
"""P3 公共模块：窗口构造、特征工程、指标计算（pandas 路径）、切分。

窗口口径：origin t = 输入窗口最后一个已知小时（t-L+1..t），预测 t+1..t+H。
窗口只在分段内部构造（P1b 已保证小时级有效段不跨界）。
泄漏审查：所有特征只使用 ≤ t 的信息；目标为 t+1..t+H。逐条见 build_features。
指标口径（用户 2026-09-10 定）：主口径 = 扁平化整体（全部预测点 pooled）；
同时报 horizon 均值（逐 horizon 指标再平均）。balanced（每 horizon 点数相等）
时 MAE 两口径数学等价，RMSE/sMAPE 可能不同。
"""
import numpy as np
import pandas as pd

L, H = 168, 24
SEED = 42
HOURLY_PATH = "data/processed/hourly.parquet"

VAL_FRAC, TEST_FRAC = 0.15, 0.15

LAG_LIST = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168]


def load_hourly(start=None, end=None):
    h = pd.read_parquet(HOURLY_PATH)
    if start:
        h = h[h.index >= pd.Timestamp(start)]
    if end:
        h = h[h.index <= pd.Timestamp(end)]
    return h


def window_origins(h):
    """返回可构造窗口的 origin 位置数组（输入 L + 输出 H 全在有效段内）。"""
    valid = h["gap_kw"].notna().to_numpy()
    d = np.diff(valid.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if valid[0]:
        starts = [0] + starts
    if valid[-1]:
        ends = ends + [len(valid)]
    pos = []
    for s, e in zip(starts, ends):
        n = e - s
        if n >= L + H:
            pos.extend(range(s + L - 1, e - H))
    return np.array(sorted(pos))


def build_features(h, origins):
    """XGBoost 特征（每个 origin 一行）。逐条泄漏审查：全部只用 ≤ t 的值。

    - hour(t), dow(t), is_weekend(t)：日历特征，硬约束 11 要求的显式 hour-of-day/DOW
    - lag_k = y[t+1-k]（k=1 → y[t] 最新值；k=168 → y[t-167] 窗口最旧值）
    - roll_mean_24h = mean(y[t-23..t])（含最新值，无泄漏）
    - roll_std_24h、roll_mean_168h = mean(y[t-167..t])
    目标为 y[t+1..t+H]，所有特征位置 ≤ t < 首个预测点，审查通过。
    """
    s = h["gap_kw"]
    y = s.to_numpy()
    hod = s.index.hour.to_numpy()
    dow = s.index.dayofweek.to_numpy()
    rows = {}
    rows["hour"] = hod[origins]
    rows["dow"] = dow[origins]
    rows["is_weekend"] = (rows["dow"] >= 5).astype(int)
    for k in LAG_LIST:
        rows[f"lag_{k}h"] = y[origins + 1 - k]
    rm24 = pd.Series(y).rolling(24)
    rm168 = pd.Series(y).rolling(168)
    m24 = rm24.mean().to_numpy()
    sd24 = rm24.std().to_numpy()
    m168 = rm168.mean().to_numpy()
    rows["roll_mean_24h"] = m24[origins]
    rows["roll_std_24h"] = sd24[origins]
    rows["roll_mean_168h"] = m168[origins]
    X = pd.DataFrame(rows, index=s.index[origins])
    X.index.name = "origin"
    return X


def build_targets(h, origins):
    """每个 origin 的 24 步目标 y[t+1..t+H]（列 h1..h24）。"""
    y = h["gap_kw"].to_numpy()
    T = np.stack([y[origins + k] for k in range(1, H + 1)], axis=1)
    return pd.DataFrame(T, index=h.index[origins], columns=[f"h{k}" for k in range(1, H + 1)])


def time_split(origins_index):
    """按 origin 时间顺序切 train/val/test（不 shuffle）。切分点取整到天。"""
    t = origins_index
    n = len(t)
    i_val = int(n * (1 - VAL_FRAC - TEST_FRAC))
    i_test = int(n * (1 - TEST_FRAC))
    cut_val = t[i_val].normalize()
    cut_test = t[i_test].normalize()
    tr = t < cut_val
    va = (t >= cut_val) & (t < cut_test)
    te = t >= cut_test
    return (np.where(tr)[0], np.where(va)[0], np.where(te)[0],
            cut_val, cut_test)


# ---------- 指标（pandas 路径；SQL 路径见 p3_run.py） ----------

def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    out = np.full_like(denom, np.nan, dtype=np.float64)
    np.divide(2 * np.abs(y_pred - y_true), denom, out=out, where=denom > 1e-8)
    return float(np.nanmean(out))


def metrics_pandas(y_true, y_pred):
    """y_true/y_pred: (n, 24)。返回 dict：扁平化整体 + horizon 均值两套口径。"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    flat_t, flat_p = yt.ravel(), yp.ravel()
    pooled = {
        "MAE": mean_absolute_error(flat_t, flat_p),
        "RMSE": float(np.sqrt(mean_squared_error(flat_t, flat_p))),
        "sMAPE": smape(flat_t, flat_p),
        "R2": r2_score(flat_t, flat_p),
    }
    h_mae = [mean_absolute_error(yt[:, k], yp[:, k]) for k in range(yt.shape[1])]
    h_rmse = [float(np.sqrt(mean_squared_error(yt[:, k], yp[:, k]))) for k in range(yt.shape[1])]
    h_smape = [smape(yt[:, k], yp[:, k]) for k in range(yt.shape[1])]
    h_r2 = [r2_score(yt[:, k], yp[:, k]) for k in range(yt.shape[1])]
    horizon = {
        "MAE": float(np.mean(h_mae)),
        "RMSE": float(np.mean(h_rmse)),
        "sMAPE": float(np.mean(h_smape)),
        "R2": float(np.mean(h_r2)),
        "MAE_by_h": h_mae,
    }
    return {"pooled": pooled, "horizon": horizon}
