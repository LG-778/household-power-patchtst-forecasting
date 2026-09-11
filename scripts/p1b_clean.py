# -*- coding: utf-8 -*-
"""P1b 清洗：确定性插补 + 分段边界 + 小时级重采样。

插补规则（用户 2026-09-10 确认）：
  - <2h（<120 分钟）gap：interpolate（线性，显式 limit_direction='forward'，limit_area='inside'）
  - 2–24h（120–1440 分钟）gap：ffill + 插补标记列
  - >24h（>1440 分钟）gap：不插补，作分段边界
重采样规则（写死）：kW 负荷 = mean；能耗 kWh = sum(minute_kW)/60，另列命名。
小时有效性规则：60 分钟全部有值（含插补值）才算有效小时；跨界小时置 NaN，天然成分段边界。
"""
import os

import numpy as np
import pandas as pd

DATA_PATH = "data/household_power_consumption.txt"
MINUTE_OUT = "data/processed/minute_clean.parquet"
HOURLY_OUT = "data/processed/hourly.parquet"
SEG_OUT = "outputs/p1b_segment_stats.csv"

L, H = 168, 24  # lookback / horizon（小时）
SHORT_MAX = 119    # <2h：interpolate 最多连续填 119 个 NaN
MID_MAX = 1440     # 2–24h：ffill 区间上限（分钟）

NUM_COLS = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
TARGET = "Global_active_power"


def nan_runs(mask: np.ndarray):
    """返回 NaN 连续段的 (start, end_exclusive, length) 列表。"""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return [(s, e, e - s) for s, e in zip(starts, ends)]


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv(
        DATA_PATH, sep=";", usecols=["Date", "Time"] + NUM_COLS,
        na_values="?", dtype={c: "float32" for c in NUM_COLS},
    )
    n_raw = len(df)
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.set_index(dt).drop(columns=["Date", "Time"])
    df.index.name = "dt"

    # 缺失整行级对齐校验（P1a 指纹结论的硬断言）
    masks = [df[c].isna().to_numpy() for c in NUM_COLS]
    assert all(np.array_equal(masks[0], m) for m in masks[1:]), "各列 NaN 掩码不一致，非整行级缺失"

    # NaN 段分类
    runs = nan_runs(masks[0])
    n_short = sum(1 for _, _, ln in runs if ln <= SHORT_MAX)
    n_mid = sum(1 for _, _, ln in runs if SHORT_MAX < ln <= MID_MAX)
    n_long = sum(1 for _, _, ln in runs if ln > MID_MAX)
    print(f"NaN 段分类: <2h={n_short}, 2-24h={n_mid}, >24h={n_long}  (总 {len(runs)} 段)")
    assert (n_short, n_mid, n_long) == (64, 1, 6), "段数与 P1a 指纹不符，停止"

    # 插补标记列：0=原始 1=interpolate 2=ffill；边界段内部为 NaN
    flag = np.zeros(n_raw, dtype="float32")

    # 1) <2h：逐段 interpolate（显式 forward + inside）。
    #    注意：不能对全序列一次性 interpolate(limit=119)——pandas 的 limit 是
    #    「每个有效值之后最多连续填 N 个」，长 gap 的前 119 分钟会被误填
    #    （已实测确认），违反「>24h 不插补」。因此只在短段内部逐段插值。
    def interp_short_runs(col_vals: np.ndarray, fill_flag: np.ndarray, lo: int, hi: int) -> np.ndarray:
        out = col_vals.copy()
        for s, e, ln in runs:
            if lo <= ln <= hi:
                assert s > 0 and np.isfinite(col_vals[s - 1]) and np.isfinite(col_vals[e]), \
                    f"短段 [{s},{e}) 边界无有效值"
                seg = pd.Series(col_vals[s - 1:e + 1])
                filled = seg.interpolate(method="linear", limit_direction="forward",
                                         limit_area="inside").to_numpy()
                assert np.isfinite(filled[1:-1]).all(), f"短段 [{s},{e}) 未被完全填充"
                out[s:e] = filled[1:-1]
                fill_flag[s:e] = 1
        return out

    flag_target = flag
    df[TARGET] = interp_short_runs(df[TARGET].to_numpy(), flag_target, 1, SHORT_MAX)
    flag = flag_target

    # 2) 2–24h：ffill（逐段确定性地取段前一个有效值）
    for s, e, ln in runs:
        if SHORT_MAX < ln <= MID_MAX:
            assert s > 0 and df[TARGET].iloc[s - 1] == df[TARGET].iloc[s - 1], "ffill 段前无有效值"
            fill_val = df[TARGET].iloc[s - 1]
            df.iloc[s:e, df.columns.get_loc(TARGET)] = fill_val
            flag[s:e] = 2
    # >24h：保持 NaN（分段边界）

    # 其余 6 列同一套插补（特征/EDA 用），flag 语义与目标列一致
    dummy = np.zeros(n_raw, dtype="float32")
    for c in [c for c in NUM_COLS if c != TARGET]:
        df[c] = interp_short_runs(df[c].to_numpy(), dummy, 1, SHORT_MAX)
        for s, e, ln in runs:
            if SHORT_MAX < ln <= MID_MAX:
                df.iloc[s:e, df.columns.get_loc(c)] = df[c].iloc[s - 1]

    df["impute_flag"] = pd.Series(flag, index=df.index)
    # 长 gap 段内 flag 置 NaN（与负荷列一致：该区间无任何可用值）
    for s, e, ln in runs:
        if ln > MID_MAX:
            df.iloc[s:e, df.columns.get_loc("impute_flag")] = np.nan

    # 硬校验：长 gap 内零插补
    for s, e, ln in runs:
        if ln > MID_MAX:
            assert df[TARGET].iloc[s:e].isna().all(), f"长 gap [{s},{e}) 内出现插补值"

    # ---- 小时级重采样 ----
    valid = df[TARGET].notna()
    cnt = valid.resample("1h").sum()          # 每小时的非缺失分钟数
    hour_valid = cnt == 60                     # 60 分钟齐才算有效小时（跨界小时自动出局）
    g = df.resample("1h")
    hourly = pd.DataFrame({
        "gap_kw": g[TARGET].mean(),                       # kW = mean（写死）
        "gap_kwh": g[TARGET].sum() / 60.0,                # kWh = sum(minute kW)/60（写死，另列）
        "impute_flag": g["impute_flag"].max(),            # 小时内最高标记：1=含插补 2=含ffill
        "n_valid_min": cnt,
    })
    hourly[~hour_valid] = np.nan
    hourly.loc[~hour_valid, "n_valid_min"] = cnt[~hour_valid]
    hourly = hourly.astype({"gap_kw": "float32", "gap_kwh": "float32",
                            "impute_flag": "float32", "n_valid_min": "int32"})

    # ---- 分段统计（小时级，建模窗口口径 L=168 + H=24）----
    hvalid = hourly["gap_kw"].notna().to_numpy()
    hourly["segment_id"] = np.nan
    seg_runs = []
    d = np.diff(hvalid.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if hvalid[0]:
        starts = [0] + starts
    if hvalid[-1]:
        ends = ends + [len(hvalid)]
    for sid, (s, e) in enumerate(zip(starts, ends), 1):
        seg_len = e - s
        n_win = seg_len - (L + H) + 1 if seg_len >= L + H else 0
        seg_runs.append({
            "segment_id": sid,
            "start": hourly.index[s],
            "end": hourly.index[e - 1],
            "length_hours": seg_len,
            "n_windows": n_win,
        })
        hourly.iloc[s:e, hourly.columns.get_loc("segment_id")] = sid
    seg_df = pd.DataFrame(seg_runs)
    seg_df["window_share"] = seg_df["n_windows"] / seg_df["n_windows"].sum()

    # ---- 保存 ----
    df.to_parquet(MINUTE_OUT)
    hourly.to_parquet(HOURLY_OUT)
    seg_df.to_csv(SEG_OUT, index=False, encoding="utf-8-sig")

    total_win = int(seg_df["n_windows"].sum())
    print(f"行数守恒: raw={n_raw}, minute_clean={len(df)}")
    print(f"插补: short(<2h,interpolate)={int((flag==1).sum())} 分钟行, "
          f"mid(2-24h,ffill)={int((flag==2).sum())} 分钟行")
    print(f"分钟级缺失剩余(>24h边界): {int(df[TARGET].isna().sum())} 行")
    print(f"小时级: 总 {len(hourly)} h, 有效 {int(hvalid.sum())} h, 无效 {int((~hvalid).sum())} h")
    print(f"分段数: {len(seg_df)}, 总建模窗口数(L=168,H=24): {total_win}")
    print(seg_df.to_string(index=False))
    print(f"\n已保存: {MINUTE_OUT} | {HOURLY_OUT} | {SEG_OUT}")


if __name__ == "__main__":
    main()
