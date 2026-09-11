# -*- coding: utf-8 -*-
"""P1a 统计指纹：只输出统计量，禁止输出原始数据行（硬约束 1/16）。"""
import sys

import numpy as np
import pandas as pd

DATA_PATH = "data/household_power_consumption.txt"
OUT_PATH = "outputs/p1a_fingerprint.txt"

NUM_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

lines = []


def emit(s=""):
    lines.append(str(s))
    print(s)


def gap_run_lengths(mask: np.ndarray) -> np.ndarray:
    """布尔序列中 True 连续段的长度（单位：行=分钟）。"""
    if not mask.any():
        return np.array([], dtype=np.int64)
    d = np.diff(mask.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    return ends - starts


def gap_distribution(runs: np.ndarray, label: str):
    emit(f"\n--- {label} ---")
    if len(runs) == 0:
        emit("无 gap")
        return 0.0, 0.0
    hours = runs / 60.0
    buckets = [(1, 2), (2, 24), (24, np.inf)]
    total_min = runs.sum()
    for lo, hi in buckets:
        sel = runs[(hours >= lo) & (hours < hi)]
        share_time = runs[(hours >= lo) & (hours < hi)].sum() / total_min if total_min else 0
        emit(f"  [{lo}h, {'∞' if np.isinf(hi) else str(hi)+'h'}) gap: 段数={len(sel)}, "
             f"占总缺失时间比例={share_time:.2%}")
    emit(f"  最长 gap: {hours.max():.2f} h, 最短: {hours.min():.4f} h, 段数: {len(runs)}")
    long_mask = hours > 24
    long_share_time = runs[long_mask].sum() / total_min if total_min else 0.0
    emit(f"  >24h 空窗：段数={long_mask.sum()}，占总缺失时间比例={long_share_time:.2%}")
    return long_share_time, total_min


def main():
    # dtype 优化读入：数值列 float32，'?' 直接转 NaN，时间列单独解析（硬约束 4）
    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        usecols=["Date", "Time"] + NUM_COLS,
        na_values="?",
        dtype={c: "float32" for c in NUM_COLS},
    )
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    emit("=" * 60)
    emit("P1a 统计指纹（仅统计量，无原始数据行）")
    emit("=" * 60)
    emit(f"读入内存占用: {mem_mb:.1f} MB")
    emit(f"shape: {df.shape}")

    # 时间索引
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    emit(f"\n时间范围: {dt.min()} ~ {dt.max()}")
    n_dup = int(dt.duplicated().sum())
    emit(f"重复时间戳行数: {n_dup}")
    emit(f"时间严格递增: {bool(dt.is_monotonic_increasing and n_dup == 0)}")
    span_min = (dt.max() - dt.min()).total_seconds() / 60 + 1
    emit(f"起止跨度理论分钟数: {int(span_min)}  ｜ 实际行数: {len(df)}  ｜ 行缺失（整行缺席）: {int(span_min - len(df))}")

    df = df.set_index(dt).drop(columns=["Date", "Time"])
    df.index.name = "dt"

    emit("\n--- 每列 dtype / 缺失率 / 唯一值数 ---")
    for c in df.columns:
        miss = df[c].isna()
        emit(f"{c:22s} dtype={str(df[c].dtype):8s} 缺失={int(miss.sum()):>8d} "
             f"({miss.mean():.3%})  唯一值(非NaN)={df[c].nunique()}")

    emit("\n--- 数值列分位数（非缺失） ---")
    q = df[NUM_COLS].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    for c in NUM_COLS:
        vals = "  ".join(f"{p:.2f}" for p in q.loc[c])
        emit(f"{c:22s} [1%,5%,25%,50%,75%,95%,99%] = {vals}")

    emit("\n--- top 类别频次（数值列最常出现的取值） ---")
    for c in NUM_COLS:
        vc = df[c].value_counts().head(3)
        parts = ", ".join(f"{v:.3f}×{int(n)}次" for v, n in vc.items())
        emit(f"{c:22s} top3: {parts}")

    # gap 口径 A：时间戳整段缺席（相邻时间差 >1 分钟）
    diffs_min = df.index.to_series().diff().dt.total_seconds().div(60).dropna()
    absent_runs = (diffs_min[diffs_min > 1] - 1).to_numpy(dtype=np.int64)  # 每段的缺席分钟数
    emit("\n=== gap 口径 A：整行缺席（文件里连行都没有） ===")
    _, _ = gap_distribution(absent_runs, "整行缺席 gap 长度分布")

    # gap 口径 B：行存在但 '?'（NaN 连续段），逐列统计，重点看目标列
    emit("\n=== gap 口径 B：行存在但值为 '?'（NaN 连续段） ===")
    for c in NUM_COLS:
        runs = gap_run_lengths(df[c].isna().to_numpy())
        if c == "Global_active_power":
            long_share, total_min_b = gap_distribution(runs, f"[目标列 {c}]")
            share_of_span = total_min_b / span_min
            emit(f"  目标列 NaN 总时长占全跨度比例: {share_of_span:.3%}")
        else:
            if len(runs):
                emit(f"  {c:22s} NaN 段数={len(runs)}, 最长={runs.max()/60:.2f}h, 总时长={runs.sum()/60:.2f}h")
            else:
                emit(f"  {c:22s} 无 NaN 连续段")

    emit("\n--- 0 值检查（Sub_metering 正常可为 0；用于甄别 '?' 是否被误读成 0） ---")
    for c in NUM_COLS:
        emit(f"{c:22s} 值为 0 的行数: {int((df[c] == 0).sum())} ({(df[c] == 0).mean():.3%})")

    emit("\n--- 目标列概览 ---")
    gap = df["Global_active_power"]
    emit(f"非缺失均值={gap.mean():.4f} kW, 非缺失中位数={gap.median():.4f} kW, 最大值={gap.max():.4f} kW")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    emit(f"\n指纹已存: {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
