# -*- coding: utf-8 -*-
"""P2 EDA：四张必含图 + 功率因数校验。图内文字用英文（GitHub 作品集 + 避免中文字体问题）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HOURLY_PATH = "data/processed/hourly.parquet"
MINUTE_PATH = "data/processed/minute_clean.parquet"
SEG_PATH = "outputs/p1b_segment_stats.csv"
OUTDIR = "outputs/eda"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})


def main():
    h = pd.read_parquet(HOURLY_PATH)
    kw = h["gap_kw"]
    valid = kw.dropna()

    # ---- 图 1+2：日内曲线（总体 / 工作日 vs 周末）----
    hv = kw.to_frame()
    hv["hod"] = hv.index.hour
    hv["dow"] = hv.index.dayofweek  # 0=Mon
    hv["is_weekend"] = hv["dow"] >= 5
    prof_all = hv.groupby("hod")["gap_kw"].mean()
    prof_wd = hv[~hv["is_weekend"]].groupby("hod")["gap_kw"].mean()
    prof_we = hv[hv["is_weekend"]].groupby("hod")["gap_kw"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(prof_all.index, prof_all.values, marker="o", ms=3)
    axes[0].set(title="Mean load by hour of day (all data)", xlabel="Hour of day", ylabel="kW")
    axes[1].plot(prof_wd.index, prof_wd.values, label="Weekday", marker="o", ms=3)
    axes[1].plot(prof_we.index, prof_we.values, label="Weekend", marker="s", ms=3)
    axes[1].set(title="Weekday vs weekend daily profile", xlabel="Hour of day", ylabel="kW")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/p2_1_intraday_profile.png")
    plt.close(fig)

    # ---- 图 3：按年日均负荷曲线（概念漂移检查，为 RevIN 提供证据）----
    daily = kw.resample("1D").mean()
    roll30 = daily.rolling(30, min_periods=15).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily.values, color="0.75", lw=0.6, label="Daily mean kW")
    ax.plot(roll30.index, roll30.values, color="tab:red", lw=1.8, label="30-day rolling mean")
    ax.set(title="Daily mean load over years (concept drift check; evidence for RevIN)",
           ylabel="kW")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/p2_2_yearly_drift.png")
    plt.close(fig)

    # 年度统计（写入报告用）
    yr = valid.groupby(valid.index.year).agg(["mean", "median"])
    print("--- 年度负荷统计 ---")
    print(yr.to_string())

    # ---- 图 4：gap 分布与分段边界 ----
    seg = pd.read_csv(SEG_PATH, parse_dates=["start", "end"])
    m = pd.read_parquet(MINUTE_PATH, columns=["Global_active_power", "Global_reactive_power", "impute_flag"])
    mv = m["Global_active_power"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1])
    ax = axes[0]
    ax.plot(mv.index, mv.values, lw=0.1, color="tab:blue")
    for _, r in seg.iterrows():
        ax.axvspan(r["start"], r["end"], color="tab:green", alpha=0.06)
    for i in range(len(seg) - 1):
        ax.axvspan(seg.loc[i, "end"], seg.loc[i + 1, "start"], color="tab:red", alpha=0.25)
    ax.set(title="Segment boundaries (red = >24h gaps, not imputed) over minute-level load",
           ylabel="kW")
    ax = axes[1]
    # 注意：gap 分布必须基于原始 '?' 掩码（清洗后短 gap 已被填补，不可再从未清洗数据数）
    raw_gap = pd.read_csv("data/household_power_consumption.txt", sep=";",
                          usecols=["Global_active_power"], na_values="?",
                          dtype={"Global_active_power": "float32"})["Global_active_power"]
    raw_na = raw_gap.isna().to_numpy()
    d = np.diff(raw_na.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if raw_na[0]:
        starts = [0] + starts
    if raw_na[-1]:
        ends = ends + [len(mv)]
    runs_h = np.array([(e - s) / 60 for s, e in zip(starts, ends)])
    bins = [0, 1, 2, 6, 24, 48, 72, 96, 125]
    cats = pd.cut(runs_h, bins)
    cnt = pd.Series(runs_h).groupby(cats, observed=True).count()
    ax.bar(range(len(cnt)), cnt.values)
    ax.set_xticks(range(len(cnt)))
    ax.set_xticklabels([str(c) for c in cnt.index], rotation=20, ha="right")
    ax.set(title="Gap length distribution (hours, log count)", ylabel="count", yscale="log")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/p2_3_gaps_segments.png")
    plt.close(fig)
    print(f">24h gaps (h): {sorted(runs_h[runs_h > 24].round(1))}")

    # ---- 图 5：功率因数校验 ----
    gap = m["Global_active_power"].to_numpy(dtype=np.float64)
    grp = m["Global_reactive_power"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        pf = gap / np.sqrt(gap ** 2 + grp ** 2)
    pf = pf[np.isfinite(pf) & (gap > 0.01)]  # 近零功率处 PF 无意义
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pf, bins=80, range=(0.85, 1.0))
    ax.set(title=f"Power factor (GAP>0.01kW): median={np.median(pf):.3f}, "
                 f"P1={np.percentile(pf,1):.3f}, P99={np.percentile(pf,99):.3f}",
           xlabel="PF", ylabel="count")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/p2_4_power_factor.png")
    plt.close(fig)
    print(f"PF: median={np.median(pf):.3f}, P1={np.percentile(pf,1):.3f}, "
          f"P0.1={np.percentile(pf,0.1):.3f}, 低于0.9占比={(pf<0.9).mean():.3%}")
    hi = (gap > 0.3) & np.isfinite(pf_all := gap / np.sqrt(gap ** 2 + grp ** 2))
    print(f"PF(GAP>0.3kW): median={np.median(pf_all[hi]):.3f}, "
          f"低于0.9占比={(pf_all[hi]<0.9).mean():.3%}, 样本占 {(gap>0.3).mean():.1%}")

    # ---- Sub_metering EDA 补充（只 EDA 不进模型）----
    sub_cols = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    mm = pd.read_parquet(MINUTE_PATH, columns=sub_cols)
    # sub_metering 单位为 Wh/分钟（UCI 口径），日均贡献 kWh/day = mean(Wh/min)×60×24/1000
    contrib = (mm.mean() * 60 * 24 / 1000).round(3)
    total_kwh_day = valid.mean() * 24
    print("--- Sub_metering 日均贡献 (kWh/day) 与占 GAP 比例 ---")
    for c in sub_cols:
        print(f"{c}: {contrib[c]} kWh/day ({contrib[c]/total_kwh_day:.1%})")
    print(f"GAP 日均: {total_kwh_day:.2f} kWh/day")

    print(f"\n图已存: {OUTDIR}/")


if __name__ == "__main__":
    main()
