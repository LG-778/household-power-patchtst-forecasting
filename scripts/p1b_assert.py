# -*- coding: utf-8 -*-
"""P1b 独立不变量断言（硬约束 17 + 用户修订②）。

与 p1b_clean.py 完全独立：本脚本从原始 txt 重新推导期望，再与产出对照。
任何一条失败即非零退出并报告失败项。
"""
import sys

import numpy as np
import pandas as pd

DATA_PATH = "data/household_power_consumption.txt"
MINUTE_PATH = "data/processed/minute_clean.parquet"
HOURLY_PATH = "data/processed/hourly.parquet"
SEG_PATH = "outputs/p1b_segment_stats.csv"

SHORT_MAX, MID_MAX = 119, 1440
TARGET = "Global_active_power"

failures = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  | {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main():
    # ---- 读原始（独立路径） ----
    raw = pd.read_csv(DATA_PATH, sep=";", usecols=["Date", "Time", TARGET],
                      na_values="?", dtype={TARGET: "float32"})
    raw_n = len(raw)
    raw_dt = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    raw_na = raw[TARGET].isna().to_numpy()

    # ---- 读产出 ----
    m = pd.read_parquet(MINUTE_PATH)
    h = pd.read_parquet(HOURLY_PATH)
    seg = pd.read_csv(SEG_PATH)

    # 1) 行数守恒
    check("行数守恒: 原始=清洗后分钟数据", raw_n == len(m), f"{raw_n} vs {len(m)}")

    # 2) 时间列严格非递减且无重复
    idx = m.index
    check("时间严格递增且无重复", idx.is_unique and idx.is_monotonic_increasing)
    check("时间索引与原始一致", len(idx) == raw_n and (idx.to_numpy() == raw_dt.to_numpy()).all())

    # 3) 缺失只在 >24h 边界段内：重算原始 NaN 段分类
    d = np.diff(raw_na.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if raw_na[0]:
        starts = [0] + starts
    if raw_na[-1]:
        ends = ends + [raw_n]
    runs = [(s, e, e - s) for s, e in zip(starts, ends)]
    n_short = sum(1 for _, _, ln in runs if ln <= SHORT_MAX)
    n_mid = sum(1 for _, _, ln in runs if SHORT_MAX < ln <= MID_MAX)
    n_long = sum(1 for _, _, ln in runs if ln > MID_MAX)
    check("NaN 段分类 (64/1/6)", (n_short, n_mid, n_long) == (64, 1, 6),
          f"{n_short}/{n_mid}/{n_long}")

    mv = m[TARGET].to_numpy()
    # 长 gap 内必须全 NaN（零插补）
    long_ok = all(np.isnan(mv[s:e]).all() for s, e, ln in runs if ln > MID_MAX)
    check(">24h 边界段内零插补", long_ok)

    # 短段：线性插值（介于左右边界值之间）；中段：常数 ffill
    short_bad, mid_bad = 0, 0
    for s, e, ln in runs:
        if ln <= SHORT_MAX:
            lo, hi = mv[s - 1], mv[e]
            if not (np.nanmin(mv[s:e]) >= min(lo, hi) - 1e-6 and np.nanmax(mv[s:e]) <= max(lo, hi) + 1e-6):
                short_bad += 1
        elif ln <= MID_MAX:
            if not np.allclose(mv[s:e], mv[s - 1], atol=1e-6):
                mid_bad += 1
    check("短段插补值介于边界值之间", short_bad == 0, f"异常段数={short_bad}")
    check("中段 ffill 为常数且等于段前值", mid_bad == 0, f"异常段数={mid_bad}")

    # flag 语义核对
    flag = m["impute_flag"].to_numpy()
    in_any = np.zeros(raw_n, dtype=bool)
    for s, e, ln in runs:
        in_any[s:e] = True
    check("原始值 flag=0 / 插补区 flag∈{1,2} / 边界段内 flag=NaN",
          (flag[~raw_na] == 0).all()
          and np.isin(flag[raw_na & ~np.isnan(mv)], [1, 2]).all()
          and np.isnan(flag[raw_na & np.isnan(mv)]).all())

    # 4) 小时级：重采样口径与有效性
    # 独立重算每小时 mean/sum/有效分钟数
    ns = m.index.values.astype("datetime64[ns]").astype(np.int64)
    hour_key = (ns // 3_600_000_000_000).astype(np.int64)  # 对齐整点的 wall-clock 小时桶
    uh, inv = np.unique(hour_key, return_inverse=True)
    cnt = np.bincount(inv, weights=(~np.isnan(mv)).astype(np.float64))
    with np.errstate(invalid="ignore"):
        means = np.bincount(inv, weights=np.where(np.isnan(mv), 0, mv)) / np.maximum(cnt, 1)
        sums = np.bincount(inv, weights=np.where(np.isnan(mv), 0, mv))
    valid_h = cnt == 60
    check("小时有效性: 60分钟齐才有效",
          (h["n_valid_min"].to_numpy() == cnt.astype(np.int32)).all()
          and (h["gap_kw"].notna().to_numpy() == valid_h).all())

    ok_h = h["gap_kw"].notna().to_numpy()
    calc_mean = np.full(len(h), np.nan)
    calc_mean[ok_h] = means[valid_h]
    calc_kwh = np.full(len(h), np.nan)
    calc_kwh[ok_h] = sums[valid_h] / 60.0
    check("kW=mean 口径一致", np.allclose(h["gap_kw"].to_numpy()[ok_h], calc_mean[ok_h], atol=1e-3),
          f"最大偏差={np.nanmax(np.abs(h['gap_kw'].to_numpy()-calc_mean)):.2e}")
    check("kWh=sum/60 口径一致", np.allclose(h["gap_kwh"].to_numpy()[ok_h], calc_kwh[ok_h], atol=1e-3))

    # 5) kWh 能量守恒（容差 <0.5%，超出需定位）
    e_minute = np.nansum(np.where(np.isnan(mv), 0, mv).astype(np.float64)) / 60.0
    e_hourly = np.nansum(h["gap_kwh"].to_numpy()[ok_h].astype(np.float64))
    rel = abs(e_hourly - e_minute) / e_minute
    print(f"    能量: 分钟级={e_minute:.2f} kWh, 小时级={e_hourly:.2f} kWh, 相对偏差={rel:.3%}")
    if rel >= 0.005:
        # 定位差异最大的小时
        diffs = np.abs(calc_kwh[ok_h] - h["gap_kwh"].to_numpy()[ok_h])
        worst = np.argsort(diffs)[-5:][::-1]
        loc = [(str(h.index[ok_h][i]), float(diffs[i])) for i in worst]
        check("kWh 能量守恒 <0.5%", False, f"偏差={rel:.3%}, 差异最大小时: {loc}")
    else:
        check("kWh 能量守恒 <0.5%", True, f"偏差={rel:.3%}")

    # 6) lag/rolling 特征手工小样例对照（纯 Python 实现 vs pandas）
    hv = h["gap_kw"]
    probe = hv.dropna()
    base_pos = probe.index.get_loc(probe.index[5000])  # 小时表位置
    series = hv.to_numpy()
    # 手工构造 lag1, lag24, roll3（numpy 切片，不用 pandas shift/rolling）
    def manual_vals(pos):
        return {
            "lag1": series[pos - 1],
            "lag24": series[pos - 24],
            "roll3": np.nanmean(series[pos - 2:pos + 1]),
        }
    s_pd = pd.DataFrame({"x": hv})
    s_pd["lag1"] = s_pd["x"].shift(1)
    s_pd["lag24"] = s_pd["x"].shift(24)
    s_pd["roll3"] = s_pd["x"].rolling(3).mean()
    mism = 0
    for pos in range(base_pos, base_pos + 5):
        man = manual_vals(pos)
        for k in man:
            a, b = man[k], s_pd[k].iloc[pos]
            if not (np.isnan(a) and np.isnan(b)) and not np.isclose(a, b, equal_nan=True, atol=1e-6):
                mism += 1
    check("lag1/lag24/roll3 手工 vs pandas 对照（5 行小样例）", mism == 0, f"不一致项={mism}")

    # 7) 分段统计表核对
    total_win = 0
    seg_ok = True
    for _, r in seg.iterrows():
        mask = (h["segment_id"] == r.segment_id).to_numpy()
        expect_win = max(0, int(mask.sum()) - 191)
        if mask.sum() != r.length_hours or r.n_windows != expect_win:
            seg_ok = False
        total_win += r.n_windows
    check("分段统计: 长度与窗口数=L+H-191 口径", seg_ok, f"总窗口={total_win}")
    check("总窗口量 >= 3 万（样本充足性）", total_win >= 30000, f"总窗口={total_win}")

    print("\n" + ("ALL PASS" if not failures else f"FAILED {len(failures)}: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
