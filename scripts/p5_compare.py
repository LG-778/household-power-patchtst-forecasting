# -*- coding: utf-8 -*-
"""P5 统一对比：所有模型共同覆盖 origins 上的滚动评估结果汇总。

输入：outputs/p5/{run_id}_{mode}_pred.npy（ origins + pred 24 列）
输出：对比表 CSV（pooled 主口径 + horizon 均值并报）+ 双路径核对
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import H, load_hourly, metrics_pandas
from p3_run import dual_path_check

MODE = sys.argv[1] if len(sys.argv) > 1 else "test"


def stateless(h, origins_pos):
    y = h["gap_kw"].to_numpy()
    o = origins_pos
    return {
        "persistence": np.repeat(y[o][:, None], H, axis=1),
        "seasonal_naive": np.stack([y[o + k - 24] for k in range(1, H + 1)], axis=1),
        "ma7_same_hour": np.stack([np.nanmean([y[o + k - 24 * j] for j in range(1, 8)], axis=0)
                                   for k in range(1, H + 1)], axis=1),
    }


def main():
    h = load_hourly()
    hv = h["gap_kw"].to_numpy()
    idx = h.index

    preds = {}
    for f in glob.glob(f"outputs/p5/*_{MODE}_pred.npy"):
        run = os.path.basename(f).replace(f"_{MODE}_pred.npy", "")
        d = np.load(f, allow_pickle=True).item()
        preds[run] = pd.DatetimeIndex(d["origins"]), d["pred"]
    if not preds:
        print("无预测文件")
        sys.exit(1)

    common = None
    for ot, _ in preds.values():
        common = ot if common is None else common.intersection(ot)
    print(f"共同 origins: {len(common)} (mode={MODE})")

    pos = idx.get_indexer(common)
    T = np.stack([hv[pos + k] for k in range(1, H + 1)], axis=1)

    rows, ok_all = [], True
    for name, (ot, p) in preds.items():
        m = np.isin(ot, common)
        p_common = p[m]
        ok, res = dual_path_check(name, T, p_common, MODE)
        ok_all &= ok
        rows.append({"model": name, "n": len(common),
                     "MAE_pooled": round(res["pooled"]["MAE"], 4),
                     "MAE_horizon_mean": round(res["horizon"]["MAE"], 4),
                     "RMSE": round(res["pooled"]["RMSE"], 4),
                     "sMAPE": round(res["pooled"]["sMAPE"], 4),
                     "R2": round(res["pooled"]["R2"], 4)})
        np.save(f"outputs/p5/final_{MODE}_{name}.npy",
                {"origins": common, "pred": p_common}, allow_pickle=True)

    for name, p in stateless(h, pos).items():
        ok, res = dual_path_check(name, T, p, MODE)
        ok_all &= ok
        rows.append({"model": name, "n": len(common),
                     "MAE_pooled": round(res["pooled"]["MAE"], 4),
                     "MAE_horizon_mean": round(res["horizon"]["MAE"], 4),
                     "RMSE": round(res["pooled"]["RMSE"], 4),
                     "sMAPE": round(res["pooled"]["sMAPE"], 4),
                     "R2": round(res["pooled"]["R2"], 4)})

    df = pd.DataFrame(rows).sort_values("MAE_pooled")
    df.to_csv(f"outputs/p5/compare_{MODE}.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"双路径: {'全部一致' if ok_all else '不一致！'} | 表已存 outputs/p5/compare_{MODE}.csv")
    sys.exit(0 if ok_all else 2)


if __name__ == "__main__":
    main()
