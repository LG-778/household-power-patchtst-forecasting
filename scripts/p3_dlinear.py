# -*- coding: utf-8 -*-
"""P3 DLinear 基线（neuralforecast 实现，不手撸；与门禁/全量共用，仅日期参数不同）。

设计要点：
- 训练损失 MSE（与论文口径一致；PatchTST 同，见硬约束 13b）
- 输入不经过全局 scaler：DLinear/PatchTST 的归一化由模型内部机制完成（硬约束 7 分层）
- 评估口径与其余基线一致：val+test origins 上 pooled 主指标
- 分段处理：cross_validation 在每个月度有效段内独立进行（时间顺序天然保持，不 shuffle）
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import H, L, SEED, load_hourly, metrics_pandas, time_split, window_origins
from p3_run import dual_path_check

OUTDIR = "outputs/p3"
os.makedirs(OUTDIR, exist_ok=True)
warnings.filterwarnings("ignore")

import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import DLinear


def segments_of(h):
    valid = h["gap_kw"].notna().to_numpy()
    d = np.diff(valid.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if valid[0]:
        starts = [0] + starts
    if valid[-1]:
        ends = ends + [len(valid)]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= L + H + 48]


def run_dlinear(h, origins, eval_mask, tag):
    """在 eval origins 上产生 DLinear 预测。逐段 cross_validation（cutoff=origin 对齐）。"""
    origin_times = h.index[origins]
    eval_times = origin_times[eval_mask]
    mats = []
    for s, e in segments_of(h):
        seg_times = h.index[s:e]
        seg_eval = eval_times.intersection(seg_times)
        # 训练窗口必须容纳 input_size + 至少 24h early-stop val：超出部分不评估（各模型共用同一 origin 集，公平）
        n_win = min(len(seg_eval), (e - s) - L - H - 24)
        if n_win <= 0:
            print(f"  segment {seg_times[0]}~{seg_times[-1]}: 过短，跳过")
            continue
        if n_win < len(seg_eval):
            print(f"  segment {seg.index[0]}~{seg.index[-1]}: 评估窗口 {len(seg_eval)}→{n_win}"
                  f"（训练长度约束，末段 {len(seg_eval)-n_win} 个 origin 不评估）")
        seg_eval = seg_eval[-n_win:]
        seg = h.iloc[s:e]
        ydf = pd.DataFrame({"unique_id": "hhs", "ds": seg.index,
                            "y": seg["gap_kw"].to_numpy()})
        model = DLinear(h=H, input_size=L, loss=MSE(),
                        max_steps=300, early_stop_patience_steps=20,
                        random_seed=SEED, enable_progress_bar=False)
        nf = NeuralForecast(models=[model], freq="h")
        val_size = min(24 * 30, max(24, (len(seg) - n_win - L - H) // 3))
        cv = nf.cross_validation(df=ydf, h=H, n_windows=n_win, step_size=1,
                                 val_size=val_size)
        cv = cv.sort_values(["cutoff", "ds"])
        assert (cv.groupby("cutoff").size() == H).all(), "每 cutoff 应恰有 H 行"
        assert (cv.groupby("cutoff")["ds"].diff().dropna() == pd.Timedelta("1h")).all()
        pred_mat = cv.groupby("cutoff", sort=True)["DLinear"].apply(np.asarray)
        pred_mat = np.stack(pred_mat.to_numpy())
        seg_eval = pd.DatetimeIndex(cv["cutoff"].sort_values().unique())
        assert len(seg_eval) == n_win
        mats.append(pd.DataFrame(pred_mat, index=seg_eval))
        print(f"  segment {seg.index[0]}~{seg.index[-1]}: n_windows={n_win}")
    return pd.concat(mats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    h = load_hourly(args.start, args.end)
    origins = window_origins(h)
    tr, va, te, cut_val, cut_test = time_split(h.index[origins])
    eval_mask = np.zeros(len(origins), dtype=bool)
    eval_mask[np.r_[va, te]] = True

    print(f"[{args.tag}] DLinear: origins={len(origins)}, eval={eval_mask.sum()}")
    mat = run_dlinear(h, origins, eval_mask, args.tag)
    T = np.stack([h["gap_kw"].to_numpy()[origins + k] for k in range(1, H + 1)], axis=1)
    idx = h.index[origins]
    common = mat.index.intersection(idx[eval_mask])
    yt = T[np.searchsorted(idx, common)]
    yp = mat.loc[common].to_numpy()
    ok, res = dual_path_check("dlinear", yt, yp, args.tag)
    print(f"  dlinear  MAE={res['pooled']['MAE']:.4f} RMSE={res['pooled']['RMSE']:.4f} "
          f"sMAPE={res['pooled']['sMAPE']:.4f} R2={res['pooled']['R2']:.4f} (n={len(common)})")
    np.save(f"{OUTDIR}/dlinear_pred_{args.tag}.npy", {"origins": common, "pred": yp})
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
