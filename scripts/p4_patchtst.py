# -*- coding: utf-8 -*-
"""P4 PatchTST（GPU，neuralforecast 实现）。

硬约束落实：
- 13a：patch_len=16 / stride=8 / revin=True 显式传入（库默认值恰好一致，已预检记录）；
  13b：loss=MSE()（库默认 MAE，必须显式覆盖）。
- 用户 GPU 规约：① 训练前验证 torch.cuda.is_available()，不可用直接退出（禁止静默回退 CPU）；
  ② batch_size=256（≤256 上限）；③ 每 epoch 打印并追加写 training_log.csv：
  epoch 耗时、max_memory_allocated、train/val loss、global_step、device；
  预测经 cross_validation 回到 CPU（pandas/numpy）并 torch.cuda.empty_cache()。
- 评估：mode=val 用于调参；mode=test 对最优配置只评一次（硬约束 12）。
- 分段约束：与 p3_dlinear 相同的 n_win 上限（训练须容纳 input_size+val+h），
  保证段短时不伪造训练窗口。
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import H, L, SEED, load_hourly, metrics_pandas, time_split, window_origins
from p3_run import dual_path_check

warnings.filterwarnings("ignore")
os.makedirs("outputs/p4", exist_ok=True)
LOG_PATH = "training_log.csv"
CONFIG_PATH = "outputs/p4/patchtst_runs.jsonl"

import pytorch_lightning as pl
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import PatchTST

BATCH = 256  # 用户上限


class EpochLogger(pl.Callback):
    """逐 epoch 记录：耗时 / 显存峰值 / loss，写 training_log.csv 并打印。"""

    def __init__(self, run_id, device):
        self.run_id = run_id
        self.device = device
        self._t0 = None
        if not os.path.exists(LOG_PATH):
            with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["run_id", "device", "epoch", "global_step", "epoch_time_s",
                     "max_mem_allocated_mb", "train_loss", "val_loss", "ts"])

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    def on_train_epoch_end(self, trainer, pl_module):
        dt = time.perf_counter() - self._t0
        mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        m = trainer.logged_metrics
        tl = m.get("train_loss_epoch", m.get("train_loss", m.get("train_loss_step", float("nan"))))
        vl = m.get("val_loss", m.get("val_loss_epoch", float("nan")))
        row = [self.run_id, self.device, trainer.current_epoch, int(trainer.global_step),
               round(dt, 3), round(mem, 1), float(tl), float(vl),
               time.strftime("%Y-%m-%d %H:%M:%S")]
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        print(f"  [epoch {trainer.current_epoch:3d}] {dt:6.2f}s mem={mem:7.1f}MB "
              f"train_loss={row[6]:.4f} val_loss={row[7]:.4f}")


def make_model(lr, dropout, max_steps):
    return PatchTST(
        h=H, input_size=L,
        patch_len=16, stride=8, revin=True,          # 13a：论文 §3.1
        n_heads=16, encoder_layers=3, dropout=dropout, fc_dropout=dropout,
        loss=MSE(),                                   # 13b：训练损失 MSE
        learning_rate=lr, batch_size=BATCH,
        max_steps=max_steps, early_stop_patience_steps=500,
        scaler_type="identity", random_seed=SEED,
        enable_progress_bar=False,
        callbacks=[EpochLogger(os.environ.get("RUN_ID", "run"), os.environ.get("DEVICE", "cuda"))],
    )


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


def run_eval(h, eval_times, lr, dropout, max_steps, run_id):
    """逐段 cross_validation，训练数据严格在首个 cutoff 之前。返回 pred DataFrame(index=origin)。"""
    mats = []
    for s, e in segments_of(h):
        seg_times = h.index[s:e]
        seg_eval = eval_times.intersection(seg_times)
        n_win = min(len(seg_eval), len(seg) - L - H - 24)
        if n_win <= 0:
            continue
        seg_eval = seg_eval[-n_win:]
        seg = h.iloc[s:e]
        ydf = pd.DataFrame({"unique_id": "hhs", "ds": seg.index, "y": seg["gap_kw"].to_numpy()})
        val_size = min(24 * 30, max(24, (len(seg) - n_win - L - H) // 3))
        model = make_model(lr, dropout, max_steps)
        nf = NeuralForecast(models=[model], freq="h")
        cv = nf.cross_validation(df=ydf, h=H, n_windows=n_win, step_size=1, val_size=val_size)
        cv = cv.sort_values(["cutoff", "ds"])
        pred_mat = np.stack(cv.groupby("cutoff", sort=True)["PatchTST"].apply(np.asarray).to_numpy())
        cutoffs = pd.DatetimeIndex(cv["cutoff"].sort_values().unique())
        assert (cutoffs == seg_eval).all(), "cutoff 与 origins 未对齐"
        mats.append(pd.DataFrame(pred_mat, index=cutoffs))
        del cv, pred_mat, nf, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  segment {seg.index[0]}~{seg.index[-1]}: n_windows={n_win}")
    return pd.concat(mats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "test"], required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--run-id", default="patchtst_base")
    ap.add_argument("--bench-cpu", action="store_true",
                    help="CPU 参考基准：同配置 200 步，用于 GPU 加速比核算")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available()=False，按规约禁止静默回退 CPU。先解决 CUDA 环境。")
        sys.exit(3)
    print(f"torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.environ["RUN_ID"] = args.run_id
    os.environ["DEVICE"] = "cpu" if args.bench_cpu else "cuda"

    h = load_hourly()
    origins = window_origins(h)
    tr, va, te, cut_val, cut_test = time_split(h.index[origins])
    y = h["gap_kw"].to_numpy()

    if args.bench_cpu:
        # 加速比参考：CPU 同配置 200 步（epoch 日志 device=cpu）
        seg = h.iloc[segments_of(h)[0][0]:segments_of(h)[0][1]]
        ydf = pd.DataFrame({"unique_id": "hhs", "ds": seg.index, "y": seg["gap_kw"].to_numpy()})
        model = make_model(args.lr, args.dropout, 200)
        model.trainer_kwargs["accelerator"] = "cpu"
        model.trainer_kwargs["devices"] = 1
        nf = NeuralForecast(models=[model], freq="h")
        nf.fit(df=ydf, val_size=min(24 * 30, (len(seg) - L - H) // 3))
        print("CPU 基准完成（200 步），见 training_log.csv 中 device=cpu 行")
        return

    if args.mode == "val":
        eval_times = h.index[origins[va]]
    else:
        eval_times = h.index[origins[te]]
    print(f"[{args.run_id}] mode={args.mode} lr={args.lr} dropout={args.dropout} "
          f"max_steps={args.max_steps} eval_origins={len(eval_times)}")
    pred = run_eval(h, eval_times, args.lr, args.dropout, args.max_steps, args.run_id)

    idx = h.index[origins]
    common = pred.index
    sel = np.searchsorted(idx, common)
    T = np.stack([y[origins[k] + j] for k in sel for j in range(1, H + 1)])
    T = T.reshape(len(common), H)
    ok, res = dual_path_check(args.run_id, T, pred.to_numpy(), args.mode)
    print(f"  {args.run_id} ({args.mode}) MAE={res['pooled']['MAE']:.4f} "
          f"RMSE={res['pooled']['RMSE']:.4f} sMAPE={res['pooled']['sMAPE']:.4f} "
          f"R2={res['pooled']['R2']:.4f} (n={len(common)})")
    np.save(f"outputs/p4/pred_{args.mode}_{args.run_id}.npy",
            {"origins": common, "pred": pred.to_numpy()}, allow_pickle=True)
    with open(CONFIG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": args.run_id, "mode": args.mode, "lr": args.lr,
                            "dropout": args.dropout, "max_steps": args.max_steps,
                            "pooled": res["pooled"], "horizon_MAE": res["horizon"]["MAE"],
                            "n": len(common)}) + "\n")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
