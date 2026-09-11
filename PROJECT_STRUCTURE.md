# GitHub 推送准备 · 项目结构盘点（只读）

生成时间：2026-09-11 ｜ 项目根：本仓库根目录 ｜ 约束：盘点为只读操作，未整理/复制/推送，未执行任何 git 命令

> 两点说明：① `lightning_logs/`（pytorch-lightning 每块训练自动产生，859 个文件、结构完全同构）按目录聚合为一条汇总，不逐文件列出；② 树中缩进即目录层级，文件行下一行为内容注释。

```
  README.md — 20.3 KB
    └─ 项目 README 全稿（14 节）
  decisions.md — 16.0 KB
    └─ 全程决策+理由留痕
  github_staging_plan.md — 7.7 KB
    └─ 本盘点文件
  training_log.csv — 19.7 MB
    └─ 逐 epoch 训练日志（19 万行）
  项目阶段与步骤详解.md — 16.8 KB
    └─ P1a–P5 逐步骤过程文档
  .idea/
    └─ PyCharm 项目配置（IDE 自动生成，推送候选：排除）
    .gitignore — 0 B
      └─ PyCharm 配置
    household_power_predict_PatchTST.iml — 291 B
      └─ PyCharm 配置
    misc.xml — 302 B
      └─ PyCharm 配置
    modules.xml — 323 B
      └─ PyCharm 配置
    workspace.xml — 1.6 KB
      └─ PyCharm 配置
    .idea/inspectionProfiles/
      Project_Default.xml — 1.4 KB
        └─ PyCharm 配置
      profiles_settings.xml — 174 B
        └─ PyCharm 配置
  data/
    └─ 数据目录（原始 txt + 清洗 parquet）
    household_power_consumption.txt — 126.8 MB
      └─ 原始 UCI 分钟级数据（2,075,259 行）
    data/processed/
      └─ P1b 清洗产物缓存
      hourly.parquet — 624.4 KB
        └─ P1b 小时级重采样产物
      minute_clean.parquet — 22.4 MB
        └─ P1b 清洗后分钟级数据缓存
  outputs/
    └─ 全部评估产物
    p1a_fingerprint.txt — 4.0 KB
      └─ P1a 指纹报告留档
    p1b_segment_stats.csv — 570 B
      └─ P1b 分段统计（7 段含窗口数）
    outputs/eda/
      └─ P2 EDA 四图
      p2_1_intraday_profile.png — 75.6 KB
        └─ 图
      p2_2_yearly_drift.png — 116.4 KB
        └─ 图
      p2_3_gaps_segments.png — 121.1 KB
        └─ 图
      p2_4_power_factor.png — 23.8 KB
        └─ 图
    outputs/p3/
      └─ P3 基线组产物（门禁/全量结果表与预测）
      baseline_results_full.csv — 292 B
        └─ 结果表
      baseline_results_gate.csv — 290 B
        └─ 结果表
      dlinear_pred_full.npy — 991.4 KB
      dlinear_pred_gate.npy — 13.0 KB
      xgb_pred_test_full.npy — 461.8 KB
      xgb_pred_test_gate.npy — 7.0 KB
      y_test_full.csv — 1.2 MB
        └─ 结果表
      y_test_gate.csv — 18.2 KB
        └─ 结果表
    outputs/p4/
      └─ 滚动评估逐块日志
      dlinear_test.log — 41.6 KB
        └─ 滚动评估逐块日志
      dlinear_val.log — 41.6 KB
        └─ 滚动评估逐块日志
      ptst_test.log — 3.8 MB
        └─ 滚动评估逐块日志
      ptst_val_lr5e4.log — 3.8 MB
        └─ 滚动评估逐块日志
      ptst_val_p2412.log — 3.8 MB
        └─ 滚动评估逐块日志
      ptst_val_rerun.log — 3.8 MB
        └─ 滚动评估逐块日志
      xgb_test.log — 103 B
        └─ 滚动评估逐块日志
      xgb_val_rerun.log — 101 B
        └─ 滚动评估逐块日志
    outputs/p5/
      └─ P5 评估产物核心区（推送候选）
      ablation_patch_24_12_val.csv — 179 B
        └─ 结果表
      compare_test.csv — 372 B
        └─ 结果表
      dlinear_base_test_pred.npy — 466.8 KB
        └─ test 滚动预测数组
      dlinear_base_val_pred.npy — 467.2 KB
        └─ val 滚动预测数组
      error_attribution.csv — 1.6 KB
        └─ 结果表
      error_by_hour.png — 109.5 KB
        └─ 图
      error_by_period.png — 42.5 KB
        └─ 图
      final_test_dlinear_base.npy — 466.8 KB
        └─ test 预测（主会话整理副本，与 dlinear_base_test_pred.npy 同 md5）
      final_test_ptst_lr5e4.npy — 466.8 KB
        └─ test 预测（主会话整理副本，与 ptst_lr5e4_test_pred.npy 同 md5）
      final_test_xgb_base.npy — 466.8 KB
        └─ test 预测（主会话整理：XGB 共同 origins 子集）
      granularity_bridge.csv — 265 B
        └─ 结果表
      horizon_mae_dual_curve.png — 124.9 KB
        └─ 图
      horizon_mae_test.csv — 1.5 KB
        └─ 结果表
      new_old_comparison.csv — 2.0 KB
        └─ 结果表
      paper_claims_check.csv — 2.5 KB
        └─ 结果表
      ptst_lr1e4_val_pred.npy — 467.2 KB
        └─ val 滚动预测数组
      ptst_lr5e4_p2412_val_pred.npy — 467.2 KB
        └─ val 滚动预测数组
      ptst_lr5e4_test_pred.npy — 466.8 KB
        └─ test 滚动预测数组
      ptst_lr5e4_val_pred.npy — 467.2 KB
        └─ val 滚动预测数组
      xgb_base_test_pred.npy — 500.9 KB
        └─ test 滚动预测数组
      xgb_base_val_pred.npy — 501.3 KB
        └─ val 滚动预测数组
      xgb_feature_importance.csv — 387 B
        └─ 结果表
  scripts/
    └─ 全部代码
    eval_rolling.py — 8.8 KB
      └─ 滚动 168h 块评估主脚本（含消融参数）
    p1a_fingerprint.py — 5.5 KB
      └─ P1a 统计指纹
    p1b_assert.py — 7.2 KB
      └─ P1b 独立断言（15 项）
    p1b_clean.py — 7.9 KB
      └─ P1b 确定性清洗
    p2_eda.py — 6.1 KB
      └─ P2 EDA 绘图
    p3_common.py — 5.0 KB
      └─ 公共模块（窗口/特征/目标/切分/指标）
    p3_dlinear.py — 4.8 KB
      └─ P3 DLinear 初版（已重构弃用，留档）
    p3_run.py — 9.1 KB
      └─ P3 基线组+rolling CV+双路径核对
    p4_patchtst.py — 8.5 KB
      └─ P4 PatchTST 构造器+EpochLogger+CPU 基准
    p5_compare.py — 3.2 KB
      └─ P5 统一对比（共同 origins+双路径核对）
  旧项目参考/
    └─ 旧日级 XGBoost 项目参照（.py + README，白名单来源；.ipynb 只存不读）
    README.md — 8.0 KB
    区域用电负荷数据分析.ipynb — 1.4 MB
      └─ 旧项目 notebook（约束：只存不读）
    区域用电负荷数据分析.py — 34.2 KB
```

## 汇总一：体积

- 项目总体积（含 lightning_logs 聚合 28.8 MB）：**223.7 MB**
- data/ 体积：**149.8 MB**（其中原始 txt 126.8 MB）
- outputs/ 体积：**23.7 MB**

## 汇总二：大于 1MB 的文件清单

- `data/household_power_consumption.txt` — 126.8 MB
- `lightning_logs/（聚合 859 个文件，pytorch-lightning 自动日志）` — 28.8 MB
- `data/processed/minute_clean.parquet` — 22.4 MB
- `training_log.csv` — 19.7 MB
- `outputs/p4/ptst_val_p2412.log` — 3.8 MB
- `outputs/p4/ptst_test.log` — 3.8 MB
- `outputs/p4/ptst_val_lr5e4.log` — 3.8 MB
- `outputs/p4/ptst_val_rerun.log` — 3.8 MB
- `旧项目参考/区域用电负荷数据分析.ipynb` — 1.4 MB
- `outputs/p3/y_test_full.csv` — 1.2 MB

## 汇总三：Git 相关文件情况

- `.git/` 目录：**不存在（当前不是 git 仓库）**
- `.gitignore`：根目录**不存在**（仅 `.idea/.gitignore` 0 B，PyCharm 自动生成）
- `requirements.txt`：**不存在**（环境版本清单在 README §13）
- `README.md`：存在（20.3 KB，项目 README 全稿）

## git 状态检查

- 是否 git 仓库：否（`.git/` 不存在，纯文件系统判断）
- 未提交改动 / remote 配置：本步禁止 git 命令，且非仓库无从谈起；第二步建仓库后统一核查。
