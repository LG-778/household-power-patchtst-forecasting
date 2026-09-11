# 数据获取

本项目使用 UCI Machine Learning Repository 的 **Individual household electric power consumption** 数据集。

- **官方下载地址**：https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- **放置路径**：下载解压后，将 `household_power_consumption.txt`（约 **127 MB**，2,075,259 行分钟级数据）放到本目录（`data/`）下，即 `data/household_power_consumption.txt`。
- **注意**：原始数据文件不入库（超过 GitHub 单文件限制且体积过大），请从官方页面自行获取。

## 清洗产物复现

`minute_clean.parquet` 与 `hourly.parquet` 均不入库。拿到原始 txt 后，按 README §13 顺序运行：

```bash
python scripts/p1a_fingerprint.py   # 统计指纹（只输出统计量）
python scripts/p1b_clean.py         # 确定性清洗 + 小时级重采样
python scripts/p1b_assert.py        # 独立不变量断言（15 项）
```

即可在 `data/processed/` 下复现全部清洗产物（清洗规则与分段统计见 README §2）。
