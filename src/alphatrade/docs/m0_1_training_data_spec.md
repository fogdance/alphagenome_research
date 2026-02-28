# M0.1 训练数据标准格式与切分策略（Training Data Spec）

更新时间：2026-02-28  
适用范围：仅用于 M0.1（2 个品种）验证“从 archive 到可训练数据”的最小闭环；后续 M0.2/M1 会在此基础上扩展到更多品种与更复杂模型。

---

## 1. 目标与非目标

### 目标
- 定义一套稳定、可复用的训练数据标准格式（文件结构 + schema + label 定义）。
- 训练集/验证集/测试集按时间切分，严格避免时间泄漏。
- 训练时在线切片（dataloader 根据 index 取窗口），避免预先物化“巨大窗口文件”。

### 非目标（M0.1 不做）
- 不做回测、策略执行、交易成本建模。
- 不做 continuous->real 映射与换月逻辑（先基于连续主力的时间序列做建模验证）。

---

## 2. 输入数据（Archive）

Archive 由 backfill job 写入 parquet 分区，目录约定：
- `provider=JUEJIN/freq=1m/symbol=<SYMBOL>/year=YYYY/month=MM/data.parquet`
- 同时写入 `_manifest/<run_id>.jsonl`（每行记录一个月文件的 rows/min_eob/max_eob/path 等）。

Archive Reader 的职责：
- 按 manifest（或自动发现）拼接多月 parquet
- `eob` 去重（同一 timestamp 多条，保留 last）
- 校验：无缺失值、`eob` 单调递增、dtype 合理

---

## 3. 中间产物：Canonical Bars（标准 K 线 + 特征 + 分段）

### 输出路径
- `data/processed/m0_1/{symbol}/bars.parquet`

### 行粒度
- 一行 = 1 分钟 bar（1m）

### 必备字段（建议 schema）
- 时间与索引
    - `eob` (timestamp)：bar 结束时间
    - `bob` (timestamp)：bar 开始时间（可选，但建议保留）
    - `segment_id` (int)：gap-based segment 分段 ID（用于避免跨 gap 采样）
- 原始行情
    - `open, high, low, close` (float32)
    - `volume` (int32)
    - `position`/`open_interest` (int32)
- 特征（float32，除非特别说明）
    - `ret_1m`：log(close[t]/close[t-1])
    - `hl_range`：(high-low)/max(eps, close)
    - `co_change`：(close-open)/max(eps, open)
    - `vol_log1p`：log(1+volume)
    - `pos_log1p`：log(1+position)
    - `minute_sin, minute_cos`：分钟相位特征（eob 的分钟/日内周期）
    - `is_open`：是否处于交易时段（0/1；实现可按交易所 session 或简单规则）

### Segment 分段规则（gap-based）
- 当相邻两条 bar 的 `eob` 时间差 > `gap_seconds`（配置项，M0.1=1800s）时，开启新 segment。
- segment 的目的：保证窗口与 label 不跨越明显断点（如收盘到开盘）。

---

## 4. 训练样本索引（Sample Index）：训练/验证/测试

### 输出路径
- `data/processed/m0_1/{symbol}/index_train.parquet`
- `data/processed/m0_1/{symbol}/index_val.parquet`
- `data/processed/m0_1/{symbol}/index_test.parquet`

### 样本定义
一个样本由以下元素组成：
- 输入窗口：长度 `L=lookback`，窗口结束点为 `t`
    - 使用 canonical bars 的特征列构成 `X[t-L+1 : t]`，shape = `[L, F]`
- 多 horizon label：对每个 `h in horizons`，预测未来收益
    - `y_h = log(close[t+h]/close[t])`
- 约束：窗口与 label endpoint 必须落在同一个 `segment_id` 内（`require_same_segment=true`）

### 采样步长（stride）
- 只对满足条件的 anchor 点 t 做采样；
- anchor 点按 `stride` 下采样（例如 stride=5 表示每 5 分钟采一个样本），减少高度相关样本。

### Index 表建议列
- `symbol` (string)
- `t_end` (int64)：窗口结束行号（指向 bars.parquet 的 row index）
- `eob_end` (timestamp)：bars[t_end].eob
- `segment_id` (int32)
- `y_h1, y_h5, y_h20, y_h60` (float32) 或用数组列

### 时间切分（避免泄漏）
按 `eob_end` 的日期区间划分 train/val/test（配置写死）：
- train: 2018-01-01 ~ 2024-06-30
- val:   2024-07-01 ~ 2024-12-31
- test:  2025-01-01 ~ 2026-01-01

---

## 5. 训练时 Dataloader 约定（关键）

- 训练脚本只读取：
    - `index_{split}.parquet`（确定每个样本的 t_end 与 labels）
    - `bars.parquet`（按需切片取窗口）
- 禁止在数据准备阶段把所有窗口物化成巨大的训练文件（成本高、更新慢、难 debug）。

---

## 6. 可复现性与版本记录（必须写入 metrics）

训练输出 `reports/m0_1_train_metrics.json` 至少包含：
- run：run_id / git_sha / created_at / device / seed
- dataset：name / config_path / lookback / stride / horizons / features / symbols / 每个 symbol 样本数
- train/val：loss last + best

（人类可读报告另写 md）

---

## 7. 验收标准（M0.1）

- 一条命令跑完 baseline 训练，稳定产出：
    - `reports/m0_1_train_metrics.json`
    - `reports/m0_1_train_run.md`
- dataloader 在线切片工作正常（不物化窗口）
- train/val loss 正常收敛（smoke 200~1000 steps 即可）