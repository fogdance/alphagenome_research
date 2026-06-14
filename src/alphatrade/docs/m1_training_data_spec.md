# M1 训练数据规范（Training Data Spec）

生成时间: 2026-02-28  
目标版本: M1（基于 M0.1 的 pipeline，扩展到 30 品种 + 标准化 + 多品种采样策略）

---

## 0. 背景与动机

M0.1 已完成最小闭环：
- 从 Archive 读取 1m bars（manifest 驱动、去重、校验）
- 生成 Canonical Bars（gap 分段 + 基础特征）
- 生成 Sample Index（lookback/stride/horizons、时间切分、同 segment 约束）
- Baseline smoke train（在线切片，产出 metrics.json + run.md）

M1 要解决的问题：
1) **规模扩展**：从 2 个品种扩到 30 个“品种级连续合约”（例如 `DCE.JM`, `SHFE.RB` 这种）
2) **跨品种可训练性**：加入 **feature/label 标准化**，避免某些品种主导梯度
3) **可复现 + 可迭代**：数据版本指纹、统计报告、训练 metrics 结构化输出，便于 agent 循环迭代

---

## 1. 范围（Scope）

### In-scope
- 数据源：JUEJIN 1m archive（parquet + manifest jsonl）
- 对象：**连续合约（品种级）**为主（例如 `DCE.JM`, `CZCE.FG`, `CFFEX.IC`）
- 产物：
    - `data/processed/m1/{symbol}/bars.parquet`（canonical）
    - `data/processed/m1/{symbol}/index_{train,val,test}.parquet`（样本索引）
    - `data/processed/m1/_stats/*.json`（标准化统计与数据集指纹）
    - `$ALPHATRADE_RUNS_ROOT/reports/m1_*.md / $ALPHATRADE_RUNS_ROOT/reports/m1_*.json`（质量与训练报告）

### Out-of-scope（先不做）
- continuous->real 的 roll 规则建模（可作为 M0.2 / M2）
- 回测与交易层评估（M2+）
- 复杂模型结构（Transformer/多模态等，M2+）

---

## 2. 输入数据（Archive）约定

### 2.1 Manifest（JSONL）
每行代表一个 parquet 月文件，字段示例：
- `symbol`, `exchange`, `underlying`, `year`, `month`, `rows`
- `min_eob`, `max_eob`
- `sha256`
- `path`（相对 archive_dir）

### 2.2 Parquet（raw bars）最低要求列
必须包含（按 M0.1 经验）：
- `bob`, `eob` (datetime)
- `open`, `high`, `low`, `close` (float)
- `volume`, `position` (int/float，允许后续 cast)
- `symbol`（可选；如无则由目录推断）

### 2.3 原始数据基本清洗规则（与 M0.1 一致）
1) 按 `eob` 排序
2) `eob` 去重：同 `eob` 保留 last
3) 必要校验：
    - `eob` 单调递增（去重后）
    - 不允许关键列缺失（OHLC/volume/eob）
4) dtype 统一（详见 canonical）

---

## 3. Symbol Universe（30 品种）定义

### 3.1 目标集合
- 30 个“品种级连续合约”，以 `EXCHANGE.VAR` 形式表达（例如 `DCE.JM`）
- 避免：
    - `*00/*01/*02` 这类“连一/连二”如果你们不想训练（可配置）
    - 覆盖月份过短 / rows 过少的 symbol（可配置门槛）

### 3.2 选择策略（两种）
- **显式列表**：`symbols: [...]`（最可控，推荐）
- **manifest 过滤**（可选实现）：按 `min_months / min_rows / allowlist/denylist` 自动筛选

M1 默认：先用“显式列表”，减少自动筛选带来的不可控差异。

---

## 4. Canonical Bars（标准化 bars）规范

### 4.1 输出路径
`data/processed/m1/{symbol}/bars.parquet`

### 4.2 Canonical schema（建议列）
必备列：
- `eob` (datetime64[ns])  # bar end time
- `open, high, low, close` (float32)
- `volume, position` (int32/int64 视数据范围)
- `segment_id` (int32)    # gap 分段 id

特征列（M1 允许在 M0.1 基础上增量）：
- 基础（M0.1 同款）：
    - `ret_1m`：close 的 1m return（可选 log return）
    - `hl_range`： (high-low)/close
    - `co_change`： (close-open)/open
    - `vol_log1p`：log1p(volume)
    - `pos_log1p`：log1p(position)
- 时间相位：
    - `minute_sin`, `minute_cos`（基于分钟-of-day 的周期编码）
    - `is_open`（是否处于交易时段；M1 可先延续 M0.1 逻辑）

可选 rolling 特征（M1 建议先留接口，默认不启用或只启用少量）：
- `ret_mean_w{w}`, `ret_std_w{w}`
- `vol_sum_w{w}`
- `range_mean_w{w}`

### 4.3 segment 定义（与 M0.1 一致）
- `gap_seconds = 1800`（30 分钟）
- 若相邻两条 bar 的 `eob` 差 > gap_seconds，则新开 segment
- **采样窗口与标签点必须在同一 segment**（见 sampling）

---

## 5. Sample Index（训练样本索引）规范

### 5.1 输出路径
- `data/processed/m1/{symbol}/index_train.parquet`
- `data/processed/m1/{symbol}/index_val.parquet`
- `data/processed/m1/{symbol}/index_test.parquet`

### 5.2 Index schema（建议列）
- `t` (int32) ：在 `bars.parquet` 中的行号（anchor）
- `eob` (datetime64[ns])：anchor 时刻
- `segment_id` (int32)
- `y_h{H}` (float32)：每个 horizon 的标签（例如 `y_h1, y_h5, y_h20, y_h60`）

可选列（M1 推荐加）：
- `symbol`（字符串）或 `symbol_id`（int）
- `split`（train/val/test）

> 重要：dataloader **在线切片**：用 `t` 在 `bars.parquet` 上切 `[t-L+1, t]`，不要把所有窗口物化成巨型文件。

---

## 6. 采样与标签（Sampling & Labels）

### 6.1 采样参数
- `lookback = L`（窗口长度）
- `stride`（anchor 采样步长）
- `horizons = [h1, h2, ...]`
- `require_same_segment = true`

### 6.2 样本有效性约束
对每个 anchor `t`：
- window: `[t-L+1, ..., t]`
- label point: `t+h`
- 必须满足：
    1) `t-L+1 >= 0`
    2) `t+h < len(bars)`
    3) window 内与 `t+h` 的 `segment_id` 全部相同（禁止跨 gap）

### 6.3 标签定义（默认）
- `y_h = close[t+h] / close[t] - 1`（future return）
- 可配置为 log return：
    - `y_h = log(close[t+h]) - log(close[t])`

---

## 7. 时间切分（Time Split）与防泄漏

### 7.1 Split 定义（左闭右开）
- train: `[train_start, train_end)`
- val:   `[val_start, val_end)`
- test:  `[test_start, test_end)`

### 7.2 防泄漏规则
- split 归属按 **anchor 的 eob(t)** 决定
- 但必须额外保证 **label 点 eob(t+h)** 仍落在同一 split 内
    - 若 `t` 在 train，但 `t+h` 跨到了 val，则该样本丢弃（或可配置归入 train，但默认丢弃更安全）

---

## 8. 标准化（Normalization）——M1 核心

M0.1 smoke train 已可跑，但跨品种训练时：
- 不同品种波动率/成交量尺度差异巨大
- 需要标准化降低训练不稳定与梯度被单一品种主导

### 8.1 Feature 标准化
- `fit_split = train`
- scope：
    - `per_symbol`：每个 symbol 一套 mean/std（默认推荐）
    - `global`：所有 symbol 合并统计一套 mean/std（可选）
- method：
    - `zscore`: (x - mean) / (std + eps)

产物（建议）：
- `data/processed/m1/_stats/feature_scaler.json`

### 8.2 Label 标准化（可选但推荐）
- 同样 `fit_split = train`
- scope 建议 `per_symbol_per_horizon`
- method `zscore`
- 目的：不同品种/不同 horizon 的 label 方差差异显著，标准化后 multi-task 更稳定

产物（建议）：
- `data/processed/m1/_stats/label_scaler.json`

---

## 9. 多品种混合采样策略（Batch Mixing）

M1 必须明确 batch 里不同 symbol 的占比，否则：
- 样本多的品种会主导训练
- 小样本品种几乎学不到

推荐策略（可配置）：
1) `uniform_by_symbol`：每个 batch 随机选 symbol，再从该 symbol 的 index 采样
2) `sqrt_weighted`：权重 ∝ sqrt(N_symbol)（折中）
3) `proportional`：权重 ∝ N_symbol（不推荐，容易被头部品种压制其它）

---

## 10. 数据集版本与可复现性（Dataset Fingerprint）

M1 要求输出 dataset 指纹（写入 metrics.json / stats.json）：
- git_sha（代码版本）
- dataset_config_sha（m1.yaml 的 hash）
- manifest_sha 或 manifest_path + 最后修改时间
- symbols 列表（排序后）
- 关键参数：L / stride / horizons / time_split / normalization scope

---

## 11. 质量门槛（Acceptance Gates）

对每个 symbol（至少）：
- Archive read：去重后 eob 单调递增、关键列无缺失
- Canonical：segment 统计合理（非 0），特征无 NaN/inf
- Index：train/val/test 样本数非 0（允许个别品种被过滤掉，但需要报告原因）
- Standardization：scaler 文件生成；std 不为 0；异常值比例受控（可选 clip）

整体（至少）：
- 能一条命令生成全量 index（或分批执行但可复现）
- 能跑 smoke train 产出：
    - `$ALPHATRADE_RUNS_ROOT/reports/m1_train_metrics.json`
    - `$ALPHATRADE_RUNS_ROOT/reports/m1_train_run.md`

---

## 12. 配置文件映射（m1.yaml）

本 spec 的所有可变项都在：
- `configs/dataset/m1.yaml`

脚本读取该 config 后：
- build_canonical_bars
- build_sample_index
- train_m1_baseline（或复用通用 train）

---

## 13. Open Questions（留给 M1 实现时确认）

1) `is_open` 的定义：是否要引入交易时段表（gm session）来做精确定义？
2) 夜盘与跨午夜：是否保留全时段，还是只训练日盘？
3) rolling 特征：M1 是否默认启用一小组（如 w=[5,20]），还是先留空？
4) 数据覆盖差异：某些品种上市晚、覆盖短，是否做自动剔除？门槛是多少？