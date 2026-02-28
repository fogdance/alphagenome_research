好，那我就按“你做中间人搬运”的模式来：我给 Claude Code 一份**里程碑 + 子任务清单**（每条尽量可直接开干），它跑完把产物/日志/关键指标回传给你，你再贴给我，我继续做下一轮设计/调参/排障。

下面这份清单默认你们走 **方案 A（主力连续，roll-safe 过滤）**，并且**尽量不依赖 MySQL**（能用 archive+gm map 就用本地 map）。

---

## Milestone 0：仓库内“训练数据标准化”最小闭环（2 个品种）

目标：只用 `DCE.JM` + `SHFE.RB` 两个连续品种，从 archive 生成可训练的 dataset，训练能跑通，eval_loss 明显下降，calibration 不再乱。

### M0.1 盘点 archive（输入清单）

**任务**

1. 读取 manifest（jsonl），列出：

    * 连续 symbol 列表（至少包含 DCE.JM / SHFE.RB）
    * 每个 symbol 的月数、总 rows、min/max eob
2. 校验 parquet 文件可读，列出每个 parquet 的列名（至少应有 bob/eob/open/high/low/close/volume/position）

**产物**

* `reports/m0_manifest_summary.md`（表格：symbol, months, rows, min_eob, max_eob）
* `reports/m0_parquet_schema.md`（列名 + dtype）

---

### M0.2 生成本地 continuous->real map（roll-safe 必备）

**任务（推荐不走 DB）**

1. 对每个连续 symbol（先 JM/RB），调用 GM `fut_get_continuous_contracts(csymbol, start, end)`
2. 保存为本地文件：`datasets/fut1m_v1/maps/{CSYMBOL}.parquet`

    * columns: `trading_date, real_symbol`
3. 如果 GM 不可用（token/环境问题），退化方案：从 DB 的 `fut_continuous_map_v2` 导出到同样格式（但优先本地化，训练不依赖 DB）

**产物**

* `datasets/fut1m_v1/maps/DCE.JM.parquet`
* `datasets/fut1m_v1/maps/SHFE.RB.parquet`
* `reports/m0_map_coverage.md`（map 覆盖的 trading_date 数、缺口）

---

### M0.3 从 archive 构建“序列缓存”（不要预展开 windows）

**任务**
为每个 symbol（JM/RB）生成 cache：

1. 读该 symbol 的所有月 parquet，按 `eob` 排序、去重
2. 转成 DataFrame（必须包含 `eob`）
3. 计算 8 个 feature：调用你现有 `preprocessing.FeaturePreprocessor.compute_features(...)`

    * 注意：minute_idx 的对齐（你现在用 trading_day_start=21:00；但如果你只用日盘，也可以设 09:00；先保持一致别乱改）
4. 输出 numpy 文件：

    * `cache/{symbol}.features.npy` → float32 `[T,8]`
    * `cache/{symbol}.close.npy` → float32 `[T]`
    * `cache/{symbol}.eob.npy` → int64 或 datetime64（建议 datetime64[ns]）
    * `cache/{symbol}.trading_date.npy`（用于 join map；如果 eob 可推 trading_date 也行）
5. 生成 `cache/{symbol}.roll_id.npy`

    * 用 trading_date join 本地 map 得到当日 real_symbol
    * 将 real_symbol 编成 int（每次 real_symbol 变化 roll_id+1），长度为 T
6. 写 `cache/{symbol}.meta.json`：T、min/max eob、缺失比例、roll 次数等

**产物**

* `datasets/fut1m_v1/cache/*.npy` + `*.meta.json`
* `reports/m0_cache_stats.md`（每个 symbol：T、日期范围、roll 次数、均匀性）

---

### M0.4 实现 roll-safe window sampler + 小训练跑通

**任务**

1. 新建 dataset 类（建议位置：`src/alphagenome_research/alphatrade/data/fut_archive_dataset.py`）
2. 核心采样逻辑：

    * 输入：symbol 列表、lookback L=4096、horizons=[1,5,20,60]
    * 随机抽 symbol，再抽 asof t
    * roll-safe 检查：`roll_id[t-L+1 : t+max_h]` 全部相等才通过
    * 通过则返回：

        * X: `features[t-L+1:t+1]`
        * y_h: `log(close[t+h]/close[t])`
        * asset_id（用于后续 embedding）
3. 写一个 `tools/smoke_train_archive.py`

    * 只跑 2k steps
    * batch 小一点（4060Ti 16G：建议 micro-batch 4~8 + grad accumulation）
    * 输出 eval_loss 曲线（每 N steps eval）
4. 指标记录：

    * 保存 `runs/.../eval_step*.json`（跟你现在风格一致）
    * 额外写 `runs/.../debug_roll_filter.json`：采样拒绝率（roll-safe reject rate）

**产物**

* `reports/m0_smoke_train.md`：关键超参、训练速度、loss 从多少降到多少、reject rate
* `runs/...` 下的 eval json + latest.pkl

---

## Milestone 1：扩到 30 品种的“数据管线”与稳定训练

目标：30 个品种都能 cache 出来；训练能稳定收敛；不同品种不会互相拖累。

### M1.1 批量 cache 生成（30 symbols）

**任务**

1. 写一个命令：`python -m tools/build_fut_cache --symbols @all_from_manifest --out datasets/fut1m_v1`
2. 支持断点续跑（cache 文件存在则跳过）
3. 输出全量统计

**产物**

* `reports/m1_cache_all.md`：30 个 symbol 的 T、日期范围、roll 次数、生成耗时

---

### M1.2 per-asset feature/target normalization

**任务**

1. 统计每个 asset 的 feature mean/std（8 维）
2. 统计每个 asset 每个 horizon 的 target mean/std
3. 训练时：

    * feature 用该 asset 的 stats 归一化
    * target 同上（你 val 用 train stats 的 bug 已修；这里把 stats 结构改成 per-asset）

**产物**

* `datasets/fut1m_v1/stats/feature_stats.json`
* `datasets/fut1m_v1/stats/target_stats.json`
* `reports/m1_stats_sanity.md`（检查极端 std、缺失）

---

### M1.3 采样策略（避免某些品种吞掉 batch）

**任务**

1. 实现 `p(asset) ∝ sqrt(T_asset)` 的采样权重
2. 打印每 1k steps 的 asset sampling 分布（debug）

**产物**

* `reports/m1_sampling.md`（采样分布是否符合预期）

---

## Milestone 2：模型改造（最小改动提升混训）

目标：跨品种训练更稳，不同品种的“微观结构差异”由 embedding 吸收。

### M2.1 加 asset embedding（极小改动）

**任务**

1. 在模型 config 增加：`num_assets`
2. 在 encoder 输入处加：

    * `asset_emb = Embedding(num_assets, d_model)`
    * 对每个 batch，取 `asset_emb[asset_id]` broadcast 到 seq_len，加到 token embedding
3. 其它不动

**产物**

* `reports/m2_ablation_asset_emb.md`：开/不开 embedding 的 eval_loss 对比（至少跑 20k steps）

---

## Milestone 3：评估与校准（让结果能“用”）

目标：不仅 loss 好看，还要 calibration/IC 等可解释指标。

### M3.1 评估指标补齐

**任务**

1. 逐 horizon 输出：

    * Pearson corr（预测 vs true）
    * R²（线性回归或直接）
    * 分位数回归的 pinball loss（你现在的 quantiles）
2. 输出 calibration 曲线（按预测分桶 vs 实际均值）

**产物**

* `reports/m3_metrics.md`
* `runs/.../metrics_step*.json`

---

## Claude Code 执行要求（你让它按这个返回给我）

请 Claude Code 在每个 milestone 完成后，给你一段“可复制回传”的总结，包含：

1. 执行的命令（完整 command）
2. 关键输出文件路径（reports、runs、datasets）
3. 关键数字（rows/T、reject rate、train/eval loss）
4. 如果失败：traceback + 复现步骤

---

## 你现在先让 Claude Code 做哪一段？

为了最快闭环：**只做 Milestone 0（M0.1~M0.4）**。

你把上面 Milestone 0 原样丢给 Claude Code 就行。
等它回传 `m0_smoke_train.md`（或至少 eval_step*.json 的曲线趋势 + reject rate），你贴回来，我就能基于真实结果继续给下一轮：是先调采样/归一化，还是先加 embedding，还是需要修 roll_id/map join 的细节。
