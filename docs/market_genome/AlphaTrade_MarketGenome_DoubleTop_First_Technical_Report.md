# AlphaTrade MarketGenome：Double-Top-First 技术方案报告

**版本**：v0.2  
**日期**：2026-07-13  
**状态**：可执行研究设计 / Double Top 端到端最小链路  
**适用仓库**：`fogdance/alphagenome_research`  
**起始分支**：`feature/market-genome-mg0`  
**建议开发分支**：`feature/market-genome-double-top-mvp`  
**目标读者**：AlphaTrade 项目负责人、K 线形态领域专家、量化研究员、数据工程师、JAX/模型工程师

---

## 0. 执行摘要

原 MarketGenome 方案试图一次完成：完整多形态 ontology、synthetic generator、dense market tracks、长序列 backbone、teachers、distillation、Market ISM 和最终 alpha benchmark。这个方向在科学上完整，但在实际开发中存在明显问题：

- 依赖链过长，任何前置定义错误都会向后传播；
- ontology 同时处理趋势、区间、M/W、头肩、突破、回踩和支撑阻力转换，领域决策面过大；
- 在尚未证明单个形态可被稳定学习前，就建设通用框架、全套 heads 和训练制度；
- 很难快速判断失败来自形态定义、标签、数据、模型、损失还是评估；
- Codex 容易为了“完成框架”而做过度抽象，而不是解决可证伪的核心问题。

本报告将路线改为 **Double-Top-First Vertical Slice**：

> 先只选择一个形态——双顶（M 头）——完成从领域定义、严格因果标签、跨周期 synthetic benchmark、真实市场 dense tracks、简单模型 learnability、MarketGenome mini-backbone、未来效果预测，到合法 K 线扰动解释的完整闭环。

这条最小链路仍然保留 AlphaGenome 的关键研究范式：

```text
原始长序列
→ 多分辨率因果表示
→ 密集 motif/phase tracks
→ motif 与 future effect 分离
→ held-out benchmark
→ 合法局部扰动与 effect scoring
```

但暂时延后：

```text
多形态统一 ontology
完整 pair/contact-map analogue
大规模 teacher ensemble
蒸馏
百万长度上下文
交易执行与仓位管理
```

Double Top MVP 的成功，不要求立即证明可交易；它需要依次证明：

1. 双顶形态可被严格因果、跨周期、一致地定义；
2. 同一模型能够在不同时间周期和不同形成时长上识别其形成过程；
3. 模型识别的不是简单“两个高点”，而是峰—颈线—第二峰—确认/失效的完整结构；
4. 模型能够区分 morphology 与未来效果；
5. 在真实样本外数据中，双顶条件下的 outcome head 至少提供超过无条件/rolling baseline 的增量信息；
6. 对第二峰、颈线深度、突破确认等结构进行合法扰动时，模型输出发生合理、可复现的变化。

如果这条链路失败，可以快速定位并停止；如果成功，再把相同接口复制到双底、头肩顶、趋势、突破/失败突破，最终扩展为原完整 MarketGenome 目标。

---

# 1. 为什么改成单形态端到端

## 1.1 M12 已经排除的工程问题

AlphaTrade M10-M12 已经证明：

- GPU/JAX 训练、checkpoint、bundle、offline inference 可运行；
- raw log return target 口径一致，未发现漏做 inverse transform；
- common-row control、schema、semantic、truncated/mutated causality 可通过；
- `feature_dim > 8` 能真实贯穿 dataloader、模型、bundle 和 inference；
- M12 21/21 formal runs 完成。

M12 的正式结论是：79 个 Chendage processed numeric features 直接加入现有全样本 pinball 模型后，没有超过 8D common-row control，也没有超过 rolling historical quantile。这个负结果说明：

> “将人工压缩的多周期状态作为 dense feature 直接输入当前短窗口收益模型”无效。

它没有证明跨周期 K 线形态不存在，也没有证明形态形成过程不可学习。

## 1.2 当前真正要验证的假设

新假设不再是：

```text
更多 processed features → 更低 pinball
```

而是：

```text
同一双顶拓扑会在不同时间周期与不同形成时长上重复出现；
长序列、多尺度因果模型可以学习其形成阶段；
形态表示能够对未来条件结果提供增量解释。
```

这个假设可被一个垂直切片明确证伪，因此适合成为下一主线。

## 1.3 为什么双顶适合作为第一形态

双顶比头肩顶更适合第一版：

- 拓扑最小：第一峰、颈线低点、第二峰；
- 形成阶段明确；
- 可以区分 shape complete 与 neckline-break confirmed；
- 可以生成丰富 synthetic positives 与 hard negatives；
- 可以在 1m、5m、15m、60m、日线等任意周期出现；
- 可以自然定义合法扰动：第二峰高度、两峰间距、颈线深度、突破强度；
- 可在后续直接镜像扩展为双底。

双顶也足够复杂，能验证 MarketGenome 是否真正处理：

```text
长程依赖 + 多 anchor 几何 + 阶段状态 + 跨尺度同构 + future effect
```

---

# 2. AlphaGenome 原则：MVP 保留什么，延后什么

AlphaGenome 的官方研究模型以最长 1 Mb DNA 为输入，在高分辨率和长上下文下预测大量功能 tracks，并通过多分辨率 encoder/decoder、coarse Transformer、任务 heads、held-out benchmarks、variant effect 和 ISM 形成完整研究闭环。

Double Top MVP 不复制 DNA 任务本身，而保留下列方法原则。

## 2.1 MVP 必须保留

| AlphaGenome 原则 | Double Top MVP 对应 |
|---|---|
| 原始长序列输入 | 原始因果 OHLCV/OI/session 序列，不输入人工交易评级 |
| 多分辨率表示 | 1m base sequence + causal encoder pyramid |
| 密集 tracks | 每个有效时刻、每个目标周期的双顶 phase/anchor/outcome tracks |
| motif 与 effect 分离 | 双顶 morphology/phase 与未来 return/barrier outcome 分开 |
| 高分辨率输出 | 在基础时间轴上对已关闭的目标周期输出状态 |
| held-out evaluation | purged time folds、held-out symbol/scale benchmark |
| variant/ISM | 合法修改第二峰、颈线、突破、量能后比较输出 |
| ablation | context、scale-sharing、heads、loss、synthetic pretraining 均做对照 |

## 2.2 MVP 暂时延后

| 延后项 | 原因 |
|---|---|
| 全部形态 family | 先证明一个 motif 的全链路可行 |
| pair/contact-map analogue | 尚无稳定市场二维监督目标 |
| 4-fold teachers 与 distillation | 单模型 learnability 尚未证明 |
| 超长 32K/百万上下文 | 先在 4K/8K 验证 |
| 数百 market tracks | 先保留最小必要 tracks |
| RL/执行/仓位 | 与形态表示 learnability 无关 |
| 直接追求 PnL | 先通过 pattern/effect benchmark |

## 2.3 “完整类比”的分阶段含义

Double Top MVP 是完整研究范式的**纵向切片**，不是最终规模：

```text
现在：一个 motif × 多尺度 × dense tracks × effect scoring
以后：多个 motif × 多模态 tracks × teachers/distillation × foundation trunk
```

只有纵向切片成功后，才横向扩展形态种类和任务数量。

---

# 3. 研究边界与非目标

## 3.1 MVP 研究范围

**唯一 morphology**：`DOUBLE_TOP`。  
**辅助结构**：因果 pivot、neckline level、break attempt、break confirmation、retest。  
**上下文状态**：prior trend 只作为独立 context track，不是双顶 morphology 的必要组成。  
**方向效果**：不在形态定义中写死；双顶形状不等于未来一定下跌。  
**目标周期**：第一版至少覆盖 `1m / 5m / 15m / 60m`；`240m / daily` 在上下文和数据量验证后加入。  
**基础分辨率**：1m。  
**正式 universe**：优先复用 AlphaTrade 现有 24-symbol universe；开发 smoke 可使用 M12 formal5。

## 3.2 明确非目标

MVP 不做：

- 头肩顶、双底、趋势等其他 pattern detector；
- 买卖动作、仓位、止损执行器；
- 将人工 grade/action/teacher outcome 作为输入；
- “看到双顶就做空”的策略结论；
- 多模型 teacher/distillation；
- 完整 MarketGenome pair stack；
- 以单一 backtest cell 作为成功标准。

---

# 4. 双顶的领域定义

## 4.1 morphology 与 effect 必须分离

MVP 把双顶定义为上方双峰 morphology：

```text
第一峰 H1
→ 中间颈线低点 N
→ 第二峰 H2
```

其结构方向为 `UPPER_MORPHOLOGY`，而不是直接标记为 `BEARISH`。

以下语义独立存在：

```text
morphology:
  是否形成双顶几何

context:
  前置趋势、波动、session、symbol

confirmation:
  是否有效跌破颈线

future_effect:
  后续收益、MFE/MAE、target-before-stop、失败/延续
```

经典“顶部反转”语义应由组合条件产生：

```text
DOUBLE_TOP morphology
+ prior_uptrend context
+ neckline_break_confirmed event
```

而不是由 `DOUBLE_TOP` 名称自动推导。

## 4.2 严格因果 pivot

禁止使用 centered-window swing 或未来 ZigZag 回填。每个 pivot 维护：

```text
pivot_id
pivot_type: HIGH | LOW
anchor_state: PROVISIONAL | CONFIRMED | REVISED | SUPERSEDED | REVOKED
event_eob
available_as_of_eob
price_basis
price_value
prominence_norm
revision
```

推荐高点 pivot 的在线生命周期：

1. 当前高点成为 provisional candidate；
2. 如果出现更高高点，candidate 被 revised/superseded；
3. 当价格从 candidate high 发生达到配置阈值的因果回撤，pivot 变为 confirmed；
4. `event_eob` 保留真实峰值时间；
5. `available_as_of_eob` 是回撤确认完成的已收盘 bar 时间。

必须满足：

```text
event_eob <= available_as_of_eob <= observation.as_of_eob
```

## 4.3 双顶 anchors

最小 anchors：

| Anchor | 含义 |
|---|---|
| `H1` | 第一 confirmed high pivot |
| `N` | H1 后第一有效 confirmed low pivot，作为 neckline candidate |
| `H2` | N 后的第二 high pivot/candidate |
| `B` | 首次 neckline break attempt/confirmation bar |
| `R` | 可选 neckline retest bar/zone |

## 4.4 几何公式

对 H1、N、H2，定义：

```text
mean_peak = (H1 + H2) / 2
formation_height = mean_peak - N
peak_similarity = abs(H1 - H2) / max(formation_height, eps)
neckline_depth_atr = formation_height / ATR_at_shape
time_left = N.event_eob - H1.event_eob
time_right = H2.event_eob - N.event_eob
time_symmetry = abs(time_left - time_right) / max(time_left + time_right, 1)
second_peak_relative_height = (H2 - H1) / max(formation_height, eps)
```

所有公式必须记录：

- 分子；
- 分母；
- 符号；
- 缺失策略；
- clip 策略；
- 所用价格 basis；
- 所用 ATR/volatility 估计及其 train/runtime 因果口径。

数值阈值属于 detector config，不属于 ontology 拓扑。

## 4.5 Online phase track

MVP 使用以下 family-specific phases：

| Phase | 首次可见条件 | 说明 |
|---|---|---|
| `NONE` | 无有效结构 | 无双顶候选 |
| `FIRST_PEAK_AVAILABLE` | H1 confirmed | 第一峰已可在线获知 |
| `NECKLINE_AVAILABLE` | N confirmed | 颈线低点已可在线获知 |
| `SECOND_PEAK_FORMING` | N 后出现上攻，尚未确认 H2 | 第二峰可能持续修订 |
| `SHAPE_COMPLETE` | H2 confirmed 且几何约束满足 | 双顶 morphology 已完成，但未必跌破颈线 |
| `BREAK_ATTEMPT` | 价格首次触碰/越过颈线阈值 | 尚未达到 accepted-break 定义 |
| `BREAK_CONFIRMED` | 已收盘 bar 满足 accepted-break contract | 经典确认事件 |
| `RETESTING` | 确认后返回颈线区域 | 可选路径 |
| `CONTINUATION` | 突破后离开颈线并持续 | 结构后续阶段，不等于盈利 |
| `INVALIDATED` | 形态在确认前/后被失效条件破坏 | 明确终止 |
| `EXPIRED` | 超过最大形成/等待时长 | 无明确失效但不再有效 |

`SHAPE_COMPLETE` 和 `BREAK_CONFIRMED` 必须分开。否则模型无法学习“形状存在但未确认”与“已发生结构性突破”的差别。

## 4.6 Invalidation 与 expiry

必须至少支持：

- H2 在 shape complete 前大幅超过 H1，破坏峰值兼容性；
- N 后超过最大形成时长仍未确认 H2；
- shape complete 后长期未触发颈线确认；
- 突破后重新完全站回颈线上方并满足 full-recross failure；
- 数据/session/roll discontinuity 导致结构不可继续。

失效规则和阈值必须版本化、只在 train/validation 上选择，test 不得参与。

---

# 5. 跨周期与尺度合同

## 5.1 三种尺度不能混淆

```text
bar_resolution：观察 K 线周期，例如 1m/5m/15m/60m
formation_span：形态在该周期上持续的 bar 数量
model_scale：encoder 内部 /1,/4,/16,/64 等表示层级
```

同一个 1m 图上可以同时存在：

- 20 根 1m bar 形成的小双顶；
- 300 根 1m bar 形成的大双顶。

所以 `scale_id` 不能只等于 timeframe。

建议 `ScaleSpec`：

```yaml
bar_resolution: 1m
aggregation_policy: SESSION_ALIGNED_CLOSED_BAR
formation_span_bars_min: ...
formation_span_bars_max: ...
pivot_prominence_scale_ref: ATR | REALIZED_VOL | FORMATION_HEIGHT
scale_octave: optional
label_close_alignment: CLOSED_BAR_ONLY
```

## 5.2 第一版目标周期

正式 MVP 至少覆盖：

```text
1m
5m
15m
60m
```

原因：

- 足以验证同一 morphology 跨多个周期；
- 1m base context 能覆盖这些周期；
- daily 形态需要更长上下文和更少独立事件，适合第二阶段。

## 5.3 Scale sharing 的验证

模型必须支持一个共享 double-top head，加 `scale embedding`，而不是为每个周期独立训练完全不同 detector。

必须做两类 benchmark：

1. **joint-scale**：所有周期共同训练；
2. **held-out-scale**：synthetic 中留出一组形成时长或一个周期，只在 test 评估。

如果模型只能记住特定 bar 数，而不能迁移到新的 time dilation，则没有学到跨尺度 motif。

---

# 6. 输入、tracks 与标签

## 6.1 原始输入

基础 1m 输入建议：

```text
ret_1m
open_to_prev_close
close_to_open
high_to_open
low_to_open
upper_wick
lower_wick
high_low_range
log_volume / volume_zscore
open_interest_change / OI_zscore
session position sin/cos
session boundary flags
roll/contract-age metadata
symbol embedding
exchange/sector embedding
```

原则：

- 价格平移不变；
- 尽量价格尺度不变；
- normalization 只 fit train；
- 不能输入未来确认的 pattern label；
- 可以输入已收盘高周期聚合值，但必须 closed-bar aligned；
- 第一版优先只输入基础原始/派生通道，不输入 Chendage grade/action。

## 6.2 MVP 必需 tracks

### A. Pivot tracks

```text
pivot_high_confirmed[scale]
pivot_low_confirmed[scale]
pivot_age[scale]
pivot_prominence[scale]
```

### B. Double-top morphology tracks

```text
double_top_phase[scale]
double_top_active[scale]
shape_completion_confidence[scale]
```

### C. Anchor/geometry tracks

```text
distance_to_h1_norm[scale]
distance_to_neckline_norm[scale]
peak_similarity[scale]
neckline_depth_atr[scale]
time_symmetry[scale]
formation_age_norm[scale]
```

### D. Confirmation/event tracks

```text
break_attempt[scale]
break_confirmed[scale]
retest_state[scale]
```

### E. Future effect tracks

在 `SHAPE_COMPLETE` 和 `BREAK_CONFIRMED` event 上计算：

```text
future_return_residual_quantiles[h]
target_before_stop_probability[h or barrier_config]
MFE[h]
MAE[h]
time_to_target
time_to_invalidation
```

其中 return head 建议预测相对 rolling historical quantile 的 residual：

```text
Q_model = Q_rolling_causal + scale_h * DeltaQ_model
```

模型初始状态应接近 rolling baseline，而不是从随机 `O(1)` quantile width 开始。

### F. 最小辅助 tracks

```text
future_realized_vol[h]
future_high_low_range[h]
```

这两个 dense target 给 trunk 提供比稀疏形态事件更稳定的监督。

## 6.3 标签 namespace

必须分为：

```text
online_causal/*
retrospective_structure/*
future_outcome/*
```

- `online_causal` 可作为 runtime target/diagnostic；
- `retrospective_structure` 只用于离线审计或 teacher comparison；
- `future_outcome` 只能作为训练 target，绝不能进入 input。

---

# 7. Synthetic Multi-Scale Double Top Benchmark

## 7.1 为什么 synthetic 是硬门禁

真实市场的双顶标签存在主观性和噪声。如果模型在已知 ground truth 的 synthetic 数据上都无法：

- 识别 H1/N/H2；
- 区分形成中与完成；
- 跨尺度迁移；
- 拒绝 hard negatives；

就不应该进入昂贵真实数据训练。

## 7.2 生成器组成

每条 synthetic sequence 包含：

```text
背景过程：trend/range/stochastic volatility
局部结构：H1 → N → H2
后续路径：break / no-break / false-break / invalidation / continuation
噪声：微结构噪声、波动变化、gap、volume/OI pattern
```

随机化参数：

- 全局价格平移与缩放；
- 波动率；
- 形成时长；
- 两峰相对高度；
- 颈线深度；
- 左右时间比例；
- 前置趋势；
- 第二峰是否突破第一峰；
- break strength；
- retest 路径；
- volume/OI confirmation；
- 不同 session boundary。

## 7.3 Hard negatives

至少包括：

- 两峰高度差过大；
- 中间回撤过浅；
- 随机震荡碰巧出现两个局部高点；
- 三重顶；
- 区间上沿多次测试；
- 第二峰后向上突破；
- 颈线影线刺穿但收盘未确认；
- 第一峰/第二峰时间跨度超出有效范围；
- 形成过程中被新高 supersede；
- 类似头肩结构但不是双顶。

## 7.4 Synthetic Gate

至少要求：

- phase macro-F1 / AUPRC 达到预先冻结阈值；
- anchor localization error 在可接受 bar 范围；
- hard-negative false positive rate 低于阈值；
- held-out time-dilation/scale 仍有明显泛化；
- shuffled-label control 显著更差；
- 修改未来数据不改变当前 track；
- 合法 affine price transform 不改变 morphology label。

---

# 8. 真实数据因果 labeler 与人工 Gold Set

## 8.1 规则 labeler 的角色

第一版 causal labeler 不是最终“真理”，而是：

- 将领域定义变成可复现 weak labels；
- 生成大规模 dense tracks；
- 暴露争议样本供人工 review；
- 为模型提供初始 phase supervision。

不能用同一规则 labeler 的输出证明该规则具有 alpha。

## 8.2 参数管理

应分为：

```text
ontology：不变的拓扑与阶段定义
labeler config：数值阈值与搜索空间
```

labeler 阈值只允许：

- 由领域专家冻结；或
- 在 train/validation 上从预先声明的小网格选择。

禁止在 test 上调：

- pivot prominence；
- peak similarity；
- minimum neckline depth；
- formation span；
- break buffer/hold；
- expiry；
- invalidation。

## 8.3 Human Gold Set

人工 review 需要三种模式：

1. `ONLINE_BLINDED`：审核者只能看到 `<=as_of_eob`；
2. `RETROSPECTIVE_STRUCTURE`：允许看后续，用于判断形态最终结构；
3. `OUTCOME`：单独评估后续效果。

每个样本至少：

```text
两名独立审核者
允许 AMBIGUOUS / INSUFFICIENT_EVIDENCE
分歧时第三人 adjudication
记录 reviewer_id、版本和 future visibility
```

Gold set 必须包含：

- positives 各阶段；
- invalidated/expired；
- hard negatives；
- 不同周期；
- 不同形成时长；
- 不同品种和波动 regime。

---

# 9. MarketGenome Mini Backbone

## 9.1 模型目标

第一版不是构建最终 foundation model，而是验证：

> 一个严格因果、多尺度、长上下文、scale-shared 的模型能否学习双顶形成轨迹，并对未来效果提供增量预测。

## 9.2 推荐形态

```text
1m input sequence
→ causal stem
→ multi-stage causal encoder (/1,/4,/16,/64)
→ small coarse Transformer
→ causal decoder + skip connections
→ base-resolution and coarse embeddings
→ scale-conditioned shared double-top heads
→ rolling-residual outcome heads
```

## 9.3 第一版资源边界

建议：

```text
context_length: 4096，之后 ablate 8192
encoder_stages: 4
coarse tokens: 64 或 128 量级
transformer_layers: 4-6
hidden_dim: 128-256
params: 5M-20M
hardware: 单卡 16GB 可运行
```

不实现：

- pair stack；
- 复杂 decoder 多任务全集；
- teacher ensemble；
- million-token context。

## 9.4 Scale-shared motif head

不要为 1m/5m/15m/60m 分别创建完全独立 detector。建议：

```text
shared motif projection
+ learned scale embedding
+ resolution-specific alignment/mask
```

输出可以在 base 1m 时间轴上对齐；非目标周期 closed-bar endpoint 位置使用 mask。

## 9.5 Heads

第一版最小 heads：

```text
pivot head
phase classification head
anchor/geometry regression head
break/retest event head
future vol/range head
rolling-residual quantile head
target-before-stop head
```

## 9.6 Loss

建议：

```text
L = w_phase * focal_or_weighted_CE
  + w_anchor * masked_anchor_loss
  + w_geometry * masked_Huber
  + w_event * weighted_BCE
  + w_vol_range * regression_loss
  + w_quantile * residual_pinball
  + w_barrier * Brier_or_BCE
  + w_consistency * cross_scale_consistency
```

原则：

- 稀疏 phase/event 使用 class-balanced loss；
- regression 只在对应 phase mask 下计算；
- outcome 只在事件 eligible rows 计算；
- loss weight 只在 train/val 调整；
- 所有权重进入 resolved config/hash/report。

---

# 10. Learnability Ladder

正式 MarketGenome 训练前必须按复杂度建立模型阶梯：

```text
rule/labeler agreement diagnostics
→ linear/logistic baseline
→ 2-layer MLP
→ small causal TCN
→ small GRU
→ MarketGenome mini-backbone
```

分别回答：

- raw features 是否含有足够信息；
- 时间结构是否必要；
- 长上下文是否优于短窗口；
- multi-scale scale sharing 是否带来收益；
-复杂 backbone 是否真的优于简单模型。

硬门禁：

- one-batch overfit；
- synthetic signal recovery；
- shuffled-label control；
- future-mutation invariance；
- simple-model ladder。

如果所有简单模型和 MarketGenome 都不能在 synthetic 或 gold-set 上学习，禁止进入大规模真实训练。

---

# 11. 正式评估体系

## 11.1 四层结论分开

### A. Engineering pass

```text
schema/semantic
causality
alignment
masking
reproducibility
GPU stability
```

### B. Pattern learnability pass

```text
phase AUPRC/F1
anchor localization
hard-negative rejection
scale transfer
human gold-set agreement
```

### C. Pattern effect pass

```text
confirmed/shape-complete event 的条件结果是否不同
outcome head 是否优于无条件/rolling/event-frequency baseline
```

### D. Alpha candidate pass

```text
样本外 rolling-residual pinball
IC/rank IC
barrier calibration
cost diagnostic
跨 time fold/symbol 稳定性
```

不能把 A/B pass 写成“可交易”。

## 11.2 Pattern benchmark

必须报告：

- per-scale phase macro-F1/AUPRC；
- per-phase confusion matrix；
- event-level precision/recall；
- anchor bar localization error；
- formation span bucket；
- symbol/time-fold/regime；
- hard negative subtype；
- held-out scale/time dilation；
- human agreement。

## 11.3 Effect benchmark

在 `SHAPE_COMPLETE` 与 `BREAK_CONFIRMED` 分开评估：

- future return distribution；
- target-before-stop AUC/Brier；
- MFE/MAE；
- time-to-event；
- rolling baseline residual；
- matched controls：同 symbol/session/vol/prior trend/time bucket。

## 11.4 Alpha benchmark

只有 pattern/effect pass 后才运行：

- pinball/CRPS；
- coverage；
- Pearson IC / rank IC；
- direction hit；
- simple cost matrix；
- turnover/drawdown diagnostic。

M9/M10/M11 backtest 仍是轻量 handoff，不是完整 execution proof。

---

# 12. Market In-Silico Mutagenesis（MVP）

## 12.1 合法编辑

对 synthetic 和真实事件做：

- 提高/降低第二峰；
- 拉近/拉远两峰；
- 加深/变浅颈线；
- 改变第二峰形成速度；
- 把收盘确认改为仅影线刺穿；
- 删除或增强第二峰量能/OI；
- 将有效 break 改为 false break；
- 将结构延长/压缩。

所有编辑必须满足：

```text
low <= min(open, close) <= max(open, close) <= high
volume >= 0
OI 合法
session/roll 不被破坏
只修改声明的时间范围
```

## 12.2 比较输出

```text
phase probability delta
shape completion delta
break probability delta
future residual quantile delta
barrier probability delta
embedding attribution delta
```

Synthetic 中可以有预期方向；真实市场中不应强行规定每个编辑的未来收益单调性，只验证模型是否对结构关键点敏感、而非只关注最后几根 bar。

---

# 13. 分阶段里程碑

| 里程碑 | 目标 | 核心 Gate |
|---|---|---|
| `DT0` | 冻结双顶 ontology 与 detector contract | 专家批准、无多形态扩张 |
| `DT1` | 因果多周期 bar/pivot primitives | prefix/future mutation pass |
| `DT2` | synthetic multi-scale generator | held-out scale learnability ready |
| `DT3` | 真实 causal labeler + gold-set workflow | label quality/人工审核 pass |
| `DT4` | dense track long-sequence dataset | data/schema/causality pass |
| `DT5` | simple baseline + learnability ladder | synthetic/gold-set gate pass |
| `DT6` | MarketGenome mini-backbone | architecture/overfit/invariance pass |
| `DT7` | 正式 end-to-end training/evaluation | pattern/effect/alpha 分层决策 |
| `DT8` | Market ISM | structure sensitivity pass |
| `DT9` | 关闭 MVP、决定扩展 | expand / redesign / pause |

---

# 14. MVP 成功与失败条件

## 14.1 `DOUBLE_TOP_REPRESENTATION_PASS`

必须满足：

- synthetic phase/anchor 指标过 gate；
- human gold-set 指标过 gate；
- held-out scale/time dilation 明显泛化；
- hard-negative false positives 可控；
- MarketGenome 优于简单 short-window baseline，或证明长上下文必要。

## 14.2 `DOUBLE_TOP_EFFECT_PROMISING`

在 representation pass 基础上：

- shape-complete/break-confirmed events 的 outcome 与 matched control 有稳定差异；
- outcome head 优于 event-frequency、zero 和 rolling baseline；
- 结果在至少多个 time fold 和多个 symbol 上方向一致；
- 不由极少事件贡献。

## 14.3 `DOUBLE_TOP_ALPHA_CANDIDATE`

在 effect promising 基础上：

- rolling-residual quantile 在 test 改善；
- calibration、IC/rank IC 不退化；
- 轻成本 diagnostic 不立即完全崩溃；
- 无单一周期/单一品种依赖。

## 14.4 `REDESIGN_OR_STOP`

以下任一发生时停止扩展其他形态：

- causal ontology 无法获得领域一致性；
- synthetic held-out-scale 学不会；
- human gold-set agreement 太低；
- simple models 与 MarketGenome 均无 learnability；
-形态可识别但 outcome 对 matched control 无任何增量；
- 结果高度集中或不稳定。

---

# 15. 成功后如何扩展到完整 MarketGenome

扩展顺序建议：

```text
DOUBLE_TOP
→ DOUBLE_BOTTOM（几何镜像，独立验证多空不对称）
→ HEAD_AND_SHOULDERS_TOP / INVERSE
→ BREAKOUT / FAILED_BREAKOUT / RETEST
→ TREND / RANGE / TRANSITION persistent states
```

每增加一个 pattern，必须复用相同接口：

```text
PatternOntology
CausalLabeler
SyntheticGenerator
DenseTrackSpec
PatternHead
EffectEvaluator
PerturbationSpec
```

但不要在 Double Top MVP 前预先实现通用 registry 的全部功能。DT9 通过后，再从真实代码中抽象最小 pattern plugin API。

之后才恢复原完整路线：

- 多 pattern 联合 tracks；
- 更长上下文；
- pairwise relation stack；
- fold teachers；
- distillation；
- 更广泛 market modalities；
- 全 universe foundation pretraining；
- pattern effect 与 alpha benchmark。

---

# 16. 主要风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| labeler 把主观规则当真值 | 模型只模仿规则 | synthetic ground truth + human gold set + weak-label provenance |
| 未来 pivot 泄漏 | 虚假高指标 | event/available timestamps、future mutation、closed-bar tests |
| 双顶稀疏 | 全样本 loss 淹没 | dense phase tracks、event sampling、class-balanced loss |
| 只记住固定时间跨度 | 无跨周期泛化 | time dilation、held-out scale、scale-shared head |
| morphology 与 bearish outcome 混淆 | 错误研究结论 | orientation/context/effect namespace 分离 |
| 规则阈值过拟合 | test 污染 | 预声明小网格、train/val only、hash/freeze |
| 复杂模型掩盖不可学性 | 浪费算力 | simple-model ladder、synthetic gate |
| 过早通用化 | 开发变慢 | 禁止实现其他 pattern，DT9 后再抽象 |
| PnL 诱导 cherry-pick | 错误 promotion | pattern/effect/alpha 三层 gate |

---

# 17. 推荐执行决策

当前 `feature/market-genome-mg0` 的 MG0 和 MG1-A broad ontology 工作不应删除；将其保留为研究资产和未来扩展参考。但新的主线应：

1. 从该分支创建 `feature/market-genome-double-top-mvp`；
2. broad `pattern_ontology_v1.yaml` 保持 draft，不作为当前开发阻塞项；
3. 新建仅包含 double top vertical slice 的 ontology/config/contracts；
4. 先执行 DT0-DT5；
5. 只有 learnability gate 通过，才实现 DT6/DT7；
6. 只有 Double Top representation/effect 至少达到明确级别，才增加第二个形态。

最终目标没有改变：构建一个类似 AlphaGenome 的严格因果、长序列、多尺度、dense-track MarketGenome。改变的是开发策略：

> 不再横向一次建设全部能力，而是用双顶完成纵向端到端闭环，再沿已验证接口逐个增加形态和任务。

---

# 18. 参考资料与项目证据

## AlphaGenome 官方资料

1. Avsec et al., **Advancing regulatory variant effect prediction with AlphaGenome**, Nature, DOI: `10.1038/s41586-025-10014-0`。
2. Google DeepMind, **AlphaGenome Research** official JAX repository：`google-deepmind/alphagenome_research`。
3. Google DeepMind, **AlphaGenome: AI for better understanding the genome**。

## AlphaTrade 项目证据

1. `M12_CHG_FORMAL_DECISION_MEMO.md`。
2. `m12_regression_report.json/md`。
3. `m12_formal_data_contract_spotcheck.json/md`。
4. `feature/market-genome-mg0` branch 中的 MG0/MG1-A ontology scaffold。

---

**最终结论**：Double-Top-First 是当前更可执行、可证伪、风险更低的 MarketGenome 路线。它保留 AlphaGenome 方法论的核心，但把工程范围缩到足以完成一次真实端到端研究闭环的大小。
