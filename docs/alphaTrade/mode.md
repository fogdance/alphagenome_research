下面是 **Markdown 格式**的《**AlphaTrade v0.2 Model Spec**》。它在 v0.1 契约不变的前提下，把 **AlphaGenome 源码里值得借鉴的工程细节**（weight standardization、repeat 上采样、RMS 系归一化、RoPE、logits soft-cap、粗尺度 tower 等）明确写进“参考实现建议”，并标注哪些是“必须因果化改造”。

---

# AlphaTrade v0.2 Model Specification (MD)

**Name**: AlphaTrade
**Version**: v0.2
**Status**: Draft (engineering-ready)
**Last updated**: 2026-02-13 (Asia/Singapore)

## 0. 设计目标与兼容性

### 0.1 目标

AlphaTrade v0.2 是一个 **多 horizon 预测器**：给定由 1 分钟 OHLCV 因果计算得到的特征序列窗口，输出多个 horizon 的 **未来对数收益分布（分位数）**，并可选输出波动/区间、成交量和市场状态（regime）。
一句话硬目标：每天（或每小时）自动跑“取数→推理→生成信号→记录拟交易→事后对账/评估”，并且可用历史回放模拟同样流程
### 0.2 与 v0.1 的兼容性（强约束）

* **输入契约不变**：`X ∈ R^{L×F}`，F=8（固定顺序），L 可配置。
* **核心输出不变**：每个 horizon 的 `log_return_quantiles[h][q]`（Q 默认 5 分位）。
* v0.2 只新增 **可选 heads** 与 **更贴近 AlphaGenome 的 trunk 工程细节**，不破坏 v0.1 调用方。

---

## 1. 输入（Input Contract）

### 1.1 输入张量

`X ∈ R^{L × 8}`（oldest → newest），最后一行对应 `asof_bar_end`。

### 1.2 特征列表（F=8，固定顺序）

> 所有特征 **必须严格因果**：只使用 `≤ asof_bar_end` 的数据。

| idx | name              | definition                            |
| --: | ----------------- | ------------------------------------- |
|   0 | `lr_close`        | `log(C_i / C_{i-1})`                  |
|   1 | `hl_range`        | `log(H_i - L_i + epsilon)`            |
|   2 | `oc_return`       | `log(C_i / O_i)`                      |
|   3 | `log_vol`         | `log(V_i + 1)`                        |
|   4 | `pos_in_range`    | `(C_i - L_i) / (H_i - L_i + epsilon)` |
|   5 | `time_sin`        | `sin(2π * minute_index / period)`     |
|   6 | `time_cos`        | `cos(2π * minute_index / period)`     |
|   7 | `is_session_open` | 0/1，处理休市/午休/夜盘断档（推荐保留）                |

**epsilon**：默认 `1e-9`（训练与线上必须一致）。

### 1.3 归一化（建议）

* per-instrument robust scaling（median/IQR），仅在训练集拟合，线上加载参数。
* 严禁在归一化中跨时间维做“未来可见”的统计（例如整段窗口均值并不泄露未来，但要保证线上与训练一致；更稳的做法是每 feature 用固定 scaler 参数）。

---

## 2. 输出（Output Contract）

### 2.1 必选输出：多 horizon 的 log-return 分位数

对每个 horizon `h ∈ H`：

* 目标：`r_t^(h) = log(C_{t+h} / C_t)`
* 输出：`q_return[h] ∈ R^{|Q|}`

默认：

* `H = [1, 5, 20, 60]`
* `Q = [0.1, 0.25, 0.5, 0.75, 0.9]`

### 2.2 可选输出（v0.2 新增，可开关）

> **不改变 v0.1 字段**，新增字段放在 `pred.optional.*` 下。

1. **波动/区间分位数**（建议启用）

* `range_t^(h) = log(max_{k∈(t,t+h]} H_k - min_{k∈(t,t+h]} L_k + epsilon)` 或 realized vol
* 输出：`q_range[h] ∈ R^{|Q|}`

2. **成交量分位数**（建议启用）

* `vol_t^(h) = log( Σ_{k=t+1..t+h} V_k + 1 )`
* 输出：`q_vol[h] ∈ R^{|Q|}`

3. **Regime 分类**（可选，偏中期）

* 类别示例：`{up, down, range}` 或 `{trend_up, trend_down, mean_revert, high_vol, low_vol}`
* 输出：`p_regime ∈ R^{K}`（softmax）

---

## 3. 模型架构（Architecture）

### 3.1 因果性（硬约束）

与 AlphaGenome 的“整段序列建模”不同，AlphaTrade 必须严格因果：

* 所有 Conv 必须 **left padding**（因果卷积）
* Pool/Downsample 必须 **不使用未来位置**
* Transformer Attention 必须 **causal mask**（严格左侧注意力）

> AlphaGenome 源码中的 encoder 采用 `pool → DownResBlock` 多次下采样，并在 decoder 用 `UpResBlock` 上采样回灌 skip。([GitHub][1])
> 但它的 conv/pool 默认 `padding='SAME'`，直接照搬会在金融时序里“看见未来”。([GitHub][2])

---

### 3.2 Backbone：Temporal U-Net + Coarse Transformer（AlphaGenome-inspired）

#### 3.2.1 总览（与 AlphaGenome 的 trunk 对齐）

Pipeline：

1. **Stem**：`[L, 8] -> [L, C0]`
2. **Encoder**：多级 Downsample + DownRes（保存 skip）
3. **Transformer Tower**：在粗分辨率 `T=L/DS` 上建模长程依赖
4. **Decoder**：多级 Upsample + UpRes（融合 skip）
5. **Readout**：取 `asof` 时刻表征（默认 last step）
6. **Heads**：multi-horizon + (optional heads)

> AlphaGenome 代码里明确是 `SequenceEncoder → TransformerTower → SequenceDecoder → Heads` 的三段式 trunk。([GitHub][1])

---

#### 3.2.2 关键工程细节（从 AlphaGenome 借鉴，需因果化）

##### A) Weight Standardization（强烈建议采用）

AlphaGenome 的 `StandardizedConv1D` 对卷积核做 **weight standardization + scale**，再做卷积（能显著提升训练稳定性、减少对 batch size 的敏感）。([GitHub][2])

**AlphaTrade v0.2 建议**：

* 在所有 1D Conv（尤其是 DownRes/UpRes 的 width=5 卷积）启用 WS：

  * PyTorch：可用 weight standardization（自实现或第三方实现）
  * 注意：因果卷积只改变 padding，不影响 WS 逻辑

##### B) Up-sampling 用 repeat（简单、稳、快）

AlphaGenome 的 `UpResBlock` 上采样使用 `repeat(x, 2)`，再加 residual_scale 和卷积融合 skip。([GitHub][2])

**AlphaTrade v0.2 建议**：

* 上采样默认：`repeat -> causal conv`
* 保留一个可学习的 `residual_scale`（避免上采样后数值放大）

##### C) RMS 系归一化（建议采用）

AlphaGenome 广泛使用 `RMSBatchNorm` / RMS 风格归一化（对 last-dim 做 RMS 归一化、带 scale/offset）。([GitHub][2])

**AlphaTrade v0.2 建议**：

* 使用 **RMSNorm**（per-token、仅按通道归一化），避免时间维/批维的统计依赖
* 若团队更熟 LayerNorm，也可用 LN，但 RMSNorm 往往更稳

##### D) Transformer：RoPE + logits soft-cap（强烈建议）

AlphaGenome 的 attention block：

* 对 q/k 应用 **RoPE**（旋转位置编码）([GitHub][3])
* 对 attention logits 做 **tanh soft-cap（logits_soft_cap=5.0）**，抑制极端 logits，提升稳定性([GitHub][3])

**AlphaTrade v0.2 建议**：

* 在 coarse Transformer 启用 RoPE（适合长上下文）
* 启用 logits soft-cap（极端行情/噪声下更稳）
* 但必须加 **causal mask**（AlphaGenome 不需要因果 mask，我们需要）

---

### 3.3 推荐的参考实现配置（v0.2 baseline）

> v0.2 相对 v0.1 的最大变化：**更深的 downsample（趋近 /64 或 /128）**，让 Transformer 在更粗的 token 上工作（更像 AlphaGenome 的做法）。([GitHub][1])

#### 3.3.1 Stem

* `C0 = 128`（可配：128/192/256）
* `causal conv kernel=7` + RMSNorm + GELU

#### 3.3.2 Encoder（Down path）

* stage 数：**6 stages**（推荐，目标下采样到 `/128`）
* 每级结构（概念）：

  * `CausalPool(by=2)` 或 `stride-2 causal conv`
  * `DownResBlock` × 1~2

**通道增长策略（可选）**

* **AlphaGenome 风格**：每个 DownRes 增加固定通道增量（例如 +64 或 +128），并用通道 pad 做残差对齐。([GitHub][2])
* AlphaTrade 建议（更轻量）：

  * `128 -> 192 -> 256 -> 320 -> 384 -> 448 -> 512`
  * Transformer 前加一层 `proj -> d_model`

#### 3.3.3 Transformer Tower（Coarse resolution）

* 输入长度：`T = L / 128`（若 L=4096，则 T=32）
* `d_model = 512`（可配：384/512）
* `n_heads = 8`
* `N_layers = 6`（可配：4~10）
* 每层：

  * RMSNorm
  * Causal MHA（RoPE + logits soft-cap）
  * MLP（2~4× expansion）
  * residual

> AlphaGenome tower 里还有 “pair update + attention bias” 的 interleaved 结构。([GitHub][1])
> **AlphaTrade v0.2 不要求实现 pair 分支**；可保留一个“可插拔 attention bias”接口，后续扩展。

#### 3.3.4 Decoder（Up path）

* 对应 6 级上采样，逐级融合 skip：

  * `UpResBlock`: `repeat x2`（默认）+ skip 融合 + conv ([GitHub][2])
* 输出回到 `T=L` 的时序表征

#### 3.3.5 Readout（asof 表征）

默认：取 decoder 输出最后一个 time step 的 embedding（对应 `asof_bar_end`）。

可选（v0.2 建议保留开关）：

* `asof pooling`：对最后 `W` 个 steps 做 attention pooling（更抗噪）

---

## 4. Heads（多任务输出设计）

### 4.1 Return Quantile Heads（必选）

* 每个 horizon 一个 head（共享 trunk）
* head 输入：readout embedding `E`
* 输出：`|Q|` 个分位数

**防止 quantile crossing（v0.2 建议）**

* 方案 A：输出 `q50` + 正增量 `Δ`（softplus）累加得到有序分位数
* 方案 B：加 crossing penalty（见训练）

### 4.2 Optional Heads（v0.2）

* `RangeQuantileHead(h)`：输出 `q_range[h]`
* `VolumeQuantileHead(h)`：输出 `q_vol[h]`
* `RegimeHead`：输出 `p_regime`

---

## 5. 训练（Training）

### 5.1 Labels

* `y_return^(h) = log(C_{t+h}/C_t)`（同 v0.1）
* 可选：

  * `y_range^(h)`（range/vol）
  * `y_vol^(h)`（log volume）
  * `y_regime`（基于规则生成 label，或弱监督）

### 5.2 Loss（v0.2 推荐组合）

#### 5.2.1 Quantile Pinball Loss（主损失）

对每个 horizon：

```
Pinball(q, y, yhat) = max(q*(y - yhat), (q-1)*(y - yhat))
Total = Σ_h w_h * mean_q Pinball
```

#### 5.2.2 Quantile Crossing Penalty（建议）

保证 `q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90`：

* penalty = `Σ max(0, q_i - q_{i+1})`

#### 5.2.3 多任务权重（建议默认）

* returns：`w_return = {1:1.0, 5:1.0, 20:0.8, 60:0.6}`
* range/vol/volume：整体系数 `λ_range, λ_vol`（默认 0.3~0.7）
* regime：`λ_regime`（默认 0.2~0.5）

### 5.3 验证协议

* walk-forward split（严格按时间）
* 不允许随机打乱跨时间样本
* 报告每个 horizon 的：pinball、coverage（区间覆盖率）、校准误差（ECE/可靠性）

---

## 6. 推理与部署（Inference & Serving）

### 6.1 推理模式

* 每分钟收盘后触发一次预测（asof）
* 输入窗口从在线 ring buffer 取最后 L 步特征

### 6.2 Streaming/缓存优化（v0.2 建议预留）

* 由于 encoder 是逐层下采样，可缓存部分中间态：

  * 例如每来 1 根新 bar，只增量更新一部分尺度的表示（实现复杂，可作为 v0.3）

### 6.3 监控（Monitoring）

* 输入漂移：feature 分布 PSI/KL
* 输出漂移：return 分位数分布、区间覆盖率稳定性
* 质量衰退：walk-forward online eval（预测 vs 真实回放）
* 延迟：P50/P95

---

## 7. 参考实现清单（AlphaGenome-inspired checklist）

> 这一节用于“编码时对照”，确保工程细节落地。

* [ ] Conv 实现：支持 **weight standardization**（AlphaGenome `StandardizedConv1D`）([GitHub][2])
* [ ] Up-sampling：默认 `repeat x2` + 可学习 residual_scale（AlphaGenome `UpResBlock`）([GitHub][2])
* [ ] Norm：优先 RMSNorm/RMSBatchNorm 风格（AlphaGenome 广泛使用）([GitHub][2])
* [ ] Transformer：RoPE（q/k）+ logits soft-cap（tanh cap=5.0）([GitHub][3])
* [ ] **但必须改造**：所有 Conv/Pool/Attention 必须因果（AlphaGenome 默认非因果）([GitHub][2])
* [ ] 可插拔 attention bias 接口（未来可做“状态/波动/跨周期关系 bias”），参考 AlphaGenome 的 bias block 思路([GitHub][1])

---


[1]: https://raw.githubusercontent.com/google-deepmind/alphagenome_research/main/src/alphagenome_research/model/model.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/google-deepmind/alphagenome_research/main/src/alphagenome_research/model/convolutions.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/google-deepmind/alphagenome_research/main/src/alphagenome_research/model/attention.py "raw.githubusercontent.com"
