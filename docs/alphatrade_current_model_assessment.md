# AlphaTrade 当前模型评估与提升计划

**日期**: 2026-06-14  
**评估对象**: `alphatrade_v0.2_baseline_cffc6a0c`  
**主要依据**: `../alphatrade_runs/m5_formal_20260614_0938`  
**补充参考**: `../alphatrade_runs/m0_m9_verify_20260614_113744`、`../alphatrade_runs/productization_verify_20260614_2014`

---

## 1. 结论

当前模型已经跑通过一次正式 3-seed 训练和 M9 bundle 导出，训练稳定性基本合格：CUDA 训练、无 NaN/Inf、无 OOM、checkpoint 可导出、批量推理链路已验证。

但从模型质量看，当前版本还不能视为可交易模型。核心问题不是工程链路，而是预测质量：

1. **方向性接近 0**：正式 best seed 的 overall IC 为 `-0.00027`，rank IC 为 `0.00053`；3-seed 均值 IC 也只有 `0.00133`。
2. **分位覆盖严重失真**：正式 3 seeds 的 q10/q30 coverage 都是 `0`，q90 coverage 都是 `1`；best seed q50 coverage 只有 `0.2828`。模型输出满足非交叉，但区间校准明显不对。
3. **seed 方差不小**：formal 3 seeds 的 primary loss 从 `0.00295` 到 `0.004996`，说明训练结果对初始化仍敏感。
4. **当前验证主要是 val split**：还缺完整 test/backtest 口径，productization backtest 目前只是 100-row smoke，不应作为收益结论。

因此当前最合理定位是：**M0-M9 工程闭环已经形成，模型还处在 baseline 阶段。下一步应优先修评估和校准，再做特征/训练/结构扩展。**

---

## 2. 训练与产物概况

正式训练 run:

| 项 | 值 |
|---|---|
| run root | `../alphatrade_runs/m5_formal_20260614_0938` |
| leaderboard generated_at | `2026-06-14T09:56:37` |
| experiment | `baseline` |
| seeds | `42, 43, 44` |
| champion run_id | `cffc6a0c` |
| champion seed | `43` |
| model_version | `alphatrade_v0.2_baseline_cffc6a0c` |
| train git_sha | `655ec59` |
| train device | `cuda` |
| train samples | `387,673` |
| val samples | `74,459` |
| symbols | `24` |
| lookback | `60` |
| horizons | `1, 5, 20, 60` |
| quantiles | `0.1, 0.3, 0.5, 0.7, 0.9` |
| model params | `6,274,774` |
| max_steps | `500` |
| batch_size | `128` |

M9 bundle:

| 项 | 值 |
|---|---|
| bundle_path | `../alphatrade_runs/m5_formal_20260614_0938/artifacts/model_bundle/alphatrade_v0.2_baseline_cffc6a0c` |
| checkpoint_step | `500` |
| M9 inference smoke | 2 symbols, 690 rows, 23 columns |
| M9 inference throughput | about `27 samples/sec` in historical smoke |

说明：正式 M9 inference metrics 是产品化 schema 变更前生成的，缺少后来新增的 `batch_size` 和 `jax_backend` 字段。等 GPU 空闲后需要重新跑 M9 export/infer/backtest/strict gate。

---

## 3. Formal 3-Seed 指标

Leaderboard:

| exp_id | primary_mean | primary_std | primary_best | best_run_id |
|---|---:|---:|---:|---|
| baseline | `0.004000` | `0.001025` | `0.002948` | `cffc6a0c` |

Seed-level eval:

| seed | run_id | overall pinball | h1 | h5 | h20 | h60 | IC | rank IC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | `b653c3f0` | `0.004057` | `0.006558` | `0.003007` | `0.003222` | `0.003440` | `0.002238` | `-0.002064` |
| 43 | `cffc6a0c` | `0.002948` | `0.002877` | `0.003023` | `0.003099` | `0.002793` | `-0.000274` | `0.000528` |
| 44 | `00f8978d` | `0.004996` | `0.005061` | `0.005755` | `0.004193` | `0.004975` | `0.002027` | `0.002155` |

3-seed aggregate:

| metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| overall pinball | `0.004000` | `0.000837` | `0.002948` | `0.004996` |
| h1 pinball | `0.004832` | `0.001511` | `0.002877` | `0.006558` |
| h5 pinball | `0.003928` | `0.001292` | `0.003007` | `0.005755` |
| h20 pinball | `0.003505` | `0.000489` | `0.003099` | `0.004193` |
| h60 pinball | `0.003736` | `0.000915` | `0.002793` | `0.004975` |
| IC | `0.001330` | `0.001138` | `-0.000274` | `0.002238` |
| rank IC | `0.000206` | `0.001737` | `-0.002064` | `0.002155` |

Coverage:

| quantile | expected | seed42 | seed43 | seed44 | 3-seed mean |
|---|---:|---:|---:|---:|---:|
| q10 | `0.10` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| q30 | `0.30` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| q50 | `0.50` | `0.0000` | `0.2828` | `0.0000` | `0.0943` |
| q70 | `0.70` | `0.0000` | `1.0000` | `0.9539` | `0.6513` |
| q90 | `0.90` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |

这个 coverage 形态说明当前分位输出的绝对位置和尺度不可信。即使 pinball loss 数值较低，模型也可能主要学到了一组偏置/尺度，而没有形成有效条件分布。

---

## 4. Symbol 层面观察

按 3-seed 平均 pinball，较好和较差的品种如下：

| 较好 symbol | avg pinball | avg IC |
|---|---:|---:|
| `CFFEX.T` | `0.003902` | `0.0186` |
| `SHFE.PB` | `0.003947` | `-0.0137` |
| `SHFE.AL` | `0.003954` | `0.0024` |
| `SHFE.CU` | `0.003955` | `-0.0081` |
| `SHFE.AU` | `0.003966` | `-0.0099` |

| 较差 symbol | avg pinball | avg IC |
|---|---:|---:|
| `CZCE.RM` | `0.004050` | `-0.0351` |
| `CZCE.OI` | `0.004050` | `-0.0338` |
| `CZCE.FG` | `0.004161` | `-0.0239` |
| `DCE.J` | `0.004120` | `0.0153` |
| `DCE.JM` | `0.004194` | `-0.0095` |

按 IC 看，存在少数正 IC 品种：

| symbol | avg IC |
|---|---:|
| `DCE.I` | `0.0537` |
| `CZCE.CF` | `0.0378` |
| `DCE.P` | `0.0317` |
| `CZCE.MA` | `0.0310` |
| `CFFEX.T` | `0.0186` |

这说明信号可能不是完全不存在，但当前统一模型没有稳定地把信号抽出来。后续应使用 symbol/horizon 分层指标作为主要诊断，而不是只看 overall pinball。

---

## 5. 补充 ablation 结果

`../alphatrade_runs/m0_m9_verify_20260614_113744` 有一组 baseline/batch/no_clip/steps 对比：

| exp_id | primary_mean | primary_std | primary_best |
|---|---:|---:|---:|
| `batch_256` | `0.133383` | `0.011707` | `0.123102` |
| `baseline` | `0.134178` | `0.010631` | `0.125448` |
| `steps_1000` | `0.134178` | `0.010631` | `0.125448` |
| `no_clip` | `0.135976` | `0.010437` | `0.129604` |

注意：这组结果与 formal run 的 loss 数值尺度不同，不建议直接跨 run 比绝对值。只能作为相对信号：

- `batch_256` 略优于 baseline，但幅度很小。
- `steps_1000` 与 baseline 完全一致，疑似复用了同配置/同结果，不能证明加步数有效。
- `no_clip` 略差，梯度裁剪仍应保留。

---

## 6. 当前主要瓶颈

### 6.1 评估口径还不够

当前正式结论主要来自 val split。缺少：

- 完整 test split 评估。
- 完整时间段回测。
- naive baseline 对照，例如 zero-return、rolling empirical quantile、linear baseline。
- 交易成本、换手、阈值、品种分层收益。

没有这些，无法判断 pinball loss 降低是否转化为交易可用信号。

### 6.2 分位校准是第一优先级

coverage 失真非常明显。可能原因：

- target scaling 或反归一化口径有问题。
- quantile loss 权重导致模型牺牲尾部分位。
- 输出头学到了单调但偏移严重的分布。
- 不同 horizon/品种的 target scale 差异太大，统一 loss 被少数尺度主导。

这类问题会直接影响策略阈值、风险估计和回测稳定性。

### 6.3 方向信号弱

IC 接近 0，说明模型的 median 预测目前基本没有方向排序能力。后续不能只优化 pinball loss，需要把 IC/rank IC、direction hit rate、backtest metrics 纳入 champion gate。

### 6.4 特征表达偏弱

当前输入只有 8 个 1m OHLCV 派生特征，lookback=60。对于期货跨品种预测，可能缺少：

- 波动率归一化特征。
- 多周期动量/反转特征。
- 成交量/持仓变化的 rolling z-score。
- session/日内结构特征。
- symbol embedding、品种板块/交易所 embedding。
- 主力合约切换、期限结构、跨品种相关性特征。

---

## 7. 后续提升路线

### P0: 先把评估口径补硬

目标：确保后续每次实验都能判断“真的变好”，不是 loss 偶然变小。

1. 重新跑 M9 productization flow，使用当前 schema 记录 `batch_size` 和 `jax_backend`。
2. 跑完整 validation/test inference，不只 smoke。
3. 跑完整 backtest：
   - horizon: `1, 5, 20, 60` 分别测。
   - signal: q50、q70/q30 spread、q90-q10 uncertainty 分别测。
   - cost: `0, 1, 2, 5 bps`。
   - thresholds: `0, 0.5sigma, 1sigma`。
4. 加 naive baseline：
   - zero-return quantile。
   - rolling historical quantile。
   - simple linear/ridge median predictor。
5. champion gate 至少包含：
   - pinball overall/by_horizon。
   - coverage calibration error。
   - IC/rank IC by horizon。
   - backtest net return、drawdown、turnover、hit rate。

### P1: 修分位校准

1. 检查 target scaling：
   - 每个 horizon 的 target mean/std/quantile。
   - 按 symbol/horizon 的 target scale。
   - 训练输出是否在 target 的合理范围内。
2. 校准 quantile loss：
   - 增加 q10/q90 和 q30/q70 的权重实验。
   - 分 horizon loss weight，避免短 horizon 主导。
   - 增加 coverage calibration report。
3. 尝试 post-hoc calibration：
   - 在 val 上对每个 horizon 做 monotonic quantile calibration。
   - 不改变模型结构，先验证 coverage 能否被修正。

### P2: 增强信号特征

优先加低风险、可解释、便于回测归因的特征：

1. rolling volatility normalized returns：`ret / rolling_vol`。
2. multi-scale returns：`ret_5m, ret_15m, ret_60m`。
3. rolling range/volume z-score。
4. time/session features：夜盘、开盘/收盘附近、交易所 session。
5. symbol embedding：让统一模型区分品种行为。
6. sector/exchange embedding：金属、黑色、农产品、能化、金融期货。

### P3: 训练和结构实验

建议实验矩阵：

| 实验 | 目的 |
|---|---|
| `steps_2000` / `steps_5000` | 判断当前 500 step 是否训练不足 |
| `batch_256` / `batch_512` | 降低梯度噪声，前提是 GPU 空闲且显存允许 |
| `lr_3e-4` / `lr_5e-5` | 检查学习率是否限制收敛 |
| `weighted_quantile_loss` | 修 tail coverage |
| `symbol_embedding` | 解决跨品种混合问题 |
| `vol_scaled_targets` | 降低 horizon/symbol scale 混杂 |
| `larger_lookback_240` | 测试更长上下文是否增加方向性 |

每个实验至少 3 seeds，先 smoke，再 formal。不要只看 single seed。

---

## 8. 推荐下一步

最优先的 5 件事：

1. 等 GPU 空闲后重跑 M9 productization flow，生成当前 schema 的 infer/backtest/validation 产物。
2. 增加 naive baseline 和完整 backtest，对当前 champion 给出可交易性基线。
3. 针对当前 champion 做 target/output 分布诊断，定位 coverage 失真原因。
4. 先做 `vol_scaled_targets + coverage calibration` 实验，再做大模型。
5. 把 champion 选择从 `pinball_loss.overall` 升级为多指标 gate。

把握评估：

- 工程产品化继续推进：`8/10`。M0-M9 产物隔离、bundle、schema、推理、回测入口已经成型。
- 修好校准问题：`6/10`。coverage 问题很明确，通常可以通过 target scaling/loss/calibration 定位。
- 获得稳定可交易 alpha：`4/10` 到 `5/10`。当前 IC 接近 0，必须靠更强特征、分层建模和严格回测验证，不能假设只加训练步数就能解决。

