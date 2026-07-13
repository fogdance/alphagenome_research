# MG0-A AlphaGenome Official Code Parity Audit

## 0. 结论与边界

- 审计状态：`FROZEN_WITH_NEEDS_VERIFICATION`
- 冻结 ID：`MG0-A-v1`；冻结时间：`2026-07-13T14:10:05+08:00`；评审决定：`APPROVED_WITH_DOCUMENTED_GAPS`。
- 冻结范围：只冻结本报告的静态审计结论、36 个组件映射、原则、非同构项、不确定性和 gate；不冻结任何训练或 GPU 运行结果。
- 代码基线：官方 `google-deepmind/alphagenome_research` commit [`232fc695d1eab27bac9e94bcd4b50499139ba4e1`](https://github.com/google-deepmind/alphagenome_research/tree/232fc695d1eab27bac9e94bcd4b50499139ba4e1)，本地 merge HEAD `ed738d92ab76df03b4d25ebb741e33048c9aae09`。
- 本地 parity：`upstream/main` 是当前 HEAD 的祖先，且 `git diff upstream/main..HEAD -- src/alphagenome_research` 为空。
- 审计覆盖：模型、attention/pair stack、heads/losses、schema/masking、公开 augmentation、IO/eval、finetuning、variant scorers、ISM wrapper、测试，以及论文中的 fold、teacher、distillation、ablation、compute 设计。
- 关键限制：官方研究仓库没有发布论文所述的 from-scratch pretraining、all-fold teacher、distillation 或 sequence-parallel trainer。本报告只把它们记录为论文已发布行为，并标记 `NEEDS_VERIFICATION`；不声称已经取得代码级 parity。
- 执行边界：本阶段不实现 MarketGenome，不下载权重/数据，不运行完整模型、训练或 GPU 推理。所有训练相关工作与 GPU 工作按用户要求延后；`2026-07-14` 只是最早可能具备算力的日期，不构成自动恢复授权。
- GPU 记录：用户要求延期前曾启动一次 JAX 测试尝试；收到 GPU 已满的提醒后立即中断，未把其部分输出作为验证结果。此后所有保留的模型探针均强制 CPU，报告/schema 测试不加载 JAX。

状态含义：

- `REUSE_PATTERN`：保留研究或工程模式，不复制生物语义。
- `CAUSALIZE`：结构可借鉴，但必须改成严格的过去到现在信息流。
- `REDESIGN`：输入、目标、扰动或统计语义必须为市场重新定义。
- `DEFER`：在 learnability、数据契约或计算预算证明必要前不实现。

证据类型：

- `CODE_OBSERVED`：当前固定 commit 中直接观察到的实现。
- `PUBLISHED_ONLY`：Nature 正文或 Supplementary Methods 描述，但仓库无相应 trainer 实现。
- `DESIGN_INFERENCE`：MarketGenome 候选映射，不是 AlphaGenome 的已发布结论。

## 1. 证据来源

1. [Nature 正文：Advancing regulatory variant effect prediction with AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0)
2. [Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-10014-0/MediaObjects/41586_2025_10014_MOESM1_ESM.pdf)
3. [Supplementary Tables](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-10014-0/MediaObjects/41586_2025_10014_MOESM3_ESM.xlsx)
4. [官方研究代码固定 commit](https://github.com/google-deepmind/alphagenome_research/tree/232fc695d1eab27bac9e94bcd4b50499139ba4e1)

重点代码路径：

- `src/alphagenome_research/model/{model,attention,convolutions,layers,embeddings,heads,losses,augmentation,schemas,splicing,dna_model}.py`
- `src/alphagenome_research/model/{variant_scoring,interval_scoring}/**/*.py`
- `src/alphagenome_research/io/{bundles,dataset,fasta,genome,splicing}.py`
- `src/alphagenome_research/evals/{track_prediction,regression_metrics}.py`
- `src/alphagenome_research/finetuning/{dataset,finetune}.py`
- 上述模块对应的 `*_test.py`。

## 2. Input representation and organism conditioning

### 已观察事实

- `schemas.DataBatch` 接收 `dna_sequence: [B,S,4]` one-hot DNA 和 `organism_index: [B]`，并携带各类 targets/masks（`schemas.py:25-52`）。
- `AlphaGenome.__call__` 在 encoder trunk 注入 organism embedding（`model.py:216-224`）；`OutputEmbedder` 与 `OutputPair` 再注入 organism embedding（`embeddings.py:58-78,93-106`）。
- 各 head 使用 organism-specific weight/bias（`heads.py:266-295`）。不同 organism 必须共享 output types，track 数由 metadata padding 对齐（`model.py:140-166`; `heads.py:409-438`）。
- 论文输入长度是 `2^20 = 1,048,576 bp`，同时训练 human 和 mouse。这个长度是已发布的基因组设置，不是市场最优 lookback 的证据。

### MarketGenome 设计推论

市场输入必须重新定义为严格 `<= as_of` 的 bar/features、availability mask、session/roll/gap boundary 与 identity/context。symbol、exchange、sector 或 regime conditioning 可以借鉴 embedding 模式，但 regime 只能由过去数据计算，并必须做 symbol-held-out 和 identity-shuffle 测试。

## 3. Encoder, downsampling, convolution and normalization

### 已观察事实

- `DnaEmbedder`：kernel 15 Conv1D 到 768 channels，再接 kernel 5 residual `ConvBlock`（`convolutions.py:106-118`）。
- `SequenceEncoder`：stem 后先 pool 一次，再依次执行 6 个 `DownResBlock`；每个 block 后再 pool，共 7 次 `MaxPool(window=2,stride=2)`。保存 `/1,/2,/4,/8,/16,/32,/64` 七个 skips，trunk 为 `/128`（`model.py:38-55`; `layers.py:31-52`）。对应 channels 为 `768,896,1024,1152,1280,1408,1536`，trunk `/128` 保持 1536。
- `DownResBlock` 每级增加 128 channels，并含两个 kernel 5 residual conv（`convolutions.py:121-137`）。
- `StandardizedConv1D` 进行 weight standardization 并使用 `padding='SAME'`（`convolutions.py:54-103`）。这不是因果卷积。
- `RMSBatchNorm` 训练态统计跨 batch 和全部 spatial axes，EMA decay 0.9；推理态使用 EMA（`layers.py:55-99`）。per-token `LayerNorm/RMSNorm` 只归一化最后一维（`layers.py:102-142`）。
- `AlphaGenome.loss` 未显式传 `is_training=True`，仓库中也未观察到 from-scratch trainer 调用，因此公开 loss path 的 BatchNorm 训练意图是 `NEEDS_VERIFICATION`（`model.py:286-300`）。

### 静态因果证据

`StandardizedConv1D`/`MaxPool` 的 `SAME` padding 没有建立 past-only contract；`RMSBatchNorm` 在训练态把 sequence axis 纳入同一个方差统计。仅凭这些代码路径即可确认它们不能原样进入严格因果市场模型。运行时因果探针未作为冻结证据；训练与 GPU/JAX 验证均延期。

### MarketGenome 设计推论

卷积必须显式 left padding；pool/stride 的输出时间戳必须定义为其覆盖窗口终点；训练态 normalization 只能使用单 token channel statistics 或已经冻结的 past-only statistics。每一级 skip、trunk 和 decoder 都要通过 full-vs-prefix equality 测试。

## 4. Transformer, pair updates and attention bias

### 已观察事实

- `TransformerTower` 有 9 个 residual MHA+MLP blocks，并在 block `0,2,4,6,8` 前执行 `PairUpdate`（`model.py:76-90`）。
- `MHABlock` 是 8 query heads、共享 1 个 K/V（multi-query）；Q/K 维 128，V 维 192，使用 RoPE、pair-derived bias、`tanh` logits soft-cap 5，BF16 dot/F32 accumulate（`attention.py:116-174`）。
- attention 对完整 sequence 做 softmax，没有 triangular causal mask（`attention.py:147-161`）。
- `SequenceToPair` 将 `/128` sequence 再按 16 mean-pool，得到 `/2048` pair representation；加入 signed relative-position features（`attention.py:224-283`）。
- `RowAttentionBlock` 在 pair row 上做完整 softmax；`AttentionBiasBlock` 将 pair state 投影成 8-head sequence attention bias（`attention.py:177-221`）。
- `PairUpdateBlock` 只调用 `RowAttentionBlock` 和 `PairMLPBlock`，公开实现中没有 column-attention path（`attention.py:286-299`）。对论文使用的 1 Mb 输入，`/2048` pair grid 为 512x512；本冻结结论不采用未经本次审计独立定位的实验动机措辞。

### MarketGenome 设计推论

MHA 必须增加 future mask、session/roll/gap mask，并用 market-time-aware positions。hard causal mask 必须在 `tanh` soft-cap 之后、softmax 之前应用，或使用真正的 masked softmax；若把 `-inf` 当作现有 attention bias 在 soft-cap 前相加，它会被截成有限值，不能保证未来权重为零。pair state 若未来启用，必须是有方向的 lower-triangular time-to-time 或 anchor relation，并在 row softmax 前屏蔽未来列。MG0 后的 tiny baseline 默认关闭 pair stack。

## 5. Decoder and skip connections

### 已观察事实

- `SequenceDecoder` 以 7 个 `UpResBlock` 从 `/128` 恢复到 `/1`，逐级融合 encoder skips（`model.py:58-73`）。
- `UpResBlock` 包含 kernel 5 transform/residual、repeat-by-2 upsample、初值 0.1 的 learned residual scale、pointwise skip projection 和 kernel 5 output conv（`convolutions.py:140-170`）。
- channel 路径为 `1536 -> 1536 -> 1408 -> 1280 -> 1152 -> 1024 -> 896 -> 768`。

### MarketGenome 设计推论

repeat upsample 会把同一个 coarse token 放入两个 fine children。若 coarse token 聚合了整个 bin，前一个 child 会看到 bin 内未来信息。必须用 terminal-aligned timestamp/delay 规则并逐级做 prefix invariance；不能只把 conv 改成 causal 就宣称 decoder causal。

## 6. Embeddings and resolutions

`Embeddings` 明确承载三种张量（`embeddings.py:25-40`）：

| Embedding | AlphaGenome shape | 分辨率 | MarketGenome 结论 |
|---|---:|---:|---|
| sequence | `[B,S,1536]` | 1 bp | 保留 typed dense embedding 模式；内容重设并因果化 |
| sequence_128bp | `[B,S/128,3072]` | 128 bp | 保留 multi-resolution 模式；时间戳必须对齐到窗口终点 |
| pair | `[B,S/2048,S/2048,128]` | 2,048 bp | 对称形式无市场同构，默认 `DEFER` |

`OutputEmbedder` 将 channels 加倍、可融合 repeat-upsampled coarse skip、归一化并注入 organism embedding（`embeddings.py:43-78`）。`OutputPair` 显式 symmetrize pair tensor（`embeddings.py:81-106`）；市场时间有方向，不能复制这种对称性。

## 7. Heads, output resolutions, scaling and losses

### Head inventory

官方 `HeadName` 有 11 项（`heads.py:54-67`）。Supplementary Table 1 的 track 数为：

| Head/output | Human / mouse tracks | 代码输出分辨率 | scaling / activation | loss |
|---|---:|---|---|---|
| ATAC | 167 / 18 | 1, 128 bp | nonzero mean; softplus | Poisson total + 5x positional multinomial |
| DNASE | 305 / 67 | 1, 128 bp | nonzero mean; softplus | 同上 |
| PROCAP | 12 / 0 | 1, 128 bp | nonzero mean; softplus | 同上 |
| CAGE | 546 / 188 | 1, 128 bp | nonzero mean; softplus | 同上 |
| RNA_SEQ | 667 / 173 | 1, 128 bp | nonzero mean; power 0.75; soft clip; inverse on output | 同上 |
| CHIP_TF | 1,617 / 127 | 128 bp | nonzero mean; softplus | 同上 |
| CHIP_HISTONE | 1,116 / 183 | 128 bp | nonzero mean; softplus | 同上 |
| CONTACT_MAPS | 28 / 8 | 2,048 bp pair | organism-specific linear | masked MSE |
| SPLICE_SITES | 4 / 4 tracks, 5 classes | 1 bp | 5-way softmax | categorical CE |
| SPLICE_SITE_USAGE | 734 / 180 | 1 bp | sigmoid | BCE |
| SPLICE_JUNCTIONS | 734 / 180 | selected-site pair | softplus counts | donor/acceptor ratio CE + 0.2 Poisson; outer weight 0.2 |

代码证据：head configs（`heads.py:137-237`）、genome-track predictions/loss（`heads.py:474-858`）、contact（`heads.py:861-915`）、splice classification（`heads.py:918-963`）、usage（`heads.py:966-1017`）、junction（`heads.py:1020-1246`）。`AlphaGenome.loss` 对 `HeadConfig.loss_weight * loss` 求和（`model.py:286-300`）。

### Scaling and masking

- Genome targets 除以每 track 的 nonzero mean 和 resolution；RNA 可做 `x^0.75`，10 以上 soft clip，prediction path 有显式 inverse（`heads.py:298-360,537-546,597-635`）。
- Genome track loss 将 1 Mb 切成 `2^17 bp` segments；每段预测 total count 的 Poisson loss，并以 5 倍 positional multinomial loss 约束空间分布（`heads.py:671-690`; `losses.py:62-121`）。各 resolution loss 相加（`heads.py:817-858`）。
- `safe_masked_mean` 对全部 masked 数据返回有限的 0 分母保护（`losses.py:26-41`）。
- optional cross-track gene loss 默认 `gene_loss_weight=0`，`create_head` 没有打开它，不能把它写成默认 AlphaGenome loss（`heads.py:85-104,482-535,708-814`）。

### MarketGenome 设计推论

保留“dense multi-task + typed head registry + per-resolution mask + total/path decomposition”模式。所有实际 target、scaler、activation 和 loss 必须重设：signed returns 不能用 softplus，统计量只能在 train fold 拟合，需保存 scaler hash 和 inverse contract。contact-map truth 和 splice-junction ontology 在市场中尚不存在，均延后。

## 8. Data schema, loader and masking

### 已观察事实

- `DataBatch` 包含 DNA、organism、7 类 dense genome tracks 及 masks、contact map、junction matrix/site positions、site usage、site classes、gene mask（`schemas.py:25-52`）。
- `get_genome_tracks` 要求 target 与 mask 同时存在，否则抛错（`schemas.py:60-86`）。schema 没有通用 input padding/attention mask。
- `bundles.py` 定义 bundle dtype 与原生 1/128/2048 resolutions。
- `io/dataset.py` 根据 organism/fold/subset/bundle 构造 TFRecord 路径；不同 bundle shard 数必须一致后 zip，`_parse_batch` 合并并写入 organism index（`dataset.py:163-232`）。公开 iterator 默认 `FOLD_0/VALID`（`dataset.py:235-255`）。
- `evals/track_prediction.py` 从 Kaggle checkpoint 载入模型，按 data axis JIT，crop prediction 到 target length，再用 mask 累积 regression metrics（`track_prediction.py:67-173`）。

### MarketGenome 设计推论

新 batch contract 必须包含 base-feature availability、每 modality/resolution target mask、warmup/valid-time、session/contract-roll/gap boundary、identity/context、fold/as-of provenance。全部 padding 和 missing labels 必须产生零梯度；数据 loader 必须验证 `source_timestamp <= as_of`。

## 9. Augmentation

### CODE_OBSERVED

公开 `augmentation.py` 只实现 prediction/output reverse-complement：1D reverse + strand reindex、contact 双轴 reverse、junction position/strand reindex，以及 DNase 一格 alignment 修正（`augmentation.py:31-122`）。`dna_model._predict`/`_predict_variant` 用它恢复 negative-strand 输出（`dna_model.py:165-185,278-284`）；interval/variant scoring 分别传入负链 mask，并在已恢复方向的预测进入 scorer 前把 interval 转为 unstranded（`dna_model.py:628-638,806-807,904-918,935-936`）。这不是公开的训练 input augmentation pipeline。

### PUBLISHED_ONLY

Supplementary Methods 的 “Pre-training” 与 “Distillation”（pp. 19-21）描述 pretraining 使用 1 Mb window、均匀 shift `[-1024,+1024] bp` 和 50% reverse complement；distillation 另加 4% 随机 nucleotide substitutions，以及每个样本 `Poisson(1)` 个 1-20 bp insertion/deletion/inversion。当前仓库未发现实现这些训练采样与 mutation 的 trainer，故标记 `NEEDS_VERIFICATION`。

### MarketGenome 设计推论

reverse complement/time reversal 无市场同构，禁止使用。候选扰动必须保持 OHLC、volume/OI、session 与 roll 合法，并且只能编辑 `<= as_of` history；不得制造或修改真实未来标签。

## 10. Pretraining folds

### PUBLISHED_ONLY

Supplementary Methods 的 “Dataset splitting and cross-validation”（p. 9）说明沿用 Borzoi 的 4 folds：每个 genome 划为 8 个互斥 sections，每 fold 为 6 train、1 validation、1 test。约 196 kb target interval 扩成 1 Mb window 后，任何与 validation/test 1 Mb window 重叠的 train window 都被移除。四个 fold-specific models 用于 held-out track evaluation，模型与数据选择冻结后 test 只评一次。

公开代码只消费由 `alphagenome.data.fold_intervals`/`ModelVersion` 提供的 fold/subset 和已生成 TFRecords；没有 fold construction 或 pretraining loop（`io/dataset.py:65-82,163-184,235-255`）。因此 split 描述是论文证据，不是本仓库代码 parity。

### MarketGenome 设计推论

映射为 purged walk-forward folds：training/validation/test 严格按时间，purge/embargo 至少覆盖最大 lookback、label horizon 和 execution delay；test 只在设计冻结后使用一次。基因组 section 随机/空间互斥不能原样映射为市场 split。

## 11. All-fold teachers and distillation

### PUBLISHED_ONLY（Supplementary Methods “Pre-training”/“Distillation”, pp. 19-21）

- 论文训练 64 个 all-fold teachers，使用全部 8 genome sections，没有 reference-genome holdout；因此不用于 track generalization 声明，只给 distillation/variant work。
- 最终 student 与 pretrained architecture 相同。64 张 H100 各放一个冻结 teacher，每个 GPU 为本地随机 interval 生成 target；student 参数跨设备复制并平均梯度。这是在 sample/step 上近似 teacher ensemble expectation，不是每个样本同时跑 64 teachers。
- 必须与 Fig. 7c teacher-count ablation 区分：该 ablation 使用 64 个独立 Fold-0 replicas 作为 teacher pool，比较 1/4/64 teachers 和 1-4 model ensembles。
- 当前仓库没有 teacher construction、teacher sampling、distillation loss 或 optimizer schedule trainer，全部标记 `NEEDS_VERIFICATION`。

### MarketGenome 设计推论

可以保留“严格 held-out student evaluation + all-training-window teacher only for pseudo-target generation”的模式，但 teacher 必须只访问其训练截止时间内数据；不能用跨 test future 的 all-fold teacher 产生评估期标签。是否需要 teacher 数量由 MG7 ablation 决定，不能把 64 当成市场超参数。

## 12. Variant scoring and in-silico mutagenesis

### CODE_OBSERVED

- `_predict_variant` 对 REF/ALT 使用同一模型；对 indel 做 ALT alignment；以 REF/ALT/annotation union 选择 splice candidates，再进行 paired junction pass（`dna_model.py:189-289`; `variant_scoring/variant_scoring.py:179-357`; `splicing.py:59-77`）。
- `score_variant` 构造 center/gene/junction/indel masks，调用 scorer，附加 metadata，并在可用时加 quantile calibration（`dna_model.py:836-975`）。最新 upstream 修复在评分前将 reverse-complemented prediction 的 interval 变为 unstranded（`dna_model.py:807,936`）。
- scorer 不是一个通用差分：center mask 支持 DIFF/ACTIVE mean/sum、L2、log transforms；gene-mask、contact-map、polyadenylation、splice-junction 各有专用 aggregation（`model/variant_scoring/*.py`）。
- `score_ism_variants` 枚举 ISM variants 后复用 variant scorer（`dna_model.py:977-1020`）。枚举器来自外部 `alphagenome.interpretation.ism`，不在本 research repo 内（`dna_model.py:30`），所以 wrapper 是 code-observed，完整 ISM implementation parity 是 `NEEDS_VERIFICATION`。
- Supplementary Methods 的 “Variant scoring” 与 “In Silico Mutagenesis for Contribution Scores”（pp. 25-29）说明 19 个推荐 scoring configurations，并用 chr22 上 348,126 个 MAF > 0.01 common SNPs 做 scorer/track empirical quantile calibration。ISM 对每个位置的三种替代 SNV 建 `L x 4` effect matrix，逐位置 mean-center，再读取 reference-base contribution。

### MarketGenome 设计推论

保留“同一模型、共同 context、reference vs legal edited-history、per-track delta、mask/aggregation/calibration”的 effect-scoring contract，但不能把 delta 称为交易或盈利因果。market ISM 必须是 domain-valid、past-only perturbation library；任意替换 K 线数值不合法。

## 13. Published ablations

以下均为 Nature Fig. 7 与 Supplementary Methods “Model and data ablations”（pp. 41-42）的已发布 AlphaGenome 结果；MarketGenome 只能继承“必须做这些 ablation”的原则，不能继承最优值。

| Ablation | AlphaGenome 已发布设置/观察 | MarketGenome 必须验证 |
|---|---|---|
| target resolution | 高分辨 tracks 比较 1/2/8/32/128 bp；ChIP 128、contact 2,048 固定；1 bp 对 fine tasks 最好 | bar resolution、head horizon 与 execution resolution，含成本后的 OOS 指标 |
| sequence length | 8k/32k/131k/512k/1M，分别做 fixed-train、fixed-inference 与 matched；1M train+inference 总体最佳 | lookback sweep；每个 lookback 重新 purge；不能从 1 Mb 推导市场窗口 |
| teacher count | 1/4/64 teachers；同时比较 1-4 model ensemble | 0/1/N teachers 与 ensemble，固定 compute/data 后比较 |
| modalities | full、single-group、leave-one-group-out、cumulative modality order | single/leave-one-out/cumulative market modalities；检查泄漏与数据可用性 |
| mutations | 无 mutation student 的部分 variant metrics 下降 | legal perturbation on/off；不能用生物 mutation 幅度 |

## 14. Compute and sequence parallelism

### PUBLISHED_ONLY（Supplementary Methods “Sequence parallelism”/“Model training parameters”, pp. 19-21）

- 模型约 450M parameters。
- pretraining：batch 64，每 sample 8-way sequence parallel，共 512 TPU v3 cores；15,000 steps，约 4 小时。
- sequence parallel：`2^20` 切成 8 个 `2^17` 子序列；encoder 每侧复制 1,024 bp overlap，降到 128 bp 后裁每侧 8 tokens；transformer all-gather K/V，pair 只沿第一 sequence axis 分片，decoder 镜像 overlap/trim。
- distillation：64 H100、batch 64、250,000 steps、约 3 天，不使用 sequence parallel。

当前仓库没有上述 distributed trainer/partition implementation。`evals/track_prediction.py` 只有 data-parallel inference，不能作为 sequence-parallel 代码证据。MarketGenome 在 `mg_tiny` profiling 和 learnability gate 前不复制 450M、TPU/H100 或 pair O(L^2) 预算。

## 15. Finetuning extension point

公开 `finetuning/finetune.py` 提供 fold-based dataset iterator（`finetune.py:36-91`）、固定构造 `AlphaGenome(..., freeze_trunk_embeddings=True).loss` 的 transformed forward（`finetune.py:94-118`），以及 Optax train step（`finetune.py:121-153`）。实际 stop-gradient 位于 `model.py:243-244`。测试代码分别演示 base params 与新 head params 的局部合并和 train step（`finetune_test.py:54-127`），以及只保存新初始化 fine-tune params/state 后调用 `dna_model.create`（`finetune_test.py:128-175`）；二者不是统一的端到端合并 checkpoint 流程，也不是 `finetune.py` 提供的通用用户-head/checkpoint API。这是下游 fine-tuning extension point，不是论文 pretraining/distillation trainer；训练执行全部延期。

## 16. Component map

规范 machine-readable map 位于 `MG0_ALPHAGENOME_PARITY_AUDIT.json`，它是冻结后的 canonical contract。下表是 36 项的一一对应人工可读摘要；若风险或验证测试措辞存在粒度差异，以 JSON 为准。每行都把官方行为与市场设计推论分开。

| AlphaGenome component | Official code evidence | Scientific role | MarketGenome candidate analogue | Required causal modification | Status | Risk | Validation test |
|---|---|---|---|---|---|---|---|
| DNA one-hot input | `schemas.py:25-29`; `model.py:198-215` | encode nucleotide identity | typed bar/feature channels + availability | all values/provenance `<= as_of`; explicit missing mask | REDESIGN | feature/label leakage | source timestamp audit; future suffix perturbation |
| organism conditioning | `model.py:216-224`; `embeddings.py:58-78`; `heads.py:266-295` | human/mouse conditional regulation | symbol/exchange/sector/session identity | past-only regime; unseen-identity fallback | REUSE_PATTERN | identity memorization | identity shuffle and symbol-held-out |
| DNA stem | `convolutions.py:106-118` | local motif extraction | local microstructure feature stem | left padding only | CAUSALIZE | SAME-conv future leakage | per-position prefix equality |
| encoder/downsampling | `model.py:38-55`; `convolutions.py:121-137` | multi-scale context | causal temporal pyramid | terminal timestamps; causal pooling/stride | CAUSALIZE | within-bin future leakage | stage shape/time and prefix tests |
| standardized convolution | `convolutions.py:54-103` | stable convolution weights | standardized causal conv | replace SAME with explicit left pad | CAUSALIZE | alignment drift | impulse/future perturbation tests |
| RMSBatchNorm | `layers.py:55-99` | normalize batch/spatial activation | none by default | replace with per-token norm or frozen past-only stats | REDESIGN | train-mode global leakage | train/infer state and prefix tests |
| transformer tower/MQA/RoPE | `model.py:76-90`; `attention.py:116-174` | long-range sequence context | long-range temporal context | hard mask after soft-cap and before softmax; session/roll/gap masks | CAUSALIZE | future attention through logits/bias | full-vs-prefix every layer |
| pair update/attention bias | `attention.py:177-299` | sequence-pair feedback | directional time/anchor relations | lower-triangular directed relation; mask future columns before row softmax | DEFER | O(L^2), hidden bidirectional path | causal pair test and memory profile |
| decoder/skips | `model.py:58-73`; `convolutions.py:140-170` | recover dense outputs | dense multi-resolution temporal decoder | delayed terminal-aligned upsample + causal conv | CAUSALIZE | coarse-to-fine leakage | perturb every skip/coarse state |
| typed embeddings | `embeddings.py:25-78` | expose 1/128/pair resolutions | fine/coarse typed embeddings | causal fusion and timestamp contract | REUSE_PATTERN | resolution misalignment | shape/time/provenance tests |
| symmetric pair embedding | `embeddings.py:81-106` | unstranded pair representation | no direct counterpart | remove symmetry; require direction | DEFER | erases temporal direction | assert asymmetric directed contract |
| head registry/weights | `heads.py:44-237`; `model.py:286-300` | auditable multi-task outputs | typed market task registry | redesign tasks/weights, log exact resolved config | REUSE_PATTERN | silent task weighting | config/head/loss-weight snapshot |
| genome track heads | `heads.py:474-858` | dense assay prediction | dense return/path/vol/range/volume/event heads | signed/quantile/class-specific outputs | REUSE_PATTERN | positive-only biological activation | domain and gradient tests per head |
| target scaling | `heads.py:298-360,537-635` | normalize count tracks | train-fold scaler per task/horizon | train-only fit; frozen inverse contract | REDESIGN | validation/test contamination | roundtrip + scaler provenance/hash |
| genome track loss | `heads.py:671-690,817-858`; `losses.py:62-121` | total count + positional distribution | total movement + path-shape decomposition | signed robust/pinball likelihoods | REDESIGN | Poisson invalid for signed targets | all-masked, scale/path, signed tests |
| contact-map head | `heads.py:861-915` | predict physical DNA contacts | no validated market truth | define target before any implementation | DEFER | invented supervision | schema/learnability gate required |
| splice classification | `heads.py:918-963` | five-class site identity | dense phase/event class | redesign labels, imbalance and transition rules | REUSE_PATTERN | rare-event collapse | PR/calibration/transition legality |
| splice usage | `heads.py:966-1017` | site-use probability | completion/invalidation probability | past-only labels and censoring | REUSE_PATTERN | censoring/miscalibration | Brier, ECE, sparse-mask tests |
| splice junction head | `heads.py:1020-1246`; `splicing.py:25-77` | pair donor/acceptor sites | swing-anchor/event relation | require directed past-only candidate ontology | DEFER | top-k/P^2 instability | candidate/mask/permutation tests |
| DataBatch/masks | `schemas.py:25-86`; `io/dataset.py:211-255` | heterogeneous label availability | market batch and target masks | add input, warmup, boundary, fold/as-of masks | REDESIGN | padding/missing-label gradients | all-masked and boundary tests |
| multi-bundle TFRecord zip | `io/dataset.py:160-232` | align separately stored assay bundles | keyed market modality join | verify explicit sample identity before merge | REDESIGN | equal shard counts do not prove record alignment | deliberate identity mismatch must hard-fail |
| reverse complement output | `augmentation.py:31-122`; `dna_model.py:165-185,278-284,628-638,806-807,904-918,935-936` | strand equivariance | none | prohibit time reversal | DEFER | destroys arrow of time | config must reject reverse-time aug |
| pretraining shift/RC | Supplementary Methods; trainer absent | genomic invariance | time crop/jitter candidates | only past-preserving, session-valid transforms | REDESIGN | illegal time/context edits | augmentation invariant/provenance tests |
| four-fold pretraining | Supplementary Methods; `io/dataset.py:65-82` consumes split | held-out genomic evaluation | purged walk-forward folds | purge/embargo lookback+horizon+delay | CAUSALIZE | temporal overlap leakage | interval overlap audit |
| all-fold teachers | Supplementary Methods; trainer absent | stronger pseudo-target teachers | train-window-only teacher ensemble | no teacher access to evaluation future | REUSE_PATTERN | teacher leakage | teacher cutoff/provenance audit |
| distillation + mutations | Supplementary Methods; trainer absent | compress teacher ensemble and improve variants | teacher-student multi-task distillation | legal history edits; strict teacher cutoff | REDESIGN | pseudo-label/future contamination | teacher/student split and mutation legality |
| paired REF/ALT forward | `dna_model.py:189-289,836-975` | controlled variant effect | reference vs edited-history effect | domain-valid, past-only paired intervention | REDESIGN | effect mistaken for profit causality | identity delta 0; invariants/locality |
| scorer families | `model/variant_scoring/*.py` | modality-specific aggregation | per-task delta/active/path scorers | market target masks/aggregation contract | REDESIGN | cherry-picked scorer | preregister scorers; OOS calibration |
| quantile calibration | `dna_model.py:965-971`; calibration module | comparable empirical score ranks | train-only empirical calibration | calibration fold isolated from test | REUSE_PATTERN | calibration leakage | calibration provenance/coverage |
| ISM wrapper | `dna_model.py:977-1020`; external enumerator | local sequence contribution | legal local history sensitivity | preserve OHLC/session/roll; never touch future | REDESIGN | invalid counterfactual bars | invariant and future-unchanged tests |
| published ablations | Nature Fig. 7; Supplementary Methods | identify drivers of performance | resolution/lookback/modality/teacher ablations | walk-forward OOS + equalized budgets | REUSE_PATTERN | importing biological optimum | fixed protocol multi-seed ablation |
| sequence parallelism | Supplementary Methods; trainer absent | fit 1 Mb model across devices | possible long-context sharding | prove need after causal tiny baseline | DEFER | complexity and boundary errors | single-device parity + overlap tests |
| 450M compute regime | Supplementary Methods | capacity for genomic tasks | none initially | enforce tiny learnability/compute gate | DEFER | unjustified GPU cost | parameter/FLOP/memory budget gate |
| evaluation path | `evals/track_prediction.py:67-199` | held-out track metrics | typed masked offline evaluation | walk-forward, costs, latency, as-of audit | REUSE_PATTERN | metrics not trading utility | deterministic OOS evaluation |
| frozen-trunk fine-tuning path | `finetuning/finetune.py:36-153`; `model.py:243-244` | instantiate typed heads over stop-gradient embeddings | later frozen causal-trunk adaptation | retain causal encoder and frozen provenance | DEFER | treating noncausal embeddings as safe | causal checkpoint and trunk-gradient-zero/head-gradient-nonzero test |
| mixed precision | `dna_model.py:1217-1245`; `attention.py:147-168` | memory/throughput | BF16 compute after parity | FP32 reference and finite-gradient checks | REUSE_PATTERN | numeric drift | FP32/BF16 tolerance and determinism |

## 17. NON_NEGOTIABLE_ALPHA_GENOME_PRINCIPLES

`NON_NEGOTIABLE_ALPHA_GENOME_PRINCIPLES`:

1. Long context is an explicit ablation dimension; its value must be demonstrated, not assumed.
2. Dense, aligned, multi-resolution predictions are first-class outputs rather than only a terminal scalar.
3. Multi-task modalities use typed heads, explicit masks, declared scaling and inspectable loss weights.
4. Evaluation is held out, overlap-aware, and performed after design/model-selection choices are frozen.
5. Teacher training and student evaluation have distinct data-provenance contracts.
6. Effect scoring compares paired inputs under the same frozen model and reports per-task deltas.
7. Perturbation validity is domain-specific and must be tested before interpreting effects.
8. Target resolution, context length, modalities and teacher count require controlled ablations.
9. Missing labels and padded tracks must be explicitly masked and contribute no gradient.
10. Compute-heavy components require measured benefit and parity tests before adoption.

## 18. NON_ISOMORPHIC_COMPONENTS

`NON_ISOMORPHIC_COMPONENTS`:

1. Bidirectional DNA sequence context and unmasked attention: market predictions require a strict arrow of time.
2. Reverse complement and strand equivariance: reversing market time is not a valid augmentation.
3. Symmetric genomic contact-map targets: no market counterpart has a validated target contract.
4. DNA one-hot alphabet and nucleotide substitutions/indels: bar/features obey different invariants.
5. Genomic spatial folds: markets require purged temporal walk-forward evaluation.
6. Organism identity and species-specific heads: symbol/exchange/regime identities have different generalization risks.
7. Splice donor/acceptor junction ontology: a market anchor relation must be defined and validated separately.
8. Gene masks, exon bodies, TSS/PAS and strand-specific track metadata: none can be renamed into market concepts without a new label contract.
9. Poisson count likelihood and positive softplus outputs: signed returns and censored events require different distributions.

## 19. NEEDS_VERIFICATION

1. `NEEDS_VERIFICATION`: pretraining input shift/reverse-complement implementation is not published in this repository.
2. `NEEDS_VERIFICATION`: four-fold split construction and overlap-removal code is external to this repository; only split consumption is visible.
3. `NEEDS_VERIFICATION`: 64 all-fold teacher construction and checkpoint selection code is absent.
4. `NEEDS_VERIFICATION`: final distillation trainer, teacher sampling, mutation generator and schedules are absent.
5. `NEEDS_VERIFICATION`: 8-way sequence-parallel encoder/transformer/pair/decoder implementation is absent.
6. `NEEDS_VERIFICATION`: ISM enumeration comes from the separately packaged `alphagenome` dependency; only the research wrapper was audited locally.
7. `NEEDS_VERIFICATION`: `AlphaGenome.loss` does not expose an observed `is_training=True` path; from-scratch `RMSBatchNorm` update behavior cannot be inferred from the public trainer because that trainer is absent.
8. `NEEDS_VERIFICATION`: full 1 Mb checkpoint inference, official data evaluation and GPU numerical parity were deliberately not run in MG0-A.
9. `NEEDS_VERIFICATION`: `dna_model_test.py` collection in the current environment requires generated `calibration_scores_pb2`; no generated artifact was added to the repository.

## 20. Verification and gate

### 冻结所依据的可追溯证据

- `git fetch upstream main` + merge completed; official current source commit is a parent of local HEAD.
- `git diff upstream/main..HEAD -- src/alphagenome_research` produced no diff.
- Static inventory covered all required source paths, scoring modules, IO/eval paths and model tests.
- 非因果结论由冻结 commit 中的 SAME padding、全序列 normalization、无 hard mask softmax 和 pair row softmax 代码直接支撑，不依赖未持久化的历史 probe 数值。
- 用户要求延期前的 JAX 测试尝试已中断，部分输出不计入冻结证据；提醒后未再启动 GPU workload。
- 报告 JSON Schema、MG0 semantic freeze checks 与 validator tests 是本次唯一新增运行验收；它们不加载 AlphaGenome/JAX。
- `dna_model_test.py` 所需 build-generated `calibration_scores_pb2` 以及完整模型/checkpoint/data 验证均保留为 `NEEDS_VERIFICATION`。

### MG0-A gate decision

`PASS_WITH_NEEDS_VERIFICATION` for the audit deliverable:

- PASS: every required component has an explicit evidence row, market mapping, causal modification, status, risk and validation test.
- PASS: published behavior is separated from `DESIGN_INFERENCE`.
- PASS: non-isomorphic components are explicit; no DNA-market full isomorphism is claimed.
- PASS: no MarketGenome implementation or scaffold is added.
- CONDITIONAL: pretraining/folds/teachers/distillation/sequence parallelism cannot achieve code parity because official trainer source is absent. These remain `PUBLISHED_ONLY/NEEDS_VERIFICATION` and cannot be promoted to observed code behavior.
- DEFERRED BY USER: 所有训练、完整 local model/checkpoint/data execution、GPU/JAX 数值与性能工作。

## 21. Freeze decision and change control

- 冻结结论：`PARTIAL_PUBLISHED_CODE_PARITY`、`PASS_WITH_NEEDS_VERIFICATION`；不得提升为 full parity 或无条件 PASS。
- 冻结基线：official commit `232fc695d1eab27bac9e94bcd4b50499139ba4e1`，local source HEAD `ed738d92ab76df03b4d25ebb741e33048c9aae09`，36 个 component rows，9 项 `NEEDS_VERIFICATION`。
- `MG0_ALPHAGENOME_PARITY_AUDIT.json` 的 `freeze` 对象锁定除自身外的 canonical payload hash、Markdown、schema 与 validator hash；这些 hash 只证明当前文件彼此自洽。外部锚点是本地 annotated tag `mg0-a-alphagenome-parity-v1`，`mg0 --strict` 还会逐字节核对该 tag 中的 audit、Markdown、schema、validator、manifest 和 validator tests。
- annotated tag 当前未签名、未推送，也没有远端保护；强制移动或删除 tag 会绕过本地锚点。这是明确保留的操作风险，不得把本地 strict 描述为签名或远端不可变保证。
- 变更控制：修改任何冻结结论、组件状态、原则、非同构项、uncertainty 或 gate，必须取得用户显式批准并升级 schema/version；后续 GPU/训练结果写入独立 addendum 或新版本，不原地改写 v1。
- 恢复条件：GPU 预计最早 `2026-07-14` 空闲，但所有训练/GPU 工作仍须用户再次显式授权，不自动开始。

## 22. 下一阶段约束

MG0-B 及之后只能从本报告提取 contracts，不能复制 AlphaGenome 的 bidirectional/SAME/RMSBatchNorm/pair symmetry。任何 MarketGenome implementation 开始前，至少先冻结：as-of data contract、purged folds、prefix-invariance test harness、target/scaler registry、domain-valid perturbation invariants，以及 tiny-model compute gate。
