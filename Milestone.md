([Past chat][1])([Past chat][1])([Past chat][1])([Past chat][1])

你这个“milestone 混乱”的感觉是对的：**M2 实际上做成了“训练管道与报告契约（infra）闭环”，但没有把 AlphaTrade v0.2 作为主模型跑起来**。所以现在最清晰的做法是：

* **冻结 M2 = Infra Milestone（已验收）**：数据→index→dataloader→训练→固定 reports schema（哪怕用 MLP 占位，也把管道/契约跑通了）
* **新开 M3 = Model Milestone**：把 **AlphaTrade v0.2（JAX）真正接入训练**，并且**不改 reports 字段名**（继续用 M2 的 schema）

下面我把**重新规划的整体 milestone** + **M3 给 Claude Code 的可执行任务清单**一次性整理好，你直接复制给 Claude 就行。

---

## 重新规划整体 Milestones（把历史“定名”，避免以后重复）

### M0 / M1（已完成，不动）

* **M0：数据标准化最小闭环**（2 个品种，bars/index/SMOKE）
* **M1：Universe 扫描与候选集**（从 archive 统计→规则打分→candidates）

### M2（已完成，定性为 Infra）

* **M2：训练管道 + reports contract v1（字段名固定）**

   * 目标：任何模型都能接这套 dataloader 与 reports schema
   * 产物：`schemas/*.schema.json` + `validate_reports_schema.py` + `reports/*`（你已经收尾完成✅）
   * 结论：**M2 不再承担“主模型效果”目标**，只承担“管道与契约稳定”目标

### M3（✅ 已完成 - 2026-03-01）

* **M3：AlphaTrade v0.2（JAX）接入训练 + 稳定跑 + 报告不变**

   * 目标：替换掉 placeholder/MLP，让训练脚本跑的就是 AlphaTrade v0.2
   * 不新增/不改 reports 字段名（继续用 M2 schema），避免再混乱
   * **状态**: ✅ 全部完成
   * **验收报告**: `reports/M3_FINAL_ACCEPTANCE.md`
   * **核心成果**:
     - ✅ 成功集成 AlphaTrade v0.2 (JAX, 6.3M 参数)
     - ✅ 100% 复用 M2 基础设施
     - ✅ 训练稳定（50 steps, 无 NaN/Inf）
     - ✅ Schema 验证通过
     - ✅ Loss 下降 64-67%
   * **交付物**:
     - 代码: `train_m3_alphatrade.py`, `m3_alphatrade_forward_smoke.py`
     - 文档: 8 个文档（契约、指南、总结、验收）
     - 输出: `m3_train_metrics.json`, `m3_train_run.md`
   * **遗留问题**: 见 `reports/M3_TO_M4_IMPROVEMENTS.md`

### M4（下一步：评估与优化）

* **M4：评估指标扩展 + 模型优化 + 完整训练**

   * 目标：
     - 完整训练（500-1000 steps）
     - 评估指标扩展（calibration, IC, Sharpe）
     - 模型优化（超参数搜索）
     - 全品种训练（24 个品种）
   * **优先改进**（见 `reports/M3_TO_M4_IMPROVEMENTS.md`）:
     - 🔴 高优先级: 梯度范数指标拆分（pre-clip vs post-clip）
     - 🟡 中优先级: 训练日志增强、Checkpoint 保存、Schema 扩展
     - 🟢 低优先级: 学习率调度
   * **预计工作量**: 3.5 小时（改进） + 2-3 小时（完整训练）

### M5（后续：规模化）

* 更大 universe、更多历史、更长 lookback、更大 batch、多卡（pmap）/checkpoint/hparam sweep

---

## 给 Claude Code 的 M3 子任务清单（按“先定 contract/验收再写代码”）

> **M3 的核心原则**：
>
> 1. **复用 M2 的数据与 reports schema**（字段名不变）
> 2. 训练脚本真正调用 **AlphaTrade v0.2（JAX）**
> 3. 最低稳定性配置默认开启（grad clip / nan&inf 统计）

你把下面整段发给 Claude：

```md
# Milestone M3: Integrate AlphaTrade v0.2 (JAX) into training (replace placeholder/MLP)

## M3 目标
- 训练脚本使用现成 AlphaTrade v0.2（JAX）实现进行训练与验证
- 复用 M2 的 dataset/dataloader/index 产物与 reports schema（字段名固定，不新增顶层 key）
- 训练能稳定跑（无 NaN/Inf 扩散），并稳定产出：
  - reports/m3_train_metrics.json（必须通过 m2_train_metrics.schema.json 校验）
  - reports/m3_train_run.md（命令+配置摘要+数据量+loss/稳定性）

---

## T0. Contract freeze（先定“怎么验收”）
### 要做
1) 明确 M3 仍然使用 M2 的 reports schema：
   - src/alphatrade/schemas/m2_train_metrics.schema.json
   - 允许 model.* 内补充字段（schema 不限制），但顶层字段名必须固定（run/dataset/model/training/loss/stability/notes）
2) 约定 M3 的 metrics 字段填充规则（关键）：
   - model.type = "alphatrade_v0.2"
   - model.backend = "jax"
   - model.total_params = 从 params tree 统计（int）
   - stability.max_grad_norm / nan_inf_steps / oom_count 必须真实记录
   - training.compile_jit: true/false（是否 jit 训练步）
3) 写/更新 docs/m3_training_contract.md（或在 docs/m2_reports_contract.md 追加一段说明 M3 复用 schema）

### 验收
- 文档里写清：M3 输出文件名、用哪个 schema 校验、字段如何填

---

## T1. 模型入口确认 + Adapter（只要能 forward）
### 要做
1) 在仓库里确认 AlphaTrade v0.2 的 import 路径与初始化入口（JAX train_state/params）
2) 写一个最薄 adapter（建议位置）：
   - src/alphatrade/models/alphatrade_v0_2_adapter.py
   - 输入：X: float32 [B, L, 8]
   - 输出：pred（至少能拿到 returns quantiles 或你们内部 logits）
3) 写一个最小脚本验证 forward（不训练也行）：
   - src/alphatrade/scripts/m3_alphatrade_forward_smoke.py
   - 随机 batch 跑一次 forward，打印输出 shape

### 验收
- 一条命令能跑通 forward，不报错，输出 shape 合理

---

## T2. 训练脚本：把 AlphaTrade 接进 M2 dataloader
### 要做
1) 新训练脚本（不要改 M2 的基线脚本，避免回归）：
   - src/alphatrade/scripts/train_m3_alphatrade.py
2) 复用 M2 dataloader/index：
   - 仍然从 data/processed/m2/<SYMBOL>/bars.parquet + index_*.parquet 读取
   - 训练时按 index 切片，不提前 materialize 全窗口
   - 确保 feature dim=8，dtype=float32
3) 训练稳定性最低配置（默认开）：
   - optax.clip_by_global_norm(1.0)
   - 统计并记录：
     - stability.max_grad_norm（全程最大）
     - stability.nan_inf_steps（出现 NaN/Inf 的 step 次数）
     - stability.oom_count（OOM 次数，捕获异常计数）
4) loss：
   - 用 AlphaTrade v0.2 的 pinball loss + crossing penalty（按你们实现）
5) 输出报告（字段名必须符合 schema）：
   - reports/m3_train_metrics.json（校验 schema = m2_train_metrics.schema.json）
   - reports/m3_train_run.md（可复制命令 + config 摘要 + 数据量 + loss 表格 + stability）
6) 可选：复用 schema validator
   - 用 src/alphatrade/scripts/validate_reports_schema.py 跑一次校验并把结果写进 run.md

### 建议 CLI（可调整）
python src/alphatrade/scripts/train_m3_alphatrade.py \
  --config src/alphatrade/configs/dataset/m2.yaml \
  --max-steps 500 \
  --batch-size 128 \
  --num-workers 2 \
  --seed 42 \
  --jit 1 \
  --clip-norm 1.0

### 验收
- 一条命令稳定跑完（至少 200~1000 steps）
- 必须产出 metrics.json + run.md
- metrics.json 必须通过 schema 校验（字段名不缺、类型正确）

---

## T3. 数据/契约一致性检查（避免“特征名 vs 特征位次”踩坑）
### 要做
1) 明确训练使用的 8 维特征顺序来自 config（不要硬编码）
2) 在 train_m3_alphatrade.py 启动时打印：
   - features 列表
   - lookback/stride/horizons/quantiles
3) 抽 3 个 symbol 做 batch dump（只 dump shape/dtype/是否有 NaN，不落大文件）

### 验收
- 3 个 symbol 的 batch：shape=[B,L,8] dtype=float32，无 NaN
- run.md 里记录了 features 顺序（防止后续线上错位）

---

## T4. 最终 M3 验收报告
### 要做
- 写 reports/M3_FINAL_ACCEPTANCE.md
  - 命令
  - 模型入口确认（type/backend/params）
  - schema validation 结果
  - loss & stability 指标摘要
  - 已知问题（如果有：编译慢、显存占用、jit 约束等）

### 验收
- M3_FINAL_ACCEPTANCE.md 完整可读
- M3 进入下一阶段（M4 eval）无阻塞项
```

---

## 你提的 M2-T3（“AlphaTrade 真接入”）应该怎么处理？

我建议**不要把它硬塞回 M2**（否则历史又会乱一次），而是：

* **把你写的 “M2-T3 spec” 原封不动迁移到 M3 的 T2**
* M2 保持“infra 验收通过”的事实，不改历史
* M3 专门为“主模型训练”负责

这样以后你看 milestone 就一眼明白：
**M2=管道；M3=主模型**，不会再出现“我以为 M2 已经是 AlphaTrade，结果跑的是 MLP”的错觉。

---

如果你愿意，我也可以顺手再给一个 **M4（评估/可比性）** 的同款任务清单模板（同样先定 reports contract），但你先把上面这段发给 Claude 跑 M3 就行。

[1]: https://chatgpt.com/c/699ee8f3-15c8-839c-b0fc-f771fb2730f7 "AlphaGenome期货模型设计"
