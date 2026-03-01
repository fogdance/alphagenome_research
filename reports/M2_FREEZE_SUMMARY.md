# M2 收尾总结：Contract Freeze + Validation

生成时间: 2026-03-01

---

## 任务目标

M2 已验收通过，不重跑训练/数据流程；只做"契约固化 + 自动校验"，保证以后任何人复跑 M2 都不会因为 JSON 字段/路径漂移而崩。

---

## 完成度

| 任务 | 状态 | 完成度 |
|------|------|--------|
| T-M2-F1: 固化 reports JSON schema | ✅ | 100% |
| T-M2-F2: 本地校验脚本 | ✅ | 100% |
| T-M2-F3: 契约文档 | ✅ | 100% |
| T-M2-F4: CI Hook (可选) | ⏳ | 待定 |

**总进度**: 3/3 必须项 (100%)

---

## 产物清单

### ✅ Schema 文件 (3 个)

1. **m2_train_metrics.schema.json**
   - 路径: `src/alphatrade/schemas/m2_train_metrics.schema.json`
   - 用途: 训练 metrics 输出验证
   - 版本: m2_train_metrics_v1

2. **m2_universe_sweep.schema.json**
   - 路径: `src/alphatrade/schemas/m2_universe_sweep.schema.json`
   - 用途: Universe 批量评估验证
   - 版本: m2_universe_sweep_v1

3. **m2_t1_dataloader_check.schema.json**
   - 路径: `src/alphatrade/schemas/m2_t1_dataloader_check.schema.json`
   - 用途: Dataloader 验证输出
   - 版本: m2_t1_dataloader_check_v1

### ✅ 校验脚本

**文件**: `src/alphatrade/scripts/validate_reports_schema.py`

**功能**:
- 读取 3 个 JSON 报告
- 使用 jsonschema 验证
- 产出 JSON + MD 报告
- Exit code: 0 (全部通过) / 2 (任意失败)

### ✅ 校验报告 (2 个)

1. **m2_schema_validation.json**
   - 路径: `reports/m2_schema_validation.json`
   - 格式: 机器可读
   - 内容: 每个文件的验证结果 + 错误详情

2. **m2_schema_validation.md**
   - 路径: `reports/m2_schema_validation.md`
   - 格式: 人类可读
   - 内容: 验证总览 + 结果表格

### ✅ 契约文档

**文件**: `docs/m2_reports_contract.md`

**内容**:
- Schema 文件路径
- 校验命令（可复制）
- 使用场景
- 版本管理规则
- 故障排查
- 最佳实践

---

## 验收结果

### 一条命令验证

```bash
python src/alphatrade/scripts/validate_reports_schema.py \
  --reports-dir reports \
  --schemas-dir src/alphatrade/schemas
```

### 验证结果

✅ **所有验证通过 (3/3)**

| Report | Status | Schema |
|--------|--------|--------|
| m2_train_metrics | ✅ pass | m2_train_metrics.schema.json |
| m2_universe_sweep | ✅ pass | m2_universe_sweep.schema.json |
| m2_t1_dataloader_check | ✅ pass | m2_t1_dataloader_check.schema.json |

**Exit code**: 0

---

## Schema 覆盖范围

### m2_train_metrics.schema.json

**必须字段** (6 个顶层):
- `run`: 运行元信息
- `dataset`: 数据集信息（含 feature_cols, quantiles）
- `model`: 模型信息
- `training`: 训练配置
- `loss`: Loss 统计（含 by_horizon）
- `stability`: 稳定性指标

**关键约束**:
- `by_horizon` 必须包含 h1, h5, h20, h60
- 每个 horizon 必须有 train 和 val
- `additionalProperties: false` (顶层)

### m2_universe_sweep.schema.json

**必须字段** (5 个顶层):
- `timestamp`: 时间戳
- `universe`: Universe 文件路径
- `total_symbols`: 总品种数
- `statistics`: 统计信息（含 trainable_rate）
- `results`: 每个品种的结果数组

**关键约束**:
- `status` 枚举: success, insufficient_samples, missing_files, nan_detected, error, unknown
- `trainable_rate` 范围: 0-1
- `additionalProperties: false` (顶层)

### m2_t1_dataloader_check.schema.json

**必须字段** (6 个顶层):
- `timestamp`: 时间戳
- `config`: 配置信息
- `dataset`: 数据集统计
- `validation`: 验证结果（含 success_rate）
- `random_samples`: 随机样本数组
- `symbol_samples`: 按品种样本数组

**关键约束**:
- `x_shape` 必须是 2 元素数组
- `y_shape` 必须是 1 元素数组
- `success_rate` 范围: 0-1
- `additionalProperties: false` (顶层)

---

## 使用场景

### 1. 修改训练脚本后验证

```bash
# 修改 train_m2_final.py
vim src/alphatrade/scripts/train_m2_final.py

# 重新运行训练
python src/alphatrade/scripts/train_m2_final.py --smoke --max-steps 20

# 验证输出
python src/alphatrade/scripts/validate_reports_schema.py
```

### 2. 添加新字段

**步骤**:
1. 修改 schema 文件（添加新字段）
2. 修改生成脚本（输出新字段）
3. 运行验证
4. 提交修改

### 3. 复现 M2 训练

```bash
# 1. 运行训练
python src/alphatrade/scripts/train_m2_final.py --smoke

# 2. 验证输出
python src/alphatrade/scripts/validate_reports_schema.py

# 3. 检查结果
cat reports/m2_schema_validation.md
```

---

## 依赖

### Python 包

```bash
pip install jsonschema
```

或在 `requirements.txt` 中添加:
```
jsonschema>=4.0.0
```

---

## 契约保证

### ✅ 字段名固定

所有核心字段名已固化在 schema 中，不可随意修改。

**示例**:
- `run.run_id` ✅ 固定
- `loss.by_horizon.h1.train` ✅ 固定
- `statistics.trainable_rate` ✅ 固定

### ✅ 类型固定

所有字段类型已定义，不可随意更改。

**示例**:
- `dataset.symbols`: integer ✅
- `loss.train_last`: number ✅
- `stability.nan_steps`: integer ✅

### ✅ 必须字段

核心字段标记为 `required`，缺失会导致验证失败。

**示例**:
- `run`, `dataset`, `model`, `training`, `loss`, `stability` 都是必须的
- `loss.by_horizon` 必须包含 h1, h5, h20, h60

### ✅ 自动验证

一条命令即可验证所有报告，exit code 明确。

```bash
python src/alphatrade/scripts/validate_reports_schema.py
# Exit 0: 全部通过
# Exit 2: 任意失败
```

---

## 版本管理

### 当前版本

- **m2_train_metrics**: v1
- **m2_universe_sweep**: v1
- **m2_t1_dataloader_check**: v1

### 升级规则

**Breaking Changes** (需要升级版本):
- 删除必须字段
- 修改字段类型
- 重命名字段

**Non-Breaking Changes** (不需要升级):
- 添加可选字段
- 修改字段描述
- 放宽约束

---

## 故障排查

### 验证失败：字段缺失

**错误**:
```
❌ fail
Error: 'field_name' is a required property
```

**解决**:
1. 检查生成脚本是否输出了该字段
2. 检查字段名拼写
3. 检查 schema 的 `required` 列表

### 验证失败：类型不匹配

**错误**:
```
❌ fail
Error: 123 is not of type 'string'
```

**解决**:
1. 检查生成脚本的数据类型
2. 检查 schema 的类型定义
3. 必要时转换类型

### 验证失败：文件不存在

**错误**:
```
⚠️ missing_report
Error: Report file not found
```

**解决**:
1. 运行对应的生成脚本
2. 检查文件路径
3. 检查文件权限

---

## 最佳实践

### 1. 修改前先验证

确保当前状态是正确的，再进行修改。

### 2. 保持 schema 简洁

只定义核心必须字段，避免过度约束。

### 3. 文档同步更新

修改 schema 后更新 `m2_reports_contract.md`。

### 4. 向后兼容

尽量添加字段而不是删除，使用可选字段。

---

## 下一步

### 可选：T-M2-F4 (CI Hook)

如果 repo 有 CI，可以添加验证步骤：

```yaml
# .github/workflows/validate.yml
- name: Validate M2 Reports Schema
  run: |
    pip install jsonschema
    python src/alphatrade/scripts/validate_reports_schema.py
```

### M3 准备

M2 契约已固化，可以安全进入 M3：
- 完整训练 (5K-10K steps)
- 真实模型接入
- 评估指标扩展

---

## 总结

✅ **M2 收尾任务完成**

**核心成果**:
- 3 个 JSON Schema 文件 ✅
- 1 个自动校验脚本 ✅
- 2 个校验报告 ✅
- 1 个契约文档 ✅

**验收标准达成**:
- ✅ 一条命令跑完校验
- ✅ Schema 完整覆盖 3 份 JSON
- ✅ 文档清晰说明契约入口
- ✅ 所有验证通过 (3/3)

**契约保证**:
- ✅ 字段名固定
- ✅ 类型固定
- ✅ 必须字段明确
- ✅ 自动验证可用

**M2 状态**: ✅ **已固化（Frozen）**

---

**最后验证**: 2026-03-01 18:26:36 - 所有验证通过 (3/3)
