# M2 Reports Contract (Frozen)

生成时间: 2026-03-01

---

## 目的

M2 任务已验收通过，为防止后续修改导致 JSON 字段/路径漂移，本文档固化 M2 reports 的契约（contract）。

**核心原则**:
- M2 的 reports JSON 以 schema 为准
- 字段名不可随意改动
- 新增字段需同步更新 schema
- 任何修改必须通过 schema 验证

---

## Schema 文件

### 1. m2_train_metrics.schema.json

**路径**: `src/alphatrade/schemas/m2_train_metrics.schema.json`

**用途**: 训练 metrics 输出（m2_train_metrics_v1）

**对应报告**: `reports/m2_train_metrics.json`

**核心字段**:
- `run`: 运行元信息（run_id, git_sha, created_at, device, seed）
- `dataset`: 数据集信息（symbols, samples, feature_cols, quantiles）
- `model`: 模型信息（type, params）
- `training`: 训练配置（max_steps, batch_size, lr, optimizer）
- `loss`: Loss 统计（train/val, by_horizon）
- `stability`: 稳定性指标（nan_steps, inf_steps, max_grad_norm）

### 2. m2_universe_sweep.schema.json

**路径**: `src/alphatrade/schemas/m2_universe_sweep.schema.json`

**用途**: Universe 批量评估输出

**对应报告**: `reports/m2_universe_sweep.json`

**核心字段**:
- `timestamp`: 时间戳
- `universe`: Universe 文件路径
- `total_symbols`: 总品种数
- `statistics`: 统计信息（success, trainable, trainable_rate）
- `results`: 每个品种的详细结果

### 3. m2_t1_dataloader_check.schema.json

**路径**: `src/alphatrade/schemas/m2_t1_dataloader_check.schema.json`

**用途**: Dataloader 验证输出

**对应报告**: `reports/m2_t1_dataloader_check.json`

**核心字段**:
- `timestamp`: 时间戳
- `config`: 配置信息（feature_dim, feature_cols, lookback, horizons）
- `dataset`: 数据集统计（total_samples, symbols）
- `validation`: 验证结果（total_checks, valid_checks, success_rate）
- `random_samples`: 随机样本验证
- `symbol_samples`: 按品种样本验证

---

## 校验命令

### 一条命令验证所有 reports

```bash
python src/alphatrade/scripts/validate_reports_schema.py \
  --reports-dir reports \
  --schemas-dir src/alphatrade/schemas
```

### 预期输出

**成功**:
```
✅ All validations passed
Exit code: 0
```

**失败**:
```
❌ N validation(s) failed
Exit code: 2
```

### 生成的报告

- `reports/m2_schema_validation.json` - 机器可读
- `reports/m2_schema_validation.md` - 人类可读

---

## 使用场景

### 1. 修改训练脚本后验证

```bash
# 修改 train_m2_final.py 后
python src/alphatrade/scripts/train_m2_final.py --smoke --max-steps 20

# 验证输出是否符合 schema
python src/alphatrade/scripts/validate_reports_schema.py
```

### 2. 添加新字段

**步骤**:
1. 修改对应的 schema 文件（添加新字段）
2. 修改生成脚本（输出新字段）
3. 运行验证确认通过
4. 提交 schema + 脚本修改

**示例**:
```json
// 在 m2_train_metrics.schema.json 中添加新字段
"loss": {
  "properties": {
    ...
    "new_metric": {"type": "number"}  // 新增
  }
}
```

### 3. CI/CD 集成

**在 CI 中添加验证步骤**:
```yaml
# .github/workflows/validate.yml
- name: Validate M2 Reports Schema
  run: |
    python src/alphatrade/scripts/validate_reports_schema.py
```

---

## Schema 版本管理

### 当前版本

- **m2_train_metrics**: v1 (m2_train_metrics_v1)
- **m2_universe_sweep**: v1 (m2_universe_sweep_v1)
- **m2_t1_dataloader_check**: v1 (m2_t1_dataloader_check_v1)

### 版本升级规则

**何时需要升级版本**:
- 删除必须字段（breaking change）
- 修改字段类型（breaking change）
- 重命名字段（breaking change）

**如何升级版本**:
1. 创建新的 schema 文件（如 `m2_train_metrics_v2.schema.json`）
2. 更新 `$id` 字段（如 `m2_train_metrics_v2`）
3. 更新生成脚本以支持新版本
4. 更新验证脚本以支持新版本
5. 更新本文档

**不需要升级版本**:
- 添加可选字段（non-breaking change）
- 修改字段描述（non-breaking change）

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

## 故障排查

### 验证失败：字段缺失

**错误**:
```
❌ fail
Error: 'field_name' is a required property
```

**解决**:
1. 检查生成脚本是否输出了该字段
2. 检查字段名是否拼写正确
3. 检查 schema 中 `required` 列表

### 验证失败：类型不匹配

**错误**:
```
❌ fail
Error: 123 is not of type 'string'
```

**解决**:
1. 检查生成脚本输出的数据类型
2. 检查 schema 中定义的类型
3. 必要时转换类型（如 `str(value)` 或 `int(value)`）

### 验证失败：文件不存在

**错误**:
```
⚠️ missing_report
Error: Report file not found: reports/xxx.json
```

**解决**:
1. 确认报告文件已生成
2. 检查文件路径是否正确
3. 运行对应的生成脚本

---

## 最佳实践

### 1. 修改前先验证

```bash
# 修改前验证当前状态
python src/alphatrade/scripts/validate_reports_schema.py

# 修改代码
# ...

# 修改后重新验证
python src/alphatrade/scripts/validate_reports_schema.py
```

### 2. 保持 schema 简洁

- 只定义核心必须字段
- 可选字段不要过多
- 避免过度嵌套

### 3. 文档同步更新

- 修改 schema 后更新本文档
- 在 commit message 中说明变更原因
- 更新相关的 SUMMARY 文档

### 4. 向后兼容

- 尽量添加字段而不是删除
- 使用可选字段而不是必须字段
- 保留旧字段的同时添加新字段

---

## 参考

### JSON Schema 规范

- [JSON Schema Draft-07](https://json-schema.org/draft-07/json-schema-release-notes.html)
- [Understanding JSON Schema](https://json-schema.org/understanding-json-schema/)

### 相关文档

- `docs/m2_training_data_spec.md` - M2 训练数据规范
- `reports/M2_FINAL_ACCEPTANCE.md` - M2 最终验收报告
- `reports/M2_T*_SUMMARY.md` - M2 各子任务总结

---

## 变更历史

| 日期 | 版本 | 变更 | 作者 |
|------|------|------|------|
| 2026-03-01 | v1 | 初始版本，固化 M2 reports schema | - |

---

**契约状态**: ✅ **已固化（Frozen）**

**最后验证**: 2026-03-01 - 所有验证通过 (3/3)
