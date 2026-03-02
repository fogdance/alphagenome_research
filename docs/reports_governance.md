# Reports Governance

生成时间: 2026-03-03

---

## Source of Truth

**优先级**: CI/validator > schemas > 本文档

当本文档与 schema 或 validator 行为冲突时，以后者为准。

---

## Profile 机制

Validator 通过 `contracts_manifest.yaml` 中的 profile 隔离不同里程碑的验证范围。

### 当前 Profiles

| Profile | 用途 | Required | CI 门禁 |
|---------|------|----------|---------|
| `m4` | 主线门禁，3-seed train+eval | 全部 required | 是 |
| `m5` | Ablation study，3-seed baseline + variants | 全部 required | 是（M5 阶段） |
| `legacy_m2_m3` | 历史留档 | 全部 optional | 否 |

### Manifest 路径

```
src/alphatrade/schemas/contracts_manifest.yaml
```

### 使用方式

```bash
# 主线门禁（CI 使用）
python src/alphatrade/scripts/validate_reports_schema.py --profile m4 --strict

# 历史检查（不阻塞）
python src/alphatrade/scripts/validate_reports_schema.py --profile legacy_m2_m3

# 自定义 manifest
python src/alphatrade/scripts/validate_reports_schema.py --manifest path/to/manifest.yaml --profile my_profile
```

---

## 命名约定

### Report 文件

```
m{milestone}_{train|eval}_metrics_seed{seed}.json
```

示例:
- `m4_train_metrics_seed42.json`
- `m4_eval_metrics_seed43.json`
- `m4_eval_metrics_seed44.json`

### Schema 文件

```
src/alphatrade/schemas/m{milestone}_{type}.schema.json
```

示例:
- `m2_train_metrics.schema.json` — 训练 metrics（M2/M3/M4 共用）
- `m4_eval_metrics.schema.json` — M4 评估 metrics

---

## 变更规则

### 安全变更（不需要升版本）

- 添加 optional 字段
- 修改字段 description
- 在 manifest 中添加新 profile
- 在现有 profile 中添加 `required: false` item

### 破坏性变更（需要新版本）

- 修改 required 字段列表
- 修改字段类型
- 删除或重命名字段

**流程**: 创建新 schema 文件（如 `m2_train_metrics_v2.schema.json`），更新 `$id`，更新 manifest 引用。

---

## CI 门禁

在 `.github/workflows/presubmit_checks.yml` 中:

```yaml
- name: Validate reports contracts
  run: python src/alphatrade/scripts/validate_reports_schema.py --profile m4 --strict
```

**行为**:
- `--strict` + 所有 required pass → exit 0
- `--strict` + 任一 required fail/missing → exit 1
- 非 strict 模式下 optional 缺失不影响退出码

---

## Schema 版本

| Schema | $id | 版本 |
|--------|-----|------|
| m2_train_metrics | `m2_train_metrics_v1` | v1 |
| m2_universe_sweep | `m2_universe_sweep_v1` | v1 |
| m2_t1_dataloader_check | `m2_t1_dataloader_check_v1` | v1 |
| m4_eval_metrics | `m4_eval_metrics_v1` | v1 |
| m5_ablation_table | `m5_ablation_table_v1` | v1 |

---

## Validator 输出

每次运行生成两个文件:

- `reports/{profile}_schema_validation.json` — 机器可读
- `reports/{profile}_schema_validation.md` — 人类可读

---

## 参考

- [JSON Schema Draft-07](https://json-schema.org/draft-07/json-schema-release-notes.html)
- `docs/m2_reports_contract.md` — M2 历史契约文档
