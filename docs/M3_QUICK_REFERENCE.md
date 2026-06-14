# M3 快速参考指南

**版本**: M3 v1
**日期**: 2026-03-01

---

## 🚀 快速开始

### 1. 修复环境 (必须先执行)

```bash
conda activate alphatrade
pip install --upgrade "numpy>=2.0.0"
```

### 2. 验证环境

```bash
python -c "import jax; rng = jax.random.PRNGKey(42); print('✓ JAX works')"
```

### 3. 运行 Forward Smoke Test

```bash
python src/alphatrade/scripts/m3_alphatrade_forward_smoke.py
```

### 4. 运行训练 (Smoke Test)

```bash
python src/alphatrade/scripts/train_m3_alphatrade.py \
  --config configs/dataset/m2.yaml \
  --max-steps 50 \
  --batch-size 32 \
  --smoke \
  --jit 0
```

### 5. 验证 Schema

```bash
python src/alphatrade/scripts/validate_reports_schema.py \
  --reports-dir "$ALPHATRADE_RUNS_ROOT/reports" \
  --schemas-dir src/alphatrade/schemas
```

---

## 📁 关键文件

### 配置

```
configs/dataset/m2.yaml              # M2 数据集配置 (M3 复用)
configs/universe/m1_selected.yaml    # 24 个品种
```

### 脚本

```
src/alphatrade/scripts/
├── m3_alphatrade_forward_smoke.py   # Forward 测试
├── train_m3_alphatrade.py           # M3 训练脚本
└── validate_reports_schema.py       # Schema 验证器
```

### 文档

```
docs/
├── m3_training_contract.md          # 训练契约
└── m3_env_fix.md                    # 环境修复指南
```

### 数据

```
data/processed/m1_f8/{symbol}/
├── bars.parquet                     # 8D 特征
├── index_train.parquet              # 训练索引
├── index_val.parquet                # 验证索引
└── index_test.parquet               # 测试索引
```

### Schema

```
src/alphatrade/schemas/
└── m2_train_metrics.schema.json     # M3 复用 M2 schema
```

### 输出 (运行后生成)

```
$ALPHATRADE_RUNS_ROOT/reports/
├── m3_train_metrics.json            # 训练指标 (JSON)
├── m3_train_run.md                  # 训练报告 (Markdown)
└── M3_FINAL_ACCEPTANCE.md           # 最终验收 (待生成)
```

---

## 🎯 训练命令

### Smoke Test (快速验证)

```bash
# 3 个品种, 50 steps, 无 JIT (调试)
python src/alphatrade/scripts/train_m3_alphatrade.py \
  --config configs/dataset/m2.yaml \
  --max-steps 50 \
  --batch-size 32 \
  --smoke \
  --jit 0
```

### 中等规模 (500 steps)

```bash
# 3 个品种, 500 steps, 开启 JIT
python src/alphatrade/scripts/train_m3_alphatrade.py \
  --config configs/dataset/m2.yaml \
  --max-steps 500 \
  --batch-size 128 \
  --smoke \
  --jit 1 \
  --clip-norm 1.0
```

### 完整训练 (1000 steps)

```bash
# 24 个品种, 1000 steps, 开启 JIT
python src/alphatrade/scripts/train_m3_alphatrade.py \
  --config configs/dataset/m2.yaml \
  --max-steps 1000 \
  --batch-size 128 \
  --jit 1 \
  --clip-norm 1.0 \
  --seed 42
```

---

## 🔧 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | configs/dataset/m2.yaml | 数据集配置 |
| `--max-steps` | 1000 | 训练步数 |
| `--batch-size` | 256 | 批大小 |
| `--num-workers` | 2 | 数据加载线程数 |
| `--seed` | 42 | 随机种子 |
| `--jit` | 1 | JIT 编译 (0=关闭, 1=开启) |
| `--clip-norm` | 1.0 | 梯度裁剪阈值 |
| `--smoke` | False | 使用 3 个品种快速测试 |

---

## 📊 数据统计

### 品种数量

- **Smoke test**: 3 个品种 (DCE.JM, SHFE.AG, CZCE.MA)
- **完整训练**: 24 个品种 (m1_selected.yaml)

### 样本数量

| 数据集 | Smoke (3 品种) | 完整 (24 品种) |
|--------|----------------|----------------|
| Train | ~57K | ~388K |
| Val | ~11K | ~74K |
| Test | - | ~148K |

### 特征

- **维度**: 8
- **顺序**: ret_1m, hl_range, co_change, vol_log1p, pos_log1p, minute_sin, minute_cos, is_session_open
- **类型**: float32
- **Lookback**: 60

### Horizons & Quantiles

- **Horizons**: [1, 5, 20, 60]
- **Quantiles**: [0.1, 0.3, 0.5, 0.7, 0.9]

---

## 🔍 验证检查

### 1. 环境检查

```bash
# JAX 版本
python -c "import jax; print(f'JAX: {jax.__version__}')"

# NumPy 版本 (需要 >= 2.0)
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"

# AlphaTrade 导入
python -c "import sys; sys.path.insert(0, 'src'); from alphatrade.core import schemas, model; print('✓ AlphaTrade OK')"
```

### 2. 数据检查

```bash
# 检查数据文件
ls -lh data/processed/m1_f8/DCE.JM/

# 检查特征维度
python -c "import pandas as pd; df = pd.read_parquet('data/processed/m1_f8/DCE.JM/bars.parquet'); print(f'Features: {df.shape[1]}')"

# 检查样本数
python -c "import pandas as pd; df = pd.read_parquet('data/processed/m1_f8/DCE.JM/index_train.parquet'); print(f'Train samples: {len(df):,}')"
```

### 3. Schema 验证

```bash
# 验证 M3 输出
python src/alphatrade/scripts/validate_reports_schema.py \
  --reports-dir "$ALPHATRADE_RUNS_ROOT/reports" \
  --schemas-dir src/alphatrade/schemas

# 预期输出:
# Validating m3_train_metrics... ✅ pass
```

### 4. 输出检查

```bash
# 检查 metrics JSON
cat "$ALPHATRADE_RUNS_ROOT/reports/m3_train_metrics.json" | python -m json.tool | head -20

# 检查关键字段
python -c "import json, os; m = json.load(open(os.path.join(os.environ['ALPHATRADE_RUNS_ROOT'], 'reports/m3_train_metrics.json'))); print(f\"Model: {m['model']['type']}\"); print(f\"Backend: {m['model']['backend']}\"); print(f\"Params: {m['model']['total_params']:,}\")"

# 检查 markdown 报告
head -30 "$ALPHATRADE_RUNS_ROOT/reports/m3_train_run.md"
```

---

## ⚠️ 故障排查

### 问题 1: JAX/NumPy 版本不兼容

**错误**: `TypeError: asarray() got an unexpected keyword argument 'copy'`

**解决**:
```bash
conda activate alphatrade
pip install --upgrade "numpy>=2.0.0"
```

### 问题 2: OOM (显存不足)

**错误**: `OutOfMemoryError`

**解决**:
```bash
# 减小 batch size
--batch-size 64  # 或 32

# 或减小模型规模 (修改 train_m3_alphatrade.py)
d_model=256  # 默认 512
```

### 问题 3: NaN/Inf 训练失败

**错误**: `NaN detected!`

**解决**:
```bash
# 降低学习率 (修改 configs/dataset/m2.yaml)
learning_rate: 0.00001  # 默认 0.0001

# 或增加梯度裁剪
--clip-norm 0.5  # 默认 1.0
```

### 问题 4: JIT 编译慢

**现象**: 第一个 step 很慢 (几分钟)

**说明**: 正常现象，JAX JIT 编译需要时间，后续 step 会快

**解决**: 耐心等待，或关闭 JIT 调试 (`--jit 0`)

### 问题 5: 数据文件缺失

**错误**: `bars.parquet not found`

**解决**:
```bash
# 检查数据路径
ls data/processed/m1_f8/

# 如果缺失，重新生成 (参考 M1 文档)
python src/alphatrade/scripts/build_m1_canonical_bars_v2.py ...
```

---

## 📈 性能优化

### 1. 开启 JIT 编译

```bash
--jit 1  # 提速 5-10x
```

### 2. 调整 Batch Size

```bash
# 根据显存调整
--batch-size 256  # 大显存 (24GB+)
--batch-size 128  # 中等显存 (12GB)
--batch-size 64   # 小显存 (8GB)
```

### 3. 使用 GPU

```bash
# 自动检测，无需配置
# 检查是否使用 GPU:
python -c "import jax; print(jax.devices())"
```

---

## 📝 输出示例

### m3_train_metrics.json (关键字段)

```json
{
  "model": {
    "type": "alphatrade_v0.2",
    "backend": "jax",
    "total_params": 1234567
  },
  "training": {
    "compile_jit": true
  },
  "loss": {
    "train_last": 0.003094,
    "val_best": 0.002275,
    "by_horizon": {
      "h1": {"train": 0.002707, "val": 0.002382},
      "h5": {"train": 0.003470, "val": 0.002488},
      "h20": {"train": 0.003031, "val": 0.002339},
      "h60": {"train": 0.002962, "val": 0.001889}
    }
  },
  "stability": {
    "nan_steps": 0,
    "inf_steps": 0,
    "max_grad_norm": 1.428,
    "oom_count": 0
  }
}
```

### m3_train_run.md (示例)

```markdown
# M3 Training Run - AlphaTrade v0.2 (JAX)

## 配置
- Backend: JAX
- JIT: enabled
- Symbols: 3
- Train samples: 57,418
- Val samples: 11,313

## 模型
- Type: AlphaTrade v0.2
- Total params: 1,234,567
- d_model: 512

## Loss
- Train best: 0.003094
- Val best: 0.002275 @ step 200

### By-Horizon Loss
| Horizon | Train | Val |
|---------|-------|-----|
| h1 | 0.002707 | 0.002382 |
| h5 | 0.003470 | 0.002488 |
| h20 | 0.003031 | 0.002339 |
| h60 | 0.002962 | 0.001889 |

## Stability
- NaN steps: 0
- Inf steps: 0
- Max grad norm: 1.428
```

---

## 🎓 关键经验

### 1. 环境管理

- ✅ 先修复 JAX/NumPy 兼容性
- ✅ 使用 conda 环境隔离
- ✅ 验证环境后再训练

### 2. 调试策略

- ✅ 先用 smoke test (3 品种, 50 steps)
- ✅ 关闭 JIT 调试 (`--jit 0`)
- ✅ 检查 NaN/Inf 指标

### 3. 性能优化

- ✅ 开启 JIT 编译 (提速 5-10x)
- ✅ 调整 batch size (根据显存)
- ✅ 使用 GPU (自动检测)

### 4. Schema 验证

- ✅ 训练后立即验证 schema
- ✅ 检查关键字段 (model.type, model.backend)
- ✅ 确认 by_horizon loss 存在

---

## 🔗 相关文档

- **M3 Training Contract**: `docs/m3_training_contract.md`
- **M3 Environment Fix**: `docs/m3_env_fix.md`
- **M3 T1+T2 Summary**: `$ALPHATRADE_RUNS_ROOT/reports/M3_T1_T2_SUMMARY.md`
- **M2 Final Acceptance**: `$ALPHATRADE_RUNS_ROOT/reports/M2_FINAL_ACCEPTANCE.md`
- **M1 Final Summary**: `$ALPHATRADE_RUNS_ROOT/reports/M1_FINAL_SUMMARY.md`

---

## ✅ 验收清单

- [ ] 环境修复完成 (NumPy >= 2.0)
- [ ] Forward smoke test 通过
- [ ] 训练 smoke test 通过 (50 steps)
- [ ] Schema 验证通过
- [ ] 完整训练完成 (500-1000 steps)
- [ ] 无 NaN/Inf 问题
- [ ] 输出文件齐全 (JSON + MD)
- [ ] 最终验收报告完成

---

**快速参考指南 v1.0** | 2026-03-01
