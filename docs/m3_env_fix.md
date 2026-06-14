# M3 环境修复指南

**问题**: JAX 0.9.0.1 与 NumPy 1.26.4 不兼容

**错误信息**:
```
TypeError: asarray() got an unexpected keyword argument 'copy'
```

**原因**: JAX 0.9.0.1 调用 `np.asarray(..., copy=...)` 时使用了 `copy` 参数，但这个参数只在 NumPy 2.0+ 中存在。

---

## 解决方案

### 方案 A: 升级 NumPy (推荐)

```bash
conda activate alphatrade
pip install --upgrade "numpy>=2.0.0"
```

**优点**:
- 简单直接
- NumPy 2.0+ 是未来趋势

**缺点**:
- 可能影响其他依赖 NumPy 1.x 的包

### 方案 B: 降级 JAX

```bash
conda activate alphatrade
pip install "jax==0.4.23" "jaxlib==0.4.23"
```

**优点**:
- 保持 NumPy 1.26.4 不变
- 兼容性更好

**缺点**:
- 使用旧版 JAX，可能缺少新特性

### 方案 C: 创建新环境 (最稳妥)

```bash
# 创建新环境
conda create -n alphatrade_m3 python=3.11 -y
conda activate alphatrade_m3

# 安装依赖
pip install "jax[cuda13]>=0.9.0" "numpy>=2.0.0"
pip install dm-haiku optax jaxtyping
pip install pandas pyarrow pyyaml
```

---

## 验证修复

修复后运行以下命令验证：

```bash
conda activate alphatrade

# 测试 JAX + NumPy
python -c "import jax; import numpy as np; print(f'JAX: {jax.__version__}'); print(f'NumPy: {np.__version__}'); rng = jax.random.PRNGKey(42); print('✓ JAX random works')"

# 测试 AlphaTrade imports
python -c "import sys; sys.path.insert(0, 'src'); from alphatrade.core import schemas, model; print('✓ AlphaTrade imports work')"

# 运行 forward smoke test
python src/alphatrade/scripts/m3_alphatrade_forward_smoke.py
```

**预期输出**:
```
============================================================
M3 AlphaTrade v0.2 Forward Smoke Test
============================================================
...
✅ SMOKE TEST PASSED
```

---

## 当前环境状态

```
JAX: 0.9.0.1
NumPy: 1.26.4
Python: 3.11
```

**状态**: ❌ 不兼容

**建议**: 使用方案 A (升级 NumPy 到 2.0+)

---

## 升级后的完整测试流程

```bash
# 1. 修复环境
conda activate alphatrade
pip install --upgrade "numpy>=2.0.0"

# 2. 验证环境
python -c "import jax; rng = jax.random.PRNGKey(42); print('✓ JAX works')"

# 3. 运行 forward smoke test
python src/alphatrade/scripts/m3_alphatrade_forward_smoke.py

# 4. 运行 M3 训练 (smoke test)
python src/alphatrade/scripts/train_m3_alphatrade.py \
  --config configs/dataset/m2.yaml \
  --max-steps 50 \
  --batch-size 32 \
  --smoke \
  --jit 0

# 5. 验证 schema
python src/alphatrade/scripts/validate_reports_schema.py \
  --reports-dir "$ALPHATRADE_RUNS_ROOT/reports" \
  --schemas-dir src/alphatrade/schemas
```

---

## 故障排查

### 问题 1: 升级 NumPy 后其他包报错

**解决**: 重新安装受影响的包
```bash
pip install --upgrade pandas pyarrow
```

### 问题 2: CUDA 相关错误

**解决**: 确认 JAX CUDA 版本匹配
```bash
pip install --upgrade "jax[cuda13]"
```

### 问题 3: Haiku 导入错误

**解决**: 重新安装 dm-haiku
```bash
pip install --upgrade dm-haiku
```

---

## 参考

- JAX 版本兼容性: https://github.com/google/jax/releases
- NumPy 2.0 迁移指南: https://numpy.org/devdocs/numpy_2_0_migration_guide.html
