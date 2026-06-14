# GPU 训练崩溃问题排查报告

**日期**: 2026-03-02 (排查) / 2026-03-03 (解决)
**环境**: RTX 4060 Ti 16GB / Driver 580.105.08
**conda env**: `alphatrade_cuda12` (JAX 0.9.1 + jax-cuda12-plugin)

---

## 已解决

**2026-03-03**: 切换到 `alphatrade_cuda12` 环境（CUDA 12 plugin 替代 CUDA 13 plugin）后，所有 GPU 问题已解决：
- `python script.py` 直接运行 GPU 训练：**通过**
- GPU + JIT=1：**通过**
- 不再需要 `python -c` workaround 或 `XLA_FLAGS` hack

**运行方式**:
```bash
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH python src/alphatrade/scripts/train_m4_alphatrade.py --smoke --max-steps 3
```

---

## 历史记录（旧 `alphatrade` env，jax-cuda13-plugin）

**旧环境**: JAX 0.9.0.1 + jax-cuda13-plugin 0.9.0.1 + cuDNN 9.19 / CUDA 13.0

## 症状

`python train_m4_alphatrade.py` 在 GPU 上崩溃，即使 JIT=0 也崩：

1. **model init (eager)**：XLA autotuner 在 `jnp.einsum()` 时触发 CUDA 非法地址访问
   ```
   Autotuning failed for HLO: %gemm_fusion_dot = f32[1,60,128]
   CUDA_ERROR_ILLEGAL_ADDRESS
   ```

2. **JIT training step**（autotuner 关闭后）：command buffer/CUDA graph capture 失败
   ```
   Failed to capture gpu graph: the requested functionality is not supported
   ```

## 测试矩阵

| 运行方式 | GPU+JIT=1 | GPU+JIT=0 | CPU |
|----------|-----------|-----------|-----|
| `python script.py` (直接运行) | ❌ | ❌ | ✅ |
| `python -c "import main; main()"` + XLA_FLAGS | ❌ (注1) | ✅ | ✅ |
| 独立 JIT 测试 (同样的 batch/model) | ✅ | ✅ | ✅ |

## 根因分析

**同样的代码逻辑，通过 `python -c` 调用正常，直接 `python script.py` 崩溃。**

已排除的因素：
- 脚本内 `os.environ["XLA_FLAGS"]` 设置（在 `import jax` 之前）→ 无效
- `JAX_DISABLE_MOST_OPTIMIZATIONS=1` → 无效
- 独立 workaround 模块 import → 无效
- batch shape / 数据内容差异 → 已验证一致

**结论**：JAX 0.9 的 `jax-cuda13-plugin` 在 Python interpreter 启动时通过 packaging entry_points 或 site-packages 机制预初始化了 XLA CUDA backend，此时 `XLA_FLAGS` 环境变量尚未被用户代码设置。`python -c` 模式下环境变量在 shell 层就已就位，所以能绕过这个时序问题。

`JAX_PLATFORMS=cpu` 能在脚本内生效是因为它控制的是 JAX 的 backend 选择逻辑（Python 层），不依赖 XLA C++ 层的初始化时序。

## GPU 使用方式

### 推荐：M5/M4 Runner（默认 GPU）

```bash
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/run_m5_sweep.py \
    --sweep-config configs/sweep/m5.yaml --smoke --resume

conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/run_m4_matrix.py \
    --seeds 42 43 44 --max-steps 500 --smoke
```

内部统一通过 `alphatrade.gpu_launcher` 用 `python -c` workaround + `XLA_FLAGS` 调 train/eval 子进程。如需强制 CPU 可加 `--no-gpu`，但正式训练/评估应使用 GPU；GPU 失败时先定位根因。

### 手动单脚本（需 shell 层设环境变量）

```bash
XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer=" \
JAX_PLATFORMS=cuda \
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH python -c "
import sys; sys.argv = ['train', '--smoke', '--max-steps', '1000', '--save-every', '100', '--seed', '42']
from alphatrade.scripts.train_m4_alphatrade import main; main()
"
```

**注1**: 当前 `alphatrade_cuda12` 环境下，M4/M5 smoke 已验证 GPU+JIT=1 可跑通。若后续全量训练再次遇到 `INTERNAL: the requested functionality is not supported` 或 CUDA graph capture 相关错误，先定位 JAX/CUDA/XLA 环境；可用 `--jit 0` 做对照排查，但正式训练/评估不应静默回退 CPU。

## 后续排查方向

1. **降级测试**：JAX 0.4.x + jax-cuda12-plugin 是否正常
2. **Plugin 分析**：检查 `jax-cuda13-plugin` 的 `entry_points` 和 `__init__` 是否触发 XLA 初始化
3. **Driver 更新**：NVIDIA driver 580 可能对 CUDA 13 支持不完整
4. **社区报告**：构造最小复现（`python script.py` vs `python -c` 的差异）提交 JAX issue
