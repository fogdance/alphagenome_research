# GPU 训练崩溃问题排查报告

**日期**: 2026-03-02
**环境**: RTX 4060 Ti 16GB / Driver 580.105.08 / CUDA 13.0
**软件**: JAX 0.9.0.1 + jax-cuda13-plugin 0.9.0.1 + cuDNN 9.19 / Haiku / Flax
**conda env**: alphatrade

---

## 当前决策

**单脚本默认 CPU**（train/eval 脚本顶部 fallback `JAX_PLATFORMS=cpu`）。

**Matrix runner 默认 GPU**（`run_m4_matrix.py` 的 `--gpu` 默认开启）。Matrix runner 内部用 `python -c` + shell 层 `XLA_FLAGS` 调子进程，自动绕过 CUDA 初始化 bug，无需手动设环境变量。

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

### 推荐：Matrix Runner（默认 GPU）

```bash
conda run -n alphatrade python src/alphatrade/scripts/run_m4_matrix.py \
    --seeds 42 43 44 --max-steps 500 --smoke
```

内部自动用 `python -c` workaround + `XLA_FLAGS` 调子进程。如需强制 CPU 可加 `--no-gpu`。

### 手动单脚本（需 shell 层设环境变量）

```bash
XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer=" \
JAX_PLATFORMS=cuda \
conda run -n alphatrade python -c "
import sys; sys.argv = ['train', '--smoke', '--max-steps', '1000', '--save-every', '100', '--seed', '42']
from alphatrade.scripts.train_m4_alphatrade import main; main()
"
```

**注1**: 之前小规模 JIT 测试通过，但全量数据 train_step JIT 仍报 `INTERNAL: the requested functionality is not supported`（CUDA graph capture 失败）。`python -c` workaround 仅解决 autotuner 崩溃，不解决 command buffer 问题。GPU+JIT=0 是当前唯一可用的 GPU 模式。

## 后续排查方向

1. **降级测试**：JAX 0.4.x + jax-cuda12-plugin 是否正常
2. **Plugin 分析**：检查 `jax-cuda13-plugin` 的 `entry_points` 和 `__init__` 是否触发 XLA 初始化
3. **Driver 更新**：NVIDIA driver 580 可能对 CUDA 13 支持不完整
4. **社区报告**：构造最小复现（`python script.py` vs `python -c` 的差异）提交 JAX issue
