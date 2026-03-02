# M4 Training Run - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-01 23:03:03

## 运行命令

```bash
python src/alphatrade/scripts/train_m4_alphatrade.py --config configs/dataset/m2.yaml --max-steps 10 --batch-size 32 --clip-norm 1.0 --jit 1 --seed 43 --smoke
```

## 配置

- Run ID: cc01971e
- Backend: JAX
- JIT: enabled
- Device: cuda
- Symbols: 3
- Train samples: 57,418
- Val samples: 11,313

## 模型

- Type: AlphaTrade v0.2
- Backend: JAX
- Total params: 6,274,774
- d_model: 256
- Transformer layers: 4

## Loss

- Train last: 0.279281
- Train best: 0.279281
- Val last: 0.258886
- Val best: 0.258886 @ step 10

### By-Horizon Loss

| Horizon | Train | Val |
|---------|-------|-----|
| h1 | 0.119000 | 0.067091 |
| h5 | 0.113663 | 0.073602 |
| h20 | 0.121047 | 0.065493 |
| h60 | 0.090296 | 0.052700 |

## Stability

- NaN steps: 0
- Inf steps: 0
- Max grad norm (pre-clip): 4.7166
- Max grad norm (post-clip): 0.2498
- OOM count: 0
