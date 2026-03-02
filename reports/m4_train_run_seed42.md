# M4 Training Run - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-01 23:01:10

## 运行命令

```bash
python src/alphatrade/scripts/train_m4_alphatrade.py --config configs/dataset/m2.yaml --max-steps 10 --batch-size 32 --clip-norm 1.0 --jit 1 --seed 42 --smoke
```

## 配置

- Run ID: c81018c9
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

- Train last: 0.327098
- Train best: 0.327098
- Val last: 0.302739
- Val best: 0.302739 @ step 10

### By-Horizon Loss

| Horizon | Train | Val |
|---------|-------|-----|
| h1 | 0.123812 | 0.071263 |
| h5 | 0.122633 | 0.073240 |
| h20 | 0.115851 | 0.070366 |
| h60 | 0.128888 | 0.087870 |

## Stability

- NaN steps: 0
- Inf steps: 0
- Max grad norm (pre-clip): 6.3908
- Max grad norm (post-clip): 0.2496
- OOM count: 0
