# M4 Training Run - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-01 23:04:56

## 运行命令

```bash
python src/alphatrade/scripts/train_m4_alphatrade.py --config configs/dataset/m2.yaml --max-steps 10 --batch-size 32 --clip-norm 1.0 --jit 1 --seed 44 --smoke
```

## 配置

- Run ID: 60ce2b21
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

- Train last: 0.334956
- Train best: 0.334956
- Val last: 0.305501
- Val best: 0.305501 @ step 10

### By-Horizon Loss

| Horizon | Train | Val |
|---------|-------|-----|
| h1 | 0.114861 | 0.065212 |
| h5 | 0.174008 | 0.094848 |
| h20 | 0.112178 | 0.070763 |
| h60 | 0.125908 | 0.074679 |

## Stability

- NaN steps: 0
- Inf steps: 0
- Max grad norm (pre-clip): 6.9422
- Max grad norm (post-clip): 0.2496
- OOM count: 0
