# M3 Training Run - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-01 20:32:52

## 运行命令

```bash
python src/alphatrade/scripts/train_m3_alphatrade.py --config configs/dataset/m2.yaml --max-steps 10 --batch-size 4 --clip-norm 1.0 --jit 0 --smoke
```

## 配置

- Run ID: 60f040b9
- Backend: JAX
- JIT: disabled
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

- Train last: 0.315659
- Train best: 0.315659
- Val last: 0.321697
- Val best: 0.321697 @ step 10

### By-Horizon Loss

| Horizon | Train | Val |
|---------|-------|-----|
| h1 | 0.121714 | 0.079985 |
| h5 | 0.120135 | 0.074871 |
| h20 | 0.115135 | 0.075911 |
| h60 | 0.122069 | 0.090930 |

## Stability

- NaN steps: 0
- Inf steps: 0
- Max grad norm: 6.4506
- OOM count: 0
