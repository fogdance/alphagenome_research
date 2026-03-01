# M2 Training Run (Final)

生成时间: 2026-03-01 16:31:31

## 运行命令

```bash
python src/alphatrade/scripts/train_m2_final.py --config configs/dataset/m2.yaml --smoke
```

## 配置

- Run ID: ad84eedf
- Device: cuda
- Symbols: 3
- Train samples: 57,418
- Val samples: 11,313

## Loss

- Train last: 0.003094
- Val best: 0.002275 @ step 200

### By-Horizon Loss

| Horizon | Train | Val |
|---------|-------|-----|
| h1 | 0.002707 | 0.002382 |
| h5 | 0.003470 | 0.002488 |
| h20 | 0.003031 | 0.002339 |
| h60 | 0.002962 | 0.001889 |
