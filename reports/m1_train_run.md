# M1 Smoke Training Run

生成时间: 2026-03-01 12:29:40

## 运行命令

```bash
python src/alphatrade/scripts/train_m1_smoke.py \
  --universe configs/universe/m1_selected.yaml \
  --processed-dir data/processed/m1_f8 \
  --max-steps 500 \
  --batch-size 256 \
  --lr 0.0001 \
  --seed 42
```

## 配置摘要

- **Run ID**: 532b1d98
- **Git SHA**: 842f3f26
- **Device**: cuda
- **Seed**: 42

### Dataset

- **Universe**: 24 symbols
- **Train samples**: 387,673
- **Val samples**: 74,459
- **Feature dim**: 8
- **Lookback**: 60
- **Horizons**: [1, 5, 20, 60]

### Model

- **Type**: SimpleMLPModel
- **Total params**: 164,548

### Training

- **Max steps**: 500
- **Batch size**: 256
- **Learning rate**: 0.0001
- **Grad clip**: 1.0

### Loss

- **Train last**: 0.000113
- **Train best**: 0.000106
- **Val last**: 0.000041
- **Val best**: 0.000041 (step 500)

### Status

✅ **Training completed successfully**

- No NaN/Inf detected
- Loss converged normally
