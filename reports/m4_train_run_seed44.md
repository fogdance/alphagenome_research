# M4 Training Run - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 00:37:50

## 运行命令

```bash
python src/alphatrade/scripts/train_m4_alphatrade.py --config configs/dataset/m2.yaml --max-steps 500 --batch-size 128 --clip-norm 1.0 --jit 0 --seed 44
```

## 配置

- Run ID: 66dec787
- Backend: JAX
- JIT: disabled
- Device: cuda
- Symbols: 24
- Train samples: 387,673
- Val samples: 74,459

## 模型

- Type: AlphaTrade v0.2
- Backend: JAX
- Total params: 6,274,774
- d_model: 256
- Transformer layers: 4

## Loss

- Train last: 0.016444
- Train best: 0.013565
- Val last: 0.013364
- Val best: 0.013364 @ step 500

### By-Horizon Loss

| Horizon | Train | Val |
|---------|-------|-----|
| h1 | 0.004136 | 0.003149 |
| h5 | 0.003699 | 0.003631 |
| h20 | 0.003633 | 0.002592 |
| h60 | 0.004484 | 0.003992 |

## Stability

- NaN steps: 0
- Inf steps: 0
- Max grad norm (pre-clip): 7.1594
- Max grad norm (post-clip): 0.2497
- OOM count: 0

## Checkpoint

- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed44`
- Best checkpoint: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed44/best` @ step 500
- Last checkpoint: step 500
- Save every: 100 steps
- Keep last: 3

### 恢复训练

```bash
# 从最后的checkpoint恢复
python src/alphatrade/scripts/train_m4_alphatrade.py --resume last --resume-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed44

# 从最佳checkpoint恢复
python src/alphatrade/scripts/train_m4_alphatrade.py --resume best --resume-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed44
```

### 评估命令

```bash
# 使用训练好的checkpoint评估
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m4_train_metrics.json \
  --ckpt-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed44/best \
  --dataset-config configs/dataset/m2.yaml \
  --split val
```
