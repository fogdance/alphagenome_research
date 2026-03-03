# M4 Training Run - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 11:04:59

## 运行命令

```bash
python src/alphatrade/scripts/train_m4_alphatrade.py --config configs/dataset/m2.yaml --max-steps 3 --batch-size 128 --clip-norm 1.0 --jit 1 --seed 42 --smoke
```

## 配置

- Run ID: 38a265ee
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

- Train last: 0.571320
- Train best: 0.571320
- Val last: 0.503533
- Val best: 0.503533 @ step 3

### By-Horizon Loss

| Horizon | Train | Val |
|---------|-------|-----|
| h1 | 0.171753 | 0.124602 |
| h5 | 0.178127 | 0.127563 |
| h20 | 0.169145 | 0.124281 |
| h60 | 0.148330 | 0.127087 |

## Stability

- NaN steps: 0
- Inf steps: 0
- Max grad norm (pre-clip): 6.3555
- Max grad norm (post-clip): 0.2496
- OOM count: 0

## Checkpoint

- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/baseline_seed42`
- Best checkpoint: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/baseline_seed42/best` @ step 3
- Last checkpoint: step 3
- Save every: 1 steps
- Keep last: 3

### 恢复训练

```bash
# 从最后的checkpoint恢复
python src/alphatrade/scripts/train_m4_alphatrade.py --resume last --resume-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/baseline_seed42

# 从最佳checkpoint恢复
python src/alphatrade/scripts/train_m4_alphatrade.py --resume best --resume-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/baseline_seed42
```

### 评估命令

```bash
# 使用训练好的checkpoint评估
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m4_train_metrics.json \
  --ckpt-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/baseline_seed42/best \
  --dataset-config configs/dataset/m2.yaml \
  --split val
```
