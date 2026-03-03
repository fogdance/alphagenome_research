# M4 Training Run - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 13:28:31

## 运行命令

```bash
python src/alphatrade/scripts/train_m4_alphatrade.py --config configs/dataset/m2.yaml --max-steps 3 --batch-size 128 --clip-norm 1000000000.0 --jit 1 --seed 43 --smoke
```

## 配置

- Run ID: b61aefa4
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

- Train last: 0.598963
- Train best: 0.598963
- Val last: 0.518546
- Val best: 0.518546 @ step 3

### By-Horizon Loss

| Horizon | Train | Val |
|---------|-------|-----|
| h1 | 0.175153 | 0.124501 |
| h5 | 0.174236 | 0.168140 |
| h20 | 0.186504 | 0.118090 |
| h60 | 0.149697 | 0.107815 |

## Stability

- NaN steps: 0
- Inf steps: 0
- Max grad norm (pre-clip): 6.7053
- Max grad norm (post-clip): 0.2503
- OOM count: 0

## Checkpoint

- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/no_clip_seed43`
- Best checkpoint: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/no_clip_seed43/best` @ step 3
- Last checkpoint: step 3
- Save every: 1 steps
- Keep last: 3

### 恢复训练

```bash
# 从最后的checkpoint恢复
python src/alphatrade/scripts/train_m4_alphatrade.py --resume last --resume-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/no_clip_seed43

# 从最佳checkpoint恢复
python src/alphatrade/scripts/train_m4_alphatrade.py --resume best --resume-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/no_clip_seed43
```

### 评估命令

```bash
# 使用训练好的checkpoint评估
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m4_train_metrics.json \
  --ckpt-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/no_clip_seed43/best \
  --dataset-config configs/dataset/m2.yaml \
  --split val
```
