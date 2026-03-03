# M4 Training Run - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 13:39:13

## 运行命令

```bash
python src/alphatrade/scripts/train_m4_alphatrade.py --config configs/dataset/m2.yaml --max-steps 3 --batch-size 256 --clip-norm 1.0 --jit 1 --seed 44 --smoke
```

## 配置

- Run ID: 6e9bb758
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

- Train last: 0.668170
- Train best: 0.668170
- Val last: 0.585839
- Val best: 0.585839 @ step 3

### By-Horizon Loss

| Horizon | Train | Val |
|---------|-------|-----|
| h1 | 0.166214 | 0.119571 |
| h5 | 0.256251 | 0.189326 |
| h20 | 0.144571 | 0.116440 |
| h60 | 0.147289 | 0.160501 |

## Stability

- NaN steps: 0
- Inf steps: 0
- Max grad norm (pre-clip): 6.9947
- Max grad norm (post-clip): 0.2496
- OOM count: 0

## Checkpoint

- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/batch_256_seed44`
- Best checkpoint: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/batch_256_seed44/best` @ step 3
- Last checkpoint: step 3
- Save every: 1 steps
- Keep last: 3

### 恢复训练

```bash
# 从最后的checkpoint恢复
python src/alphatrade/scripts/train_m4_alphatrade.py --resume last --resume-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/batch_256_seed44

# 从最佳checkpoint恢复
python src/alphatrade/scripts/train_m4_alphatrade.py --resume best --resume-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/batch_256_seed44
```

### 评估命令

```bash
# 使用训练好的checkpoint评估
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m4_train_metrics.json \
  --ckpt-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/batch_256_seed44/best \
  --dataset-config configs/dataset/m2.yaml \
  --split val
```
