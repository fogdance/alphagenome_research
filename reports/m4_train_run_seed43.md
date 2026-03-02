# M4 Training Run - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-02 20:57:00

## 运行命令

```bash
python src/alphatrade/scripts/train_m4_alphatrade.py --config configs/dataset/m2.yaml --max-steps 20 --batch-size 128 --clip-norm 1.0 --jit 0 --seed 43 --smoke
```

## 配置

- Run ID: 5a685bdc
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

- Train last: 0.208618
- Train best: 0.208618
- Val last: 0.197509
- Val best: 0.197509 @ step 20

### By-Horizon Loss

| Horizon | Train | Val |
|---------|-------|-----|
| h1 | 0.068041 | 0.053359 |
| h5 | 0.065926 | 0.051996 |
| h20 | 0.055823 | 0.052567 |
| h60 | 0.053385 | 0.039587 |

## Stability

- NaN steps: 0
- Inf steps: 0
- Max grad norm (pre-clip): 6.7054
- Max grad norm (post-clip): 0.2498
- OOM count: 0

## Checkpoint

- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed43`
- Best checkpoint: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed43/best` @ step 20
- Last checkpoint: step 20
- Save every: 100 steps
- Keep last: 3

### 恢复训练

```bash
# 从最后的checkpoint恢复
python src/alphatrade/scripts/train_m4_alphatrade.py --resume last --resume-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed43

# 从最佳checkpoint恢复
python src/alphatrade/scripts/train_m4_alphatrade.py --resume best --resume-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed43
```

### 评估命令

```bash
# 使用训练好的checkpoint评估
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m4_train_metrics.json \
  --ckpt-dir /home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed43/best \
  --dataset-config configs/dataset/m2.yaml \
  --split val
```
