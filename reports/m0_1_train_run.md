# M0.1 Baseline Training Run

生成时间: 2026-02-28 17:35:38

## 运行命令

```bash
python src/alphatrade/scripts/train_m0_1_baseline.py \
  --config src/alphatrade/configs/m0_1_dataset.yaml \
  --seed 42 \
  --max-steps 200 \
  --batch-size 128 \
  --num-workers 2 \
  --device auto
```

## 配置摘要

- **Dataset**: m0_1
- **Symbols**: DCE.JM, SHFE.RB
- **Features**: 8 (ret_1m, hl_range, co_change...)
- **Lookback**: 60
- **Stride**: 5
- **Horizons**: [1, 5, 20, 60]

## 数据量统计

- **Train total**: 10,870
- **Val total**: 1,452

### 按 Symbol

- **DCE.JM**: train=6,333, val=726
- **SHFE.RB**: train=4,537, val=726

## 训练结果

- **Device**: cuda
- **Max steps**: 200
- **Batch size**: 128
- **Learning rate**: 0.001

### Loss

- **Train last**: 0.000155
- **Train best**: 0.000155
- **Val last**: 0.000134
- **Val best**: 0.000134 (step 200)

