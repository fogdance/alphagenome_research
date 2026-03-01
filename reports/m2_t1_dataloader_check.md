# M2-T1 Dataloader Check Report

生成时间: 2026-03-01 16:16:24

## 配置

- **Processed dir**: `data/processed/m1_f8`
- **Feature dim**: 8
- **Lookback**: 60
- **Horizons**: [1, 5, 20, 60]

## Dataset 统计

- **Total samples**: 57,418
- **Symbols**: 3

### Symbol Sample Counts

| Symbol | Samples |
|--------|----------|
| CZCE.MA | 7,358 |
| DCE.JM | 5,726 |
| SHFE.AG | 44,334 |

## 验证结果

- **Total checks**: 19
- **Valid**: 19 (100.0%)
- **Invalid**: 0

### Random Samples

| Symbol | X Shape | Y Shape | Segment OK | Valid |
|--------|---------|---------|------------|-------|
| DCE.JM | (60, 8) | (4,) | ✅ | ✅ |
| CZCE.MA | (60, 8) | (4,) | ✅ | ✅ |
| SHFE.AG | (60, 8) | (4,) | ✅ | ✅ |
| SHFE.AG | (60, 8) | (4,) | ✅ | ✅ |
| SHFE.AG | (60, 8) | (4,) | ✅ | ✅ |
| SHFE.AG | (60, 8) | (4,) | ✅ | ✅ |
| CZCE.MA | (60, 8) | (4,) | ✅ | ✅ |
| SHFE.AG | (60, 8) | (4,) | ✅ | ✅ |
| SHFE.AG | (60, 8) | (4,) | ✅ | ✅ |
| SHFE.AG | (60, 8) | (4,) | ✅ | ✅ |

### Per-Symbol Samples

| Symbol | X Shape | Y Shape | Segment OK | Valid |
|--------|---------|---------|------------|-------|
| DCE.JM | (60, 8) | (4,) | ✅ | ✅ |
| DCE.JM | (60, 8) | (4,) | ✅ | ✅ |
| DCE.JM | (60, 8) | (4,) | ✅ | ✅ |
| SHFE.AG | (60, 8) | (4,) | ✅ | ✅ |
| SHFE.AG | (60, 8) | (4,) | ✅ | ✅ |
| SHFE.AG | (60, 8) | (4,) | ✅ | ✅ |
| CZCE.MA | (60, 8) | (4,) | ✅ | ✅ |
| CZCE.MA | (60, 8) | (4,) | ✅ | ✅ |
| CZCE.MA | (60, 8) | (4,) | ✅ | ✅ |

### 错误详情

✅ 无错误

