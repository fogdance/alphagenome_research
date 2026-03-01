# M2-T2 Loss Sanity Check Report

生成时间: 2026-03-01 16:22:40

## 配置

- **Max steps**: 200
- **Quantiles**: [0.1, 0.3, 0.5, 0.7, 0.9]
- **Horizon weights**: [1.0, 1.0, 1.0, 1.0]
- **Crossing penalty**: True
- **Crossing weight**: 0.1

## Loss 统计

### Total Loss

- **First**: 0.106073
- **Last**: 0.003094
- **Min**: 0.003094
- **Mean**: 0.010719
- **Trend**: decreasing

### Pinball Loss

- **First**: 0.089192
- **Last**: 0.002877
- **Min**: 0.002877

### Crossing Penalty

- **First**: 0.168808
- **Last**: 0.002170
- **Min**: 0.002079
- **Active**: ✅

### By-Horizon Loss

| Horizon | First | Last | Min | Mean |
|---------|-------|------|-----|------|
| h1 | 0.032885 | 0.002524 | 0.002524 | 0.007839 |
| h5 | 0.118390 | 0.003210 | 0.003210 | 0.010687 |
| h20 | 0.099407 | 0.002949 | 0.002891 | 0.009893 |
| h60 | 0.106086 | 0.002825 | 0.002825 | 0.010454 |

## 收敛性检查

- **Loss decreased**: ✅
- **No NaN**: ✅
- **No Inf**: ✅
- **Stable**: ✅

### 总体状态: ✅ PASS

