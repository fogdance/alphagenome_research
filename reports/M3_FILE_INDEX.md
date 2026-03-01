# M3 文件索引

**生成时间**: 2026-03-01
**状态**: ✅ 全部完成

---

## 📂 文件结构

```
alphagenome_research/
├── src/alphatrade/scripts/
│   ├── m3_alphatrade_forward_smoke.py    # Forward 测试脚本
│   └── train_m3_alphatrade.py            # M3 训练脚本 (~600 行)
│
├── docs/
│   ├── m3_training_contract.md           # 训练契约（详细）
│   ├── m3_env_fix.md                     # 环境修复指南
│   └── M3_QUICK_REFERENCE.md             # 快速参考指南
│
└── reports/
    ├── M3_T0_SUMMARY.md                  # T0 总结
    ├── M3_T1_T2_SUMMARY.md               # T1+T2 总结
    ├── M3_PROGRESS.md                    # 进度报告
    ├── M3_FINAL_ACCEPTANCE.md            # 最终验收报告
    ├── M3_WORK_SUMMARY.md                # 工作总结
    ├── M3_FILE_INDEX.md                  # 本文件
    ├── m3_train_metrics.json             # 训练指标 (JSON)
    └── m3_train_run.md                   # 训练报告 (Markdown)
```

---

## 📄 文件说明

### 代码文件

#### 1. `src/alphatrade/scripts/m3_alphatrade_forward_smoke.py`
**用途**: Forward pass 测试脚本
**功能**:
- 创建 AlphaTrade v0.2 配置
- 初始化模型
- 运行 forward pass
- 验证输出 shape 和 NaN/Inf

**运行**:
```bash
conda activate alphatrade
python src/alphatrade/scripts/m3_alphatrade_forward_smoke.py
```

#### 2. `src/alphatrade/scripts/train_m3_alphatrade.py`
**用途**: M3 完整训练脚本
**功能**:
- 加载 M2 数据（转换为 JAX 数组）
- 初始化 AlphaTrade v0.2 模型
- 训练循环（支持 JIT 编译）
- 输出 metrics（M2 schema 兼容）

**运行**:
```bash
# Smoke test (10 steps)
python src/alphatrade/scripts/train_m3_alphatrade.py \
  --config configs/dataset/m2.yaml \
  --max-steps 10 \
  --batch-size 4 \
  --smoke \
  --jit 0

# 完整训练 (500 steps)
python src/alphatrade/scripts/train_m3_alphatrade.py \
  --config configs/dataset/m2.yaml \
  --max-steps 500 \
  --batch-size 128 \
  --smoke \
  --jit 1
```

---

### 文档文件

#### 3. `docs/m3_training_contract.md`
**用途**: M3 训练契约文档
**内容**:
- Schema 复用规则
- M3 特有字段定义
- 输出文件规范
- 数据复用说明
- 验收标准

**适用场景**: 了解 M3 的契约和规范

#### 4. `docs/m3_env_fix.md`
**用途**: 环境修复指南
**内容**:
- JAX/NumPy 兼容性问题
- 解决方案（3 种）
- 验证步骤
- 故障排查

**适用场景**: 遇到环境问题时查阅

#### 5. `docs/M3_QUICK_REFERENCE.md`
**用途**: 快速参考指南
**内容**:
- 快速开始命令
- 关键文件路径
- 训练命令示例
- 参数说明
- 验证检查
- 故障排查

**适用场景**: 日常使用的快速参考

---

### 报告文件

#### 6. `reports/M3_T0_SUMMARY.md`
**用途**: T0 任务总结
**内容**:
- Contract freeze 完成情况
- Schema 复用确认
- 输出文件定义

**适用场景**: 了解 T0 的完成情况

#### 7. `reports/M3_T1_T2_SUMMARY.md`
**用途**: T1+T2 任务总结
**内容**:
- Model import 完成情况
- Training script 实现细节
- 环境问题说明
- 下一步行动

**适用场景**: 了解 T1+T2 的完成情况

#### 8. `reports/M3_PROGRESS.md`
**用途**: M3 整体进度报告
**内容**:
- 总体进度统计
- 已完成工作
- 待执行工作
- 时间估算
- 风险评估

**适用场景**: 了解 M3 的整体进度

#### 9. `reports/M3_FINAL_ACCEPTANCE.md`
**用途**: 最终验收报告
**内容**:
- 任务完成度
- 核心成果验证
- 训练结果
- 技术实现
- 与 M2 对比
- 验收标准检查
- 后续建议

**适用场景**: 了解 M3 的最终验收情况

#### 10. `reports/M3_WORK_SUMMARY.md`
**用途**: 工作总结
**内容**:
- 成果概览
- 完成的任务
- 解决的问题
- 交付物清单
- 技术亮点
- 关键经验

**适用场景**: 了解 M3 的整体工作情况

#### 11. `reports/M3_FILE_INDEX.md`
**用途**: 文件索引（本文件）
**内容**:
- 文件结构
- 文件说明
- 使用场景

**适用场景**: 查找 M3 相关文件

---

### 输出文件

#### 12. `reports/m3_train_metrics.json`
**用途**: 训练指标（JSON 格式）
**内容**:
- run: 运行信息
- dataset: 数据集信息
- model: 模型信息
- training: 训练配置
- loss: 损失指标
- stability: 稳定性指标

**Schema**: `src/alphatrade/schemas/m2_train_metrics.schema.json`

**验证**:
```bash
python src/alphatrade/scripts/validate_reports_schema.py \
  --reports-dir reports \
  --schemas-dir src/alphatrade/schemas
```

#### 13. `reports/m3_train_run.md`
**用途**: 训练报告（Markdown 格式）
**内容**:
- 运行命令
- 配置信息
- 模型信息
- Loss 指标
- By-horizon loss
- Stability 指标

**适用场景**: 人类可读的训练报告

---

## 🔍 使用场景索引

### 场景 1: 第一次使用 M3
**推荐阅读顺序**:
1. `docs/M3_QUICK_REFERENCE.md` - 快速开始
2. `docs/m3_training_contract.md` - 了解契约
3. `reports/M3_FINAL_ACCEPTANCE.md` - 了解成果

### 场景 2: 遇到环境问题
**推荐阅读**:
1. `docs/m3_env_fix.md` - 环境修复指南
2. `docs/M3_QUICK_REFERENCE.md` - 故障排查部分

### 场景 3: 运行训练
**推荐阅读**:
1. `docs/M3_QUICK_REFERENCE.md` - 训练命令
2. `src/alphatrade/scripts/train_m3_alphatrade.py` - 查看参数

### 场景 4: 了解实现细节
**推荐阅读**:
1. `reports/M3_T1_T2_SUMMARY.md` - 实现总结
2. `src/alphatrade/scripts/train_m3_alphatrade.py` - 源代码

### 场景 5: 验收和评审
**推荐阅读**:
1. `reports/M3_FINAL_ACCEPTANCE.md` - 验收报告
2. `reports/M3_WORK_SUMMARY.md` - 工作总结
3. `reports/m3_train_metrics.json` - 训练指标

### 场景 6: 继续开发 M4
**推荐阅读**:
1. `reports/M3_FINAL_ACCEPTANCE.md` - 后续建议部分
2. `reports/M3_WORK_SUMMARY.md` - 后续工作部分

---

## 📊 文件统计

| 类型 | 数量 | 总行数（估算） |
|------|------|----------------|
| 代码文件 | 2 | ~800 行 |
| 文档文件 | 8 | ~3000 行 |
| 输出文件 | 2 | ~150 行 |
| **总计** | **12** | **~3950 行** |

---

## 🔗 相关文件

### M2 相关
- `reports/M2_FINAL_ACCEPTANCE.md` - M2 验收报告
- `src/alphatrade/schemas/m2_train_metrics.schema.json` - M2 schema

### M1 相关
- `reports/M1_FINAL_SUMMARY.md` - M1 总结
- `configs/universe/m1_selected.yaml` - 24 个品种

### M0.1 相关
- `reports/M0_1_FINAL_ACCEPTANCE.md` - M0.1 验收报告

---

## ✅ 检查清单

使用此清单确保所有文件都已创建：

- [x] ✅ m3_alphatrade_forward_smoke.py
- [x] ✅ train_m3_alphatrade.py
- [x] ✅ m3_training_contract.md
- [x] ✅ m3_env_fix.md
- [x] ✅ M3_QUICK_REFERENCE.md
- [x] ✅ M3_T0_SUMMARY.md
- [x] ✅ M3_T1_T2_SUMMARY.md
- [x] ✅ M3_PROGRESS.md
- [x] ✅ M3_FINAL_ACCEPTANCE.md
- [x] ✅ M3_WORK_SUMMARY.md
- [x] ✅ M3_FILE_INDEX.md
- [x] ✅ m3_train_metrics.json
- [x] ✅ m3_train_run.md

**状态**: ✅ 全部完成 (13/13)

---

**最后更新**: 2026-03-01
**维护者**: Claude (Opus 4.6)
