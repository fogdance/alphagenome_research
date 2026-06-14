# Claude Kickstart

**AlphaTrade 的唯一大目标

运行环境: conda 的 alphatrade_cuda12

> 借鉴 AlphaGenome 的工程化思想（稳定训练、强表达、可扩展 trunk），但做成**严格因果**的期货多周期预测器：输入 1min 因果特征窗口 → 输出多个 horizon 的“未来涨跌/收益分布（分位数）”，并且能用统一评估框架持续迭代。
---

## 进入项目第一条命令

```bash
python src/alphatrade/scripts/validate_reports_schema.py --profile m4 --strict
```

全绿 = 项目状态正常。

---

## 当前里程碑

| 里程碑 | 状态 | 说明 |
|--------|------|------|
| M2 | 完成 | 数据管道 + 基础训练 |
| M3 | 完成 | 训练稳定性 |
| M4 | 完成 | 3-seed train+eval + 契约治理 |
| M5 | 准备中 | Ablation study（contract-first 已定义） |


---

## 关键路径索引

| 文件 | 用途                  |
|------|---------------------|
| `src/alphatrade/schemas/contracts_manifest.yaml` | Profile 定义（SoT）     |
| `src/alphatrade/schemas/*.schema.json` | JSON Schema 文件      |
| `docs/reports_governance.md` | 治理规则 + 变更流程         |
| `docs/Milestone.md` | 里程碑定义               |
| `src/alphatrade/scripts/validate_reports_schema.py` | Validator 脚本        |
| `$ALPHATRADE_RUNS_ROOT/reports/` | 所有实验报告输出，重点是FINAL报告 |
