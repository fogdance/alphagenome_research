---

## AlphaTrade 契约治理规则（团队共识版）

### 0. 目标

确保任何人说“严格按契约”时，指的是**同一套可执行规则**，并且：

* 人读文档、机器 schema、CI 裁判三者一致
* 允许演进，但必须版本化、可回溯、可兼容
* 任何改动都能在 PR/CI 中被正确拦截或放行

---

## 1）权威顺序（谁说了算）

### ✅ 最终裁判：`validate_reports_schema.py` + CI

* CI 的判定结果就是“过/不过”的唯一事实来源
* CI 失败 = 契约不满足（无论文档怎么写）

### ✅ 可执行契约：`schemas/*.schema.json`

* Schema 是机器可执行的契约定义
* 任何字段名、required、类型、文件存在性要求，都必须以 schema 为准

### ✅ 人类说明书：`docs/*_reports_contract.md`

* 文档只负责解释口径/意图/示例
* 文档不得与 schema 冲突；冲突时以 schema/CI 为准

> 一句话：**CI 裁判 > Schema 定义 > Docs 解释**

---

## 2）单一权威源（Single Source of Truth）

你们要明确：**“契约”在仓库里的唯一权威源是 schema**。
Docs 必须从 schema 派生（或至少引用 schema 版本），validator 必须只按 schema 判。

**硬性规则：**

* 任何 PR 如果改了 reports 输出结构（字段名、类型、文件名、必选/可选），必须同步改 schema
* docs 必须引用具体 schema 文件名（含版本），不能只写“满足契约”

---

## 3）版本化规则（Strict = 对某个版本严格）

### ✅ Schema 必须带版本号

建议命名：

* `m4_eval_metrics_v1.schema.json`
* `m4_eval_metrics_v2.schema.json`

并在报告 json 里写入：

* `schema_name`
* `schema_version`

### ✅ 变更规则

* **新增字段**：允许（prefer 放到 `extras` / `optional`），不破坏旧版
* **删除/改名/改类型/改 required**：必须 bump major（或至少 v+1），并提供兼容策略

> 结论：严格不是“永远不变”，严格是“对 vN 严格”。

---

## 4）兼容策略（避免历史遗留把当前卡死）

你们可以选一种，团队必须统一：

### 方案 A（推荐）：Validator 支持多 profile

`validate_reports_schema.py` 支持：

* `--profile current`：只验当前 milestone 的产物 + 当前 schema
* `--profile full`：验全量（M0~Mx）
* `--profile legacy`：验历史产物（用于回归）

CI 默认跑：

* PR 影响 M4 → 跑 `current(M4)` + `full`（可选 nightly）

### 方案 B：永远全量验收

可以，但代价是：任何历史 schema 漂移都会阻塞现在开发（你们已经感受到痛了）。

---

## 5）文件存在性契约（你们这次踩坑的点）

“m4 非 seed 的文件不存在”这类问题要写成明确规则：

### ✅ 规则：Schema 不得隐含“文件集合”

* 如果是 multi-seed 输出：schema 应该描述 `runs[]` 列表或 `seed_metrics_map`
* 不应要求固定存在 `reports/m4_train_metrics.json` 这种“单文件汇总”，除非你们明确规定必须产出

建议统一：

* 每个 seed 产出：`reports/m4/train_metrics_seed{seed}.json`
* 再产出一个汇总：`reports/m4/matrix_summary.json`（可选但一旦规定就要 schema）

---

## 6）PR 流程（让所有人自然遵守）

### PR 模板必须加勾选项

* [ ] 我改了 reports 输出 → 我已同步更新对应 schema
* [ ] 我 bump 了 schema version → 我更新了 docs 引用版本
* [ ] 我变更 required/字段名/类型 → 我提供了兼容策略（validator profile / 旧 schema 保留 / migration）
* [ ] 我本地跑过：`python .../validate_reports_schema.py --profile current`

### CODEOWNERS（建议）

* `schemas/**`、`validate_reports_schema.py` 必须由“契约管理员/你”或指定 reviewer approve

---

## 7）一句话团队共识（建议贴在文档开头）

> **契约以 schema 为准，CI 为最终裁判；docs 负责解释。所有契约变更必须版本化，并由 validator profile 管理兼容。**

---