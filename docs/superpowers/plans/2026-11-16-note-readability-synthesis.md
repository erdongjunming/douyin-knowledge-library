# 笔记管线改造：可读合成方案（实施步骤，交给 Claude Code / Codex 执行）

> 背景与动机：`docs/operations/2026-11-16-note-quality-comparison.md`（用户逐维度对比结论）。
> 一句话目标：把"语义版"从证据转储改回可读笔记——保留 atom 证据层，恢复可读骨架，加确定性质量闸门。
> 约束：只改笔记生成/渲染层，不动 extract/review 证据阶段；不重下视频、不覆盖源资产；旧笔记不批量重生成。

## 根因定位（已核实，勿重复排查）

1. **重复膨胀总源头**：`notes/prompts.py` SYNTHESIZE_SYSTEM_PROMPT 中
   "每个纳入 atom 必须同时出现在主题结构和源顺序时间线中"
   + `notes/render.py` `_render_learning_note`（约 298–353 行）把 teaching sections（八章节）与
   synthesis.themes / methods_and_steps / examples_and_counterexamples / boundaries_and_risks 都输出，
   同一批 atom 原文在"核心观点—逐段证据—主题复盘—时间线"出现 3–4 次。

2. **方法与步骤漏判**：OUTLINE/SECTION prompt 让 4B 模型自行分桶，`深度审查`只报告不修复，
   分错桶（如三个方法被标"视频未提供"）无第二道拦截。确定性规则可解，不必依赖模型。

3. **passed 名不副实**：`service.py` 的 quality_gate == "passed" 只反映证据覆盖（格式合规），
   没有可读性/语义闸门。

## 改动步骤（按顺序执行，每步可独立验证）

### Step 1 — SYNTHESIZE prompt 去重（一行改动，影响最大）

文件：`backend/src/douyin_library/notes/prompts.py`，SYNTHESIZE_SYSTEM_PROMPT。

原文（约 49 行）：
```
每个纳入 atom 必须同时出现在主题结构和源顺序时间线中。
```
改为：
```
每个纳入 atom 必须出现在源顺序时间线中（时间线是全量事实的载体）。
主题结构只做归类与提炼：每个主题项给出 atom_ids、timestamps 和一句不超过 30 字的概括，
不得在主题结构中重复 atom 的完整原文。
```
同步把 prompt 版本号 `SYNTHESIZE_PROMPT_VERSION` 升一档（如 evidence-synthesize-v2），
保证审计可区分新旧输出。

### Step 2 — render 输出去重

文件：`backend/src/douyin_library/notes/render.py`，`_render_learning_note`。

现状：`_append_teaching_sections(lines, generation)` 后接
`_append_sections(lines, "主题复盘", synthesis.themes)` 等四个 synthesis 输出。

改法（二选一，推荐 a）：
- a) 保留八章节 teaching sections 作为正文；把 synthesis 的 themes/methods_and_steps/
  examples_and_counterexamples/boundaries_and_risks 合并为一个"主题复盘"小节，
  每条只输出 atom_ids + 一句话概括（对齐 Step 1 的新输出结构），不再重复原文。
- b) 若 synthesis 输出已含全文，则 teaching sections 的"逐段证据"章节降级为
  只列 atom_id 映射表（ID → 时间戳），全文只出现在时间线。

无论 a/b：同一条 atom 的**完整原文**在整篇渲染输出中最多出现 2 次（正文一次 + 时间线一次），
atom_id 引用可以多次。保留"推广与行动号召"单独小节（当前更好）。

### Step 3 — 方法与步骤空桶确定性闸门

位置：`backend/src/douyin_library/notes/coverage.py`（或 service.py 的 gate 判定前，选择不破坏现有
quality_gate 语义的位置），新增纯规则函数（不调模型）：

```
def _methods_section_suspicious(outline, source_text) -> bool:
    若 outline.methods_steps 标记为缺失（"视频未提供"）且 source_text 命中序列信号
    （正则：第[一二三四五]|第一步|第二步|然后|接着|最后|首先|之后|比如|例如 出现 ≥ 4 次）
    → 返回 True
```
为 True 时：把 quality_gate 从 passed 降为 `passed_with_warnings`，
并在 generation_audit 记录 `methods_steps_suspicious_missing`。
（可选增强：触发一次 bounded repair 重新分桶，初版先做告警不做自动修复。）

### Step 4 — 可读性闸门（passed 的语义侧校验）

位置：`backend/src/douyin_library/notes/service.py` quality_gate 判定附近（约 490–525 行）。

在现有 evidence coverage 检查通过之后、置 passed 之前，追加确定性校验：
1. 空章节计数：rendered sections 中固定文案"视频未提供这一类信息"出现 ≥ 3 次 → 不通过；
2. 残句检测：核心观点条目中字数 ≤ 6 且不含动词的条目占比 > 20% → 不通过；
3. 膨胀度：输出总字符数 / 唯一 atom 数 > 900 → 不通过（当前样本 630 行/不足百 atom 远超此值）。

任一不通过 → quality_gate = `passed_with_warnings`，note_score.json 增加
`readability` 分项（0–100，由上述三项线性合成），learning_note.md 头部的质量行如实显示
"质量状态：passed_with_warnings（可读性 X 分）"。**不得**直接标 failed（证据覆盖仍是硬闸门）。

### Step 5 — 术语表与转写疑点聚合（render 层）

文件：`backend/src/douyin_library/notes/render.py`。
- 术语表：从 synthesis 的概念/专名字段 + atom 高频名词（规则：连续 2+ 次出现的中文 2–6 字词，
  过滤停用词）提取，输出"术语表"小节。初版规则提取即可，不做模型调用。
- 转写疑点：聚合所有 disposition=unintelligible_asr 与 ASR uncertainty 标记的 atom，
  输出"转写疑点清单"小节（格式：`时间戳 · 原文片段`），替代目前散落的标注。

### Step 6 — OUTLINE/SECTION prompt 微调（防回归）

文件：`backend/src/douyin_library/notes/prompts.py`。
- OUTLINE_SYSTEM_PROMPT 追加："有序流程、做法、'怎么做'类内容必须分入 methods_steps；
  只有当源材料完全没有此类内容时才允许使用缺失文案。"
- SECTION_SYSTEM_PROMPT 的 missing 文案条款同样追加："unprovided_items 仅允许在源材料
  确实没有对应类别时使用；源材料含有序步骤或做法时不得声明缺失。"
- 同步提升 OUTLINE_PROMPT_VERSION / SECTION_PROMPT_VERSION 版本号。

## 验证（必须实跑，不许只跑单测）

1. 单元：`cd backend && .venv\Scripts\python.exe -m pytest tests/ -k note -q`（先确认现有绿）。
2. 样本重生成：
   ```
   cd backend
   .venv\Scripts\python.exe -m douyin_library.cli generate-notes --dry-run --aweme-id 6978749729874218247
   .venv\Scripts\python.exe -m douyin_library.cli generate-notes --force --aweme-id 6978749729874218247
   ```
   旧 bundle 自动进 versions/，可随时比对。
3. 验收标准（对照样本 6978749729874218247 及另外随机 3 篇）：
   - "方法与步骤"非空且含主动无聊/舒尔特方格/切换场景/声音辅助四法；
   - 固定文案"视频未提供这一类信息"出现次数 ≤ 2 且不与原文矛盾；
   - 同一 atom 完整原文在输出中出现 ≤ 2 次；总行数较改造前显著下降（预期 < 300 行）；
   - 新笔记头部质量行显示 passed 或 passed_with_warnings（非误导性 passed）。
4. 回归：跑 3–5 个随机 aweme_id 的 --force 重生成，确认无渲染异常、无证据链断裂。

## 禁止事项

- 不改 extract / review / repair 的校验逻辑（证据链是好的，只改输出层）。
- 不引入新的外部依赖或网络调用；术语提取用 stdlib/re 完成。
- 不批量重生成历史笔记；只影响新生成与显式 --force 的目标。
- 遵守 README Hard constraints（不重下媒体、键不进 payload、证据链接不断）。
