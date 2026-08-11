# DyNote 工作流设计检查点

日期：2026-08-11  
项目：`douyin-knowledge-library`  
分支：`feat/initial-implementation`  
状态：设计进行中，尚未进入 DyNote 能力移植实施

## 恢复说明

上下文压缩或新会话后，从本文件继续。下一步不是编写代码，而是继续 `superpowers:brainstorming` 的设计流程。用户已同意使用可视化伴侣，所以下一步应先完整读取 `skills/brainstorming/visual-companion.md`，再展示本地网页界面草图或布局对比。

当前精确节点：

1. 总体架构与主工作流已获得用户基本认可。
2. “jiji 主采集、Edge 只补证、转写决定笔记等级”的可靠处理管线已明确获得用户确认。
3. 用户已同意下一步使用浏览器可视化伴侣讨论网页界面。
4. 尚需完成网页信息架构、数据模型、错误处理、调度策略和测试设计。
5. 全部设计得到确认后，写正式设计文档到 `docs/superpowers/specs/`，自检并提交，等待用户审阅；之后才能调用 `superpowers:writing-plans`。

## 用户目标

打造一条本地优先、可批量增量运行的抖音知识资产工作流，功能目标以 `Rimagination/dy-note` 为基线，但原生集成到现有 FastAPI、React、SQLite 和 jiji 应用中，而不是把 DyNote Skill 当作外部黑盒调用。

已确认需求：

- 同步指定主播的公开视频。
- 同步登录账号的全部收藏视频，包括收藏夹。
- 规模为 100–1000 条，支持每周或每月增量运行。
- 以 `aweme_id` 去重。
- 运行数据保存在 `D:\DouyinKnowledgeLibrary`，不得静默回落到 C 盘。
- 本地网页支持主播选择、收藏同步、任务进度、转写阅读和主题筛选。
- 最终只保留音频、封面、元数据、转写和派生知识资产；原视频仅作临时文件。
- 评论采用按需采集，不阻塞默认批处理。
- AI 采用混合模式：字幕优先、本地 ASR 兜底，通用 OpenAI 兼容接口生成摘要和学习笔记。
- 第一阶段选择本地核心闭环，不包含豆包、抖音问 AI 深度自动化、OCR 和账号研究的完整等价实现。

## 选择的实现路径

采用“将 DyNote 工作流原生移植到现有应用”：

- 保留现有 FastAPI、React、SQLite、持久任务队列和 jiji 适配器。
- 移植 DyNote 的资产先行、证据分级、笔记预算、学习笔记和质量评分思想。
- 不围绕 DyNote 重建应用。
- 不把 DyNote Skill 作为后台服务接口。
- Get笔记仅可能作为后续可选导出目标，不作为主存储。

## 已确认的总体原则

目标不是“下载视频”，而是把短视频转换成可验证、可复用、可检索的知识资产。

- 事实层不可被模型覆盖：元数据、字幕、ASR 分段和评论原文独立保存。
- 生成层可以重建：更换模型或提示词时复用已有素材。
- 每个阶段独立持久化和恢复，失败只重试当前阶段。
- 主播、收藏和单链接最终进入同一作品队列。
- 笔记中的重要结论应能关联原始转写片段和时间戳。
- Cookie 存入 Windows 凭据库，不写入笔记、日志或元数据。

建议的单作品资产目录：

```text
D:\DouyinKnowledgeLibrary\works\<aweme_id>\
  audio.m4a
  cover.jpg
  metadata.json
  asset_manifest.json
  transcript.txt
  transcript.cleaned.md
  segments.json
  comments\
  note_budget.json
  learning_note.md
  note_score.json
```

## 已确认的可靠处理管线

```text
发现作品
-> jiji 获取完整媒体与元数据
-> FFmpeg 验证完整性
-> 提取并验证音频
-> 字幕或本地 ASR
-> 证据清洗与资产清单
-> 笔记预算
-> 学习笔记与评分
```

约束：

- jiji 是主播、收藏和单链接的主媒体入口。
- Edge/web-access 只用于抖音问 AI、章节和视觉补证，不能用 `<video>.currentSrc` 获取完整视频。
- 只有音频时长与源视频基本一致、文件非空且哈希生成成功，才能删除临时原视频。
- 独立字幕优先；中文无字幕时使用 Qwen3-ASR，失败后回退 faster-whisper。
- 有完整字幕或 ASR 才能生成正式学习笔记。
- 只有抖音 AI 章节时只能生成明确标注的快读草稿。
- 没有转写时必须标记高证据风险。
- 推荐视频、页脚和评论 DOM 不能混入目标视频时间线。

处理状态建议：

```text
discovered
-> metadata_ready
-> media_downloaded
-> audio_verified
-> transcript_ready
-> evidence_ready
-> note_ready
```

每个阶段记录输入哈希、处理器版本、尝试次数和错误原因。

## DyNote 实跑记录

测试分享链接最终解析为作品：`7670472545413401865`。

输出目录：

```text
D:\DouyinKnowledgeLibrary\dy-note\7670472545413401865
```

已完成的 quick-pass 产物：

- `douyin_ai_brief.json`
- `douyin_ai_brief.md`
- `assets/asset_manifest.json`
- `note_budget.json`
- `learning_note.md`
- `note_score.json`

验证结果：

- 证据等级：`douyin-web-ai-chapters`
- 学习笔记实际长度：1010 字
- 建议长度：800–1200 字
- DyNote 评分：`ok`
- Edge 后台标签页已关闭。
- 工作区在写入本检查点前为干净状态。

实跑发现：

1. DyNote 直接选择页面第一个媒体 URL，下载到的是 2.6 秒、无音轨的媒体分片，而页面主视频实际时长约 260.48 秒。
2. 该无效 MP4 已删除，不保留原视频。
3. 抖音 AI 章节在 03:58 后混入推荐视频、评论和页脚内容，不能整体视作目标视频证据。
4. `note_budget.json` 在零转写时仍判断视觉风险为低，证据闸门存在缺陷。
5. 现有 jiji 安全存储中没有 Cookie，因此本次未能取得完整媒体和 ASR 转写。
6. 评论没有采集，符合“按需采集”的已确认策略。

## 当前本机环境

- 已安装 DyNote Skill：`C:\Users\CJM\.codex\skills\dy-note`
- 已安装并审计 web-access Codex 适配版：`C:\Users\CJM\.codex\skills\web-access`
- Edge CDP 已验证可连接端口 `9222`。
- FFmpeg 已从 Gyan 下载并通过 SHA-256 校验：
  - 目录：`D:\DouyinKnowledgeLibrary\tools\ffmpeg`
  - 下载包 SHA-256：`e6b54767a6065919048f1a098eb27211ca4e12b4348a05d88777a5855d0b6e71`
- Qwen3-ASR 共享环境尚未安装。
- Whisper CLI 曾检测到，但应用 Python 环境未配置为可靠后端。

## 下一步设计重点

使用可视化伴侣讨论本地网页信息架构。界面至少应覆盖：

- 总览与系统诊断。
- 主播管理和增量同步。
- 全部收藏与收藏夹同步。
- 任务队列、阶段进度、失败原因和重试。
- 作品库、主题筛选和全文搜索。
- 作品详情：音频播放器、转写分段、证据等级、学习笔记和按需评论分析。
- 设置：Douyin Cookie、FFmpeg、ASR、OpenAI 兼容接口和调度周期。

界面设计必须体现证据状态，不能只显示一个笼统的“已完成”：至少区分“仅元数据”“快读草稿”“完整转写”“正式笔记”“需要视觉复核”。

## 禁止提前实施

正式设计尚未全部确认。恢复后不要直接修改后端或前端代码，也不要调用实现类技能。先完成剩余设计章节并获得用户批准，再按 brainstorming 流程写正式 spec、提交、等待审阅，最后转入 writing-plans。
