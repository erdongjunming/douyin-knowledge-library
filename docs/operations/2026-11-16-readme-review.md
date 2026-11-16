# README 新增章节技术审查(2026-11-16)

## 背景

项目已公开到 GitHub(erdongjunming/douyin-knowledge-library)。README.md 新增了三章:
`Prerequisites`(依赖安装)、`Configuration`(`DKL_*` 环境变量表)、`Known issues and gotchas`。

你的任务:**只做只读技术审查,报告事实性错误。禁止修改任何文件,禁止 git commit。**

## 对照检查清单

逐条把 README 新增内容与以下源码核实:

1. **config.py**(`backend/src/douyin_library/config.py`)
   - 每个 `DKL_*` 环境变量的真实名称、默认值、枚举范围是否与 README 表格一致
   - 特别核对:storage_root 默认值、note_ollama_model、transcription_runtime 的枚举、qwen_offline 默认、embedded_workers、subtitle/cleaning 开关默认值
   - 外部 provider 的三个变量名(note_external_base_url/_model/_api_key)
2. **pyproject.toml**(`backend/pyproject.toml`)
   - Python 版本要求(requires-python)、依赖名、dev 组名(dependency-groups)
   - README 写的 `uv sync --group dev` 与 `uv pip install -e .` 是否可用(检查 uv.lock 存在与否)
   - douyin-downloader 的 source:path 指向是否正确(`../upstream/jiji`)
3. **dev.ps1**(`scripts/dev.ps1`)
   - 端口:FastAPI 8002、Vite 5175 是否与 README 一致
   - Ollama 查找路径、qwen3:4b 等待逻辑、`DKL_STOP_BACKEND_ON_EXIT` 语义(0=不杀后端?)是否与 README 表格一致
4. **local-runtime.ps1**(`scripts/local-runtime.ps1`)
   - 导入的用户环境变量名是否与 README/表格一致
5. **package.json**(`frontend/package.json`)
   - 前端框架版本表述(Vite 7、React 19)、npm scripts 是否与 README 一致
6. **.gitmodules** 与 `upstream/jiji`
   - README 说的 submodule 名称与初始化命令是否正确
7. **ffmpeg 解析逻辑**(config.py 的 `resolve_tool`)
   - README 写的三级解析顺序(PATH → DKL_FFMPEG_PATH → storage tools/ffmpeg)是否准确
8. **Known issues 章节**
   - 每条 claim 与代码/事实的对应关系(如 GPU lease、job 无自动重试、无 GET /api/jobs/<id> 路由)
   - 检查 routes/jobs.py 确认确实没有单个 job 的 GET 路由

## 输出要求(中文)

1. 结论一句话:通过 / 发现 N 处问题
2. 问题列表,每条含:位置(README 行号或章节)、当前表述、正确值(附源码依据:文件名+行号)、严重度(高/中/低)
3. 若全部核实无误,明确说"全部核实无误",不要含糊
4. 只报告,不修改
