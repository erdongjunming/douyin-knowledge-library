# 2026-11-16 项目整理：creator sync 全链路验证 + ASR 环境修复

## 项目现状（截至 2026-11-16）

- 数据规模：~2.3k 条视频（metadata / transcript / dy-note 各一份），来源主要是收藏同步（favorites）
- 运行形态：start-local.cmd → dev.ps1 四进程（Ollama qwen3:4b + FastAPI:8002 + general/ASR worker + Vite:5175）
- 数据根：`D:\DouyinKnowledgeLibrary`；项目代码：`Documents\Codex\2026-08-06\hermes-deepseek-3\outputs\douyin-knowledge-library`

## 今日完成：定向收集博主全部视频（creator sync）已验证可用

代码链此前已存在但从未跑通（缺登录态 cookie）：

1. 配置抖音 Cookie：用户浏览器登录"无铭"号（非运营号"嘉大学绅"）→ 前端设置页「账号凭据」粘贴 → `PUT /api/settings/douyin-cookie` 先验证后落库（Windows 凭据管理器，分块+SHA256）
2. `POST /api/creators/preview` 验证：起航✨、全能的野人（203775400）均正确解析
3. `POST /api/creators/sync {"reference":"<主页链接>","confirmed":true}` → job #8221 `sync_creator` succeeded，野人全部作品入 Work 表
4. 结论：**"定向收集指定博主全部视频"功能可用**。采集网关 = `upstream/jiji_gateway.py`（依赖 git submodule `upstream/jiji`，github.com/jiji262/douyin-downloader），限速 1.5 req/s，分页 cursor 续传，增量去重

## 故障与修复：ASR 转写管线全挂（qwen_asr_result_invalid）

**症状**：sync 产出的 transcribe_work job 全部报 `qwen_asr_result_invalid`（transcription/service.py `_save_transcript`：text 空或 segments 非 list）。

**根因链**：
1. ASR resident server 由 worker 用 `E:\CosyVoiceRealtime\env\python.exe` 启动（DKL_QWEN_PYTHON_PATH 指向该环境），依赖 `qwen_asr` 包——该包已从环境中消失（pip list 无此包，import 直接 ModuleNotFoundError）
2. 模型权重仍在 `C:\Users\CJM\.cache\huggingface\hub\models--Qwen--Qwen3-ASR-0.6B`，但 repo 结构升级过：cache 快照里是 `preprocessor_config.json`，新版 qwen-asr 要 `processor_config.json`（blobs 目录为空，新版 huggingface_hub 找不到 → 联网 → huggingface.co 被墙 SSL 失败）

**修复步骤（已验证）**：
1. `E:\CosyVoiceRealtime\env\python.exe -m pip install qwen-asr -i https://pypi.tuna.tsinghua.edu.cn/simple` → qwen-asr 0.0.6
2. `HF_ENDPOINT=https://hf-mirror.com` 跑一次模型加载，补齐 processor_config.json 到 cache
3. 实测转写成功（中文文本正常输出）

## 副作用与风险（重要）

- **CosyVoiceRealtime env 被连带升级**：pip 安装 qwen-asr 时把 transformers 升到 4.57.6（另有 accelerate、tokenizers 等）。该环境同时是 AIRI/CosyVoice 语音管线用的（E:\CosyVoiceRealtime\source\cosyvoice 以 PYTHONPATH 方式引入，site-packages 里没有）。**CosyVoice 若后续报错，优先怀疑 transformers 版本不兼容，回退方案是锁定旧版 transformers**。
- 采集账号：cookie 用"无铭"号，非运营号。仍建议控制频率、避免批量连采。
- 抖音 Cookie 有效期约 30-60 天，失效后需重新在设置页粘贴。

## 运维速查

```bash
# 服务状态（四进程，start-local.cmd 启动）
curl -s http://127.0.0.1:8002/api/health
# cookie 状态 / 验证
curl -s http://127.0.0.1:8002/api/settings/douyin-cookie
# 收集博主全量视频
curl -s -X POST http://127.0.0.1:8002/api/creators/preview -H "Content-Type: application/json" -d '{"reference":"<主页链接>"}'
curl -s -X POST http://127.0.0.1:8002/api/creators/sync -H "Content-Type: application/json" -d '{"reference":"<主页链接>","confirmed":true}'
# job 列表（进度/错误）
curl -s "http://127.0.0.1:8002/api/jobs?limit=20"
```

关键文件位置：
- 日志：`.runtime\logs\`（backend / worker-general / worker-asr / ollama / frontend）
- 数据：`D:\DouyinKnowledgeLibrary\{audio,metadata,subtitle,transcript,dy-note,audit,data}`
- ASR 依赖：qwen-asr 0.0.6 @ E:\CosyVoiceRealtime\env；模型 Qwen3-ASR-0.6B @ HF cache
