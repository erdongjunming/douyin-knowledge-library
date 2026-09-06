import {
  Archive,
  BookOpenText,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  FileText,
  Gauge,
  KeyRound,
  ListChecks,
  LoaderCircle,
  Radio,
  RefreshCw,
  Search,
  Settings,
  Square,
  Trash2,
} from "lucide-react"
import { useEffect, useState } from "react"

import "./styles.css"
import { NoteReader } from "./NoteReader"

type ViewKey = "creator" | "favorites" | "jobs" | "library" | "settings"
type ConnectionState = "connecting" | "connected" | "degraded" | "disconnected"

type CreatorPreview = {
  sec_uid: string
  nickname: string | null
  unique_id: string | null
  profile_url: string | null
  avatar_url?: string | null
}

type Job = {
  id: number
  kind: string
  work_id: number | null
  status: string
  stage: string
  progress: number
  attempts: number
  error_code: string | null
}

type Work = {
  id: number
  aweme_id: string
  description: string | null
  media_type: string
  availability: string
  asset_kinds: string[]
  creator_nickname: string | null
  published_at: string | null
  topics: string[]
  transcript_status: string | null
  transcript_preview: string | null
  transcript_variant?: string | null
  transcript_source_kind?: string | null
  cleaner_model?: string | null
  cleaning_fallback?: string | null
  note_status: string
  note_available: boolean
  note_preview: string | null
  extraction_status: string
  failure: { stage: string; reason: string } | null
  note_quality_status?: "semantic_complete" | "fallback" | "legacy" | "unknown" | "missing" | string | null
  note_provider?: string | null
  note_model?: string | null
  note_quality_gate?: string | null
  note_coverage?: { segment_total?: number | null; segment_disposition_count?: number | null; evidence_atom_total?: number | null; evidence_atom_included?: number | null } | null
  note_segment_coverage?: { covered?: number | null; total?: number | null; percent?: number | null } | null
  note_atom_coverage?: { covered?: number | null; total?: number | null; percent?: number | null } | null
}

type WorkDetail = {
  id: number
  aweme_id?: string
  description?: string | null
  note: { path: string; text: string } | null
  note_status: string
  extraction_status: string
  failure: { stage: string; reason: string } | null
  transcript: {
    engine: string
    model: string
    language: string | null
    text: string
    segments: Array<{ start?: number; end?: number; text?: string }>
    variant?: string | null
    source_kind?: string | null
    cleaner_model?: string | null
    cleaner_prompt_version?: string | null
    cleaning_fallback?: string | null
  } | null
  transcript_variant?: string | null
  transcript_source_kind?: string | null
  cleaner_model?: string | null
  cleaning_fallback?: string | null
  note_quality_status?: string | null
  note_provider?: string | null
  note_model?: string | null
  note_quality_gate?: string | null
  note_coverage?: { segment_total?: number | null; segment_disposition_count?: number | null; evidence_atom_total?: number | null; evidence_atom_included?: number | null } | null
  note_segment_coverage?: { covered?: number | null; total?: number | null; percent?: number | null } | null
  note_atom_coverage?: { covered?: number | null; total?: number | null; percent?: number | null } | null
}

type WorkListResponse = {
  items: Work[]
  total?: number
  limit?: number
  offset?: number
  has_more?: boolean
  failed_total?: number
}

type CookieStatus = {
  configured: boolean
}

type CookieSaveResult = CookieStatus & {
  valid: boolean
  account_nickname: string | null
  expires_or_unknown: string
  risk_controlled: boolean
}

type ProcessingProgress = {
  works: {
    collected: number
    audio_ready: number
  }
  audio_extraction: {
    total: number
    completed: number
    running: number
    pending: number
    failed: number
    not_queued: number
    percent: number
  }
  transcription: {
    total: number
    completed: number
    running: number
    pending: number
    failed: number
    not_queued: number
    percent: number
  }
}

type NoteProgress = {
  total: number
  completed: number
  running: number
  pending: number
  failed: number
  skipped: number
  not_queued: number
  percent: number
  recent_aweme_id: string | null
  failures: Array<{ aweme_id: string; reason: string }>
  quality_counts?: Record<string, number>
  upgrade?: {
    total?: number
    completed?: number
    remaining?: number
    active?: number
    pending?: number
    running?: number
    failed?: number
    not_queued?: number
    eligible?: number
    queued?: number
    depth_failed?: number
    input_insufficient?: number
    percent?: number
    failures?: Array<{ aweme_id: string; reason: string }>
  }
  provider_summary?: { provider?: string | null; model?: string | null; ready?: boolean; external_configured?: boolean }
}

type NoteQueueResult = {
  queued: number
  already_queued: number
  skipped: number
  eligible?: number
  selected?: number
}

type ProviderStatus = {
  ready?: boolean
  provider?: string | null
  model?: string | null
  external_configured?: boolean
  ollama?: { ready?: boolean; model?: string | null; service_status?: string; model_available?: boolean; action_required?: string | null }
  external?: { ready?: boolean; configured?: boolean; model?: string | null }
}

const navigation: Array<{ key: ViewKey; label: string; icon: typeof Radio }> = [
  { key: "creator", label: "主播", icon: Radio },
  { key: "favorites", label: "我的收藏", icon: Archive },
  { key: "jobs", label: "任务中心", icon: ListChecks },
  { key: "library", label: "资料库", icon: BookOpenText },
  { key: "settings", label: "设置", icon: Settings },
]

const viewTitles: Record<ViewKey, string> = {
  creator: "主播资料",
  favorites: "收藏同步",
  jobs: "任务中心",
  library: "知识资料库",
  settings: "运行设置",
}

const apiErrorMessages: Record<string, string> = {
  douyin_cookie_store_unavailable: "Windows 凭据库当前不可用，请在普通用户会话中启动后端后重试。",
}

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  })
  const responseText = await response.text()
  let body: unknown = null
  if (responseText.trim()) {
    try {
      body = JSON.parse(responseText)
    } catch {
      if (response.ok) {
        throw new Error("本地服务返回了无法解析的数据")
      }
    }
  }
  if (!response.ok) {
    const detail = body && typeof body === "object" && "detail" in body ? body.detail : null
    const message = typeof detail === "string" ? apiErrorMessages[detail] || detail : null
    throw new Error(message || `本地服务请求失败 (${response.status})`)
  }
  return body as T
}

function StatusMessage({ tone, children }: { tone: "success" | "error"; children: string }) {
  return (
    <p className={`status-message ${tone}`} role="status">
      {tone === "success" ? <CheckCircle2 size={16} aria-hidden="true" /> : <CircleAlert size={16} aria-hidden="true" />}
      {children}
    </p>
  )
}

function GlobalProgress({
  progress,
  notes,
  noteBusy,
  noteMessage,
  onGenerateNotes,
  onUpgradeNotes,
  providerReady,
  providerStatus,
}: {
  progress: ProcessingProgress | null
  notes: NoteProgress | null
  noteBusy: boolean
  noteMessage: { tone: "success" | "error"; text: string } | null
  onGenerateNotes: () => void
  onUpgradeNotes: () => void
  providerReady: boolean
  providerStatus: ProviderStatus | null
}) {
  if (!progress) {
    return (
      <section className="progress-dock is-loading" aria-label="处理总进度">
        <Gauge size={18} aria-hidden="true" />
        <span>正在读取处理进度</span>
      </section>
    )
  }

  const { works, audio_extraction: extraction, transcription } = progress
  const audioPercent = Math.max(0, Math.min(100, extraction?.percent ?? (works.collected ? works.audio_ready / works.collected * 100 : 0)))
  const transcriptionPercent = Math.max(0, Math.min(100, transcription.percent))
  const upgrade = notes?.upgrade
  const upgradeCandidates = (notes?.quality_counts?.legacy ?? 0) + (notes?.quality_counts?.fallback ?? 0) + (notes?.quality_counts?.unknown ?? 0)
  const upgradeRemaining = notes?.upgrade?.remaining ?? notes?.upgrade?.eligible ?? upgradeCandidates
  const upgradeTotal = notes?.upgrade?.total ?? notes?.upgrade?.eligible ?? upgradeCandidates
  const upgradeCompleted = notes?.upgrade?.completed ?? Math.max(0, upgradeTotal - upgradeRemaining)
  const upgradePercent = notes?.upgrade?.percent ?? (upgradeTotal ? upgradeCompleted / upgradeTotal * 100 : 0)
  return (
    <section className="progress-dock" aria-label="处理总进度">
      <div className="progress-lane">
        <div className="progress-summary">
          <div className="progress-title"><Gauge size={18} aria-hidden="true" /><div><strong>音频提取</strong><span>已收集 {works.collected} 条视频</span></div></div>
          <strong className="progress-count">{extraction?.completed ?? works.audio_ready} / {extraction?.total ?? works.collected}</strong>
        </div>
        <div className="global-progress-track" role="progressbar" aria-label="音频提取进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={audioPercent} aria-valuetext={`${extraction?.completed ?? works.audio_ready} / ${extraction?.total ?? works.collected}，${audioPercent}%`}><span style={{ width: `${audioPercent}%` }} /></div>
        <div className="progress-meta"><strong>{audioPercent}%</strong><span className="progress-running">处理中 {extraction?.running ?? 0}</span><span>待处理 {(extraction?.pending ?? 0) + (extraction?.not_queued ?? 0)}</span><span className={extraction?.failed ? "progress-failed" : undefined}>失败 {extraction?.failed ?? 0}</span></div>
      </div>
      <div className="progress-lane">
        <div className="progress-summary"><div className="progress-title"><Gauge size={18} aria-hidden="true" /><div><strong>音频转写</strong><span>仅对已提取音频进行转写</span></div></div><strong className="progress-count">{transcription.completed} / {transcription.total}</strong></div>
        <div className="global-progress-track" role="progressbar" aria-label="音频转写进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={transcriptionPercent} aria-valuetext={`${transcription.completed} / ${transcription.total}，${transcriptionPercent}%`}><span style={{ width: `${transcriptionPercent}%` }} /></div>
        <div className="progress-meta"><strong>{transcriptionPercent}%</strong><span className="progress-running">处理中 {transcription.running}</span><span>待处理 {transcription.pending + transcription.not_queued}</span><span className={transcription.failed ? "progress-failed" : undefined}>失败 {transcription.failed}</span></div>
      </div>
      {notes && <div className="progress-lane note-progress-lane">
        <div className="progress-summary">
          <div className="progress-title"><FileText size={18} aria-hidden="true" /><div><strong>知识笔记</strong><span>{notes.failures[0] ? `失败 ${notes.failures[0].aweme_id}：${notes.failures[0].reason}` : notes.recent_aweme_id ? `最近处理 ${notes.recent_aweme_id}` : "基于现有 ASR 转写增量生成"}</span></div></div>
          <strong className="progress-count">{notes.completed} / {notes.total}</strong>
        </div>
        <div className="global-progress-track" role="progressbar" aria-label="知识笔记生成进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={notes.percent} aria-valuetext={`${notes.completed} / ${notes.total}，${notes.percent}%`}><span style={{ width: `${Math.max(0, Math.min(100, notes.percent))}%` }} /></div>
        <div className="note-progress-actions">
          <div className="progress-meta"><strong>{notes.percent}%</strong><span className="progress-running">处理中 {notes.running}</span><span>待处理 {notes.pending + notes.not_queued}</span><span className={notes.failed ? "progress-failed" : undefined}>失败 {notes.failed}</span><span>跳过 {notes.skipped}</span>{notes.quality_counts && <span>语义 {notes.quality_counts.semantic_complete ?? 0} · 回退 {notes.quality_counts.fallback ?? 0} · 旧 {notes.quality_counts.legacy ?? 0} · 未知 {notes.quality_counts.unknown ?? 0}</span>}{providerStatus?.ollama && <span className={providerStatus.ollama.ready ? "progress-running" : "progress-failed"}>Ollama：{providerStatus.ollama.ready ? `已就绪${providerStatus.ollama.model ? ` · ${providerStatus.ollama.model}` : ""}` : providerStatus.ollama.model_available === false ? `模型未安装${providerStatus.ollama.model ? ` · ${providerStatus.ollama.model}` : ""}` : "未就绪"}</span>}{(providerStatus?.external?.configured || notes.provider_summary?.external_configured) && <span>外部提供方已配置</span>}</div>
          <div className="note-action-buttons"><button className="button primary note-generate-button" type="button" disabled={noteBusy || notes.not_queued === 0} onClick={onGenerateNotes}>
            {noteBusy ? <LoaderCircle className="spin" size={15} aria-hidden="true" /> : <FileText size={15} aria-hidden="true" />}
            {noteBusy ? "正在入队" : notes.not_queued === 0 ? "笔记已齐全" : "生成缺失笔记"}
          </button>{upgradeRemaining > 0 && <button className="button subtle note-generate-button" type="button" disabled={noteBusy || !providerReady} onClick={onUpgradeNotes}><RefreshCw size={15} aria-hidden="true" />升级旧笔记</button>}</div>
        </div>
        {noteMessage && <p className={`progress-action-message ${noteMessage.tone}`} role="status">{noteMessage.text}</p>}
        {upgradeTotal > 0 && <div className="upgrade-progress-lane" aria-label="旧笔记语义升级">
          <div className="progress-summary"><div className="progress-title"><RefreshCw size={18} aria-hidden="true" /><div><strong>旧笔记语义升级</strong><span>{upgrade?.failures?.[0] ? `失败 ${upgrade.failures[0].aweme_id}：${upgrade.failures[0].reason}` : "仅接受本地 Ollama 的完整语义笔记"}</span></div></div><strong className="progress-count">{upgradeCompleted} / {upgradeTotal}</strong></div>
          <div className="global-progress-track" role="progressbar" aria-label="旧笔记语义升级进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.max(0, Math.min(100, upgradePercent))} aria-valuetext={`${upgradeCompleted} / ${upgradeTotal}，${upgradePercent}%`}><span style={{ width: `${Math.max(0, Math.min(100, upgradePercent))}%` }} /></div>
          <div className="progress-meta"><strong>{upgradePercent}%</strong><span className="progress-running">处理中 {upgrade?.active ?? ((upgrade?.pending ?? 0) + (upgrade?.running ?? 0))}</span><span>剩余 {upgradeRemaining}</span><span className={upgrade?.failed ? "progress-failed" : undefined}>失败 {upgrade?.failed ?? 0}</span>{(upgrade?.depth_failed || upgrade?.input_insufficient) ? <span>深度失败 {upgrade.depth_failed ?? 0} · 输入不足 {upgrade.input_insufficient ?? 0}</span> : null}</div>
        </div>}
      </div>}
    </section>
  )
}

function CreatorView() {
  const [reference, setReference] = useState("")
  const [preview, setPreview] = useState<CreatorPreview | null>(null)
  const [busy, setBusy] = useState<"preview" | "sync" | null>(null)
  const [message, setMessage] = useState<{ tone: "success" | "error"; text: string } | null>(null)

  const previewCreator = async () => {
    setBusy("preview")
    setMessage(null)
    try {
      const result = await apiJson<CreatorPreview>("/api/creators/preview", {
        method: "POST",
        body: JSON.stringify({ reference: reference.trim() }),
      })
      setPreview(result)
    } catch (error) {
      setPreview(null)
      setMessage({ tone: "error", text: error instanceof Error ? error.message : "预览失败" })
    } finally {
      setBusy(null)
    }
  }

  const syncCreator = async () => {
    if (!preview) return
    setBusy("sync")
    setMessage(null)
    try {
      const result = await apiJson<Job>("/api/creators/sync", {
        method: "POST",
        body: JSON.stringify({ reference, confirmed: true }),
      })
      if (result.status !== "pending") throw new Error("同步任务未能加入队列")
      setMessage({ tone: "success", text: "主播同步任务已加入队列，可前往任务中心查看进度。" })
    } catch (error) {
      setMessage({ tone: "error", text: error instanceof Error ? error.message : "同步失败" })
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="content-grid creator-grid">
      <section className="intro-panel">
        <p className="section-kicker">COLLECTOR</p>
        <h2>把一个主播的知识内容收进资料库。</h2>
        <p className="muted-copy">输入公开主页链接或 sec_uid。预览身份后，再确认同步作品，避免误抓错误账号。</p>
      </section>
      <section className="panel form-panel" aria-labelledby="creator-form-title">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">新增来源</p>
            <h3 id="creator-form-title">主播主页</h3>
          </div>
          <Radio size={20} aria-hidden="true" />
        </div>
        <label htmlFor="creator-reference">主播主页或 sec_uid</label>
        <div className="input-row">
          <input
            id="creator-reference"
            aria-label="主播主页或 sec_uid"
            value={reference}
            onChange={(event) => setReference(event.target.value)}
            placeholder="https://www.douyin.com/user/..."
          />
          <button className="button primary" type="button" disabled={!reference.trim() || busy !== null} onClick={previewCreator}>
            {busy === "preview" ? <LoaderCircle className="spin" size={16} aria-hidden="true" /> : <Search size={16} aria-hidden="true" />}
            预览主播
          </button>
        </div>
        {preview && (
          <div className="preview-result">
            <div className="avatar-placeholder" aria-hidden="true">{(preview.nickname || "知").slice(0, 1)}</div>
            <div>
              <strong>{preview.nickname || "未命名主播"}</strong>
              <span>{preview.unique_id ? `抖音号 ${preview.unique_id}` : preview.sec_uid}</span>
            </div>
            <button className="button dark" type="button" disabled={busy !== null} onClick={syncCreator}>
              {busy === "sync" ? <LoaderCircle className="spin" size={16} aria-hidden="true" /> : <RefreshCw size={16} aria-hidden="true" />}
              确认并同步主播
            </button>
          </div>
        )}
        {message && <StatusMessage tone={message.tone}>{message.text}</StatusMessage>}
      </section>
      <section className="signal-band" aria-label="处理规则">
        <div><strong>增量同步</strong><span>按 aweme_id 去重，仅拉取新作品</span></div>
        <div><strong>原始视频临时保存</strong><span>音频验证后自动清理视频文件</span></div>
        <div><strong>数据落在 D 盘</strong><span>D:\DouyinKnowledgeLibrary</span></div>
      </section>
    </div>
  )
}

function FavoritesView() {
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<{ tone: "success" | "error"; text: string } | null>(null)
  const syncFavorites = async () => {
    setBusy(true)
    setMessage(null)
    try {
      const result = await apiJson<Job>("/api/favorites/sync", { method: "POST" })
      if (result.status !== "pending") throw new Error("收藏同步任务未能加入队列")
      setMessage({ tone: "success", text: "收藏同步任务已加入队列，可前往任务中心查看进度。" })
    } catch (error) {
      setMessage({ tone: "error", text: error instanceof Error ? error.message : "收藏同步失败" })
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="content-grid">
      <section className="intro-panel">
        <p className="section-kicker">SAVED CONTENT</p>
        <h2>同步账号里的全部收藏。</h2>
        <p className="muted-copy">默认收藏和自定义收藏夹会一起扫描。同步过程支持不完整运行保护，不会用半份结果覆盖上一次完整状态。</p>
      </section>
      <section className="panel action-panel">
        <div className="panel-heading"><div><p className="eyebrow">账号收藏</p><h3>全部收藏视频</h3></div><Archive size={20} aria-hidden="true" /></div>
        <p className="muted-copy">首次同步可能需要较长时间，完成后可在任务中心查看后续音频处理进度。</p>
        <button className="button primary" type="button" disabled={busy} onClick={syncFavorites}>
          {busy ? <LoaderCircle className="spin" size={16} aria-hidden="true" /> : <RefreshCw size={16} aria-hidden="true" />}
          {busy ? "同步中" : "同步全部收藏"}
        </button>
        {message && <StatusMessage tone={message.tone}>{message.text}</StatusMessage>}
      </section>
    </div>
  )
}

const stageLabels: Record<string, string> = {
  discover: "发现作品",
  download: "下载视频",
  extract: "提取音频",
  transcribe: "转写文本",
  analyze: "生成知识笔记",
  complete: "已完成",
}

function JobsView() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [error, setError] = useState<string | null>(null)
  const [cancelBusy, setCancelBusy] = useState<number | null>(null)
  useEffect(() => {
    let active = true
    const load = async () => {
      try {
        const result = await apiJson<{ items: Job[] }>("/api/jobs")
        if (active) { setJobs(result.items); setError(null) }
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : "无法读取任务")
      }
    }
    void load()
    const timer = window.setInterval(() => void load(), 5000)
    return () => { active = false; window.clearInterval(timer) }
  }, [])
  const cancelJob = async (jobId: number) => {
    setCancelBusy(jobId)
    try {
      const cancelled = await apiJson<Job>(`/api/jobs/${jobId}/cancel`, { method: "POST" })
      setJobs((current) => current.map((job) => job.id === jobId ? cancelled : job))
      setError(null)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法取消任务")
    } finally {
      setCancelBusy(null)
    }
  }
  return (
    <div className="content-grid wide-grid">
      <section className="intro-panel compact-intro"><p className="section-kicker">PROCESSING</p><h2>每一步都留下可追踪状态。</h2><p className="muted-copy">下载、抽取、转写和知识笔记生成在同一个任务队列里串联。</p></section>
      <section className="panel table-panel" aria-labelledby="jobs-title">
        <div className="panel-heading"><div><p className="eyebrow">实时队列</p><h3 id="jobs-title">最近任务</h3></div><ListChecks size={20} aria-hidden="true" /></div>
        {error && <StatusMessage tone="error">{error}</StatusMessage>}
        {jobs.length === 0 && !error ? <p className="empty-copy">还没有处理任务。同步主播或收藏后，任务会显示在这里。</p> : (
          <div className="table-scroll"><table><thead><tr><th>任务</th><th>作品</th><th>阶段</th><th>进度</th><th>状态</th><th aria-label="操作" /></tr></thead><tbody>
            {jobs.map((job) => <tr key={job.id}><td><strong>{job.kind}</strong><span className="cell-sub">#{job.id}</span></td><td>{job.work_id ?? "批量同步"}</td><td>{stageLabels[job.stage] || job.stage}</td><td><div className="progress-wrap"><div className="progress-track"><span style={{ width: `${Math.max(0, Math.min(100, job.progress * 100))}%` }} /></div><span>{Math.round(job.progress * 100)}%</span></div></td><td><span className={`state state-${job.status}`}>{job.status}</span></td><td>{job.status === "running" || job.status === "pending" ? <button className="icon-button danger-button" type="button" aria-label={`取消任务 ${job.id}`} title="取消任务" disabled={cancelBusy === job.id} onClick={() => void cancelJob(job.id)}>{cancelBusy === job.id ? <LoaderCircle className="spin" size={15} aria-hidden="true" /> : <Square size={14} aria-hidden="true" />}</button> : null}</td></tr>)}
          </tbody></table></div>
        )}
      </section>
    </div>
  )
}

function LibraryView() {
  const [works, setWorks] = useState<Work[]>([])
  const [query, setQuery] = useState("")
  const [page, setPage] = useState(0)
  const [total, setTotal] = useState(0)
  const [failedTotal, setFailedTotal] = useState(0)
  const [failedOnly, setFailedOnly] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<WorkDetail | null>(null)
  const [detailBusy, setDetailBusy] = useState(false)
  const [generateBusy, setGenerateBusy] = useState<number | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    let active = true
    const params = new URLSearchParams({ limit: "100" })
    if (page > 0) params.set("offset", String(page * 100))
    if (query.trim()) params.set("query", query.trim())
    if (failedOnly) params.set("failed_only", "true")
    void apiJson<WorkListResponse>(`/api/works?${params.toString()}`, { signal: controller.signal })
      .then((result) => {
        if (!active) return
        setWorks(result.items)
        setTotal(result.total ?? result.items.length)
        setFailedTotal(result.failed_total ?? 0)
        setError(null)
      })
      .catch((reason) => {
        if (active) setError(reason instanceof Error ? reason.message : "无法读取资料库")
      })
    return () => {
      active = false
      controller.abort()
    }
  }, [failedOnly, page, query])
  const openWork = async (work: Work) => {
    setDetailBusy(true)
    setError(null)
    try {
      setSelected(await apiJson<WorkDetail>(`/api/works/${work.id}`))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法读取资料")
    } finally {
      setDetailBusy(false)
    }
  }
  const generateWorkNote = async (work: Work) => {
    setGenerateBusy(work.id)
    try {
      await apiJson<NoteQueueResult>("/api/notes/generate", { method: "POST", body: JSON.stringify({ aweme_ids: [work.aweme_id], provider_policy: "auto", force: false }) })
      setError(null)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法创建笔记任务")
    } finally {
      setGenerateBusy(null)
    }
  }
  useEffect(() => {
    if (!selected) return undefined
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelected(null)
    }
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = "hidden"
    window.addEventListener("keydown", closeOnEscape)
    return () => {
      document.body.style.overflow = previousOverflow
      window.removeEventListener("keydown", closeOnEscape)
    }
  }, [selected])
  const pageCount = Math.max(1, Math.ceil(total / 100))
  const selectedNote = selected?.note?.text || null
  const selectedTranscript = selected?.transcript?.text || null
  const selectedContent = selectedNote || selectedTranscript
  const selectedIsNote = Boolean(selectedNote)
  const stageLabels: Record<string, string> = {
    audio_extraction: "音频提取",
    transcription: "ASR 转写",
    note_generation: "笔记生成",
  }
  const statusLabels: Record<string, string> = {
    succeeded: "已完成",
    failed: "失败",
    running: "处理中",
    pending: "排队中",
    not_queued: "未排队",
  }
  return (
    <div className="content-grid wide-grid">
      <section className="library-toolbar"><div><p className="section-kicker">LIBRARY</p><h2>从转写里找到知识。</h2><p className="library-count">{total} 条资料 · 第 {page + 1} / {pageCount} 页</p></div><div className="library-controls"><label className="search-field"><Search size={17} aria-hidden="true" /><input aria-label="筛选资料" value={query} onChange={(event) => { setQuery(event.target.value); setPage(0) }} placeholder="筛选主播或关键词" /></label><button className={failedOnly ? "button subtle is-active" : "button subtle"} type="button" onClick={() => { setFailedOnly((current) => !current); setPage(0) }}><CircleAlert size={15} aria-hidden="true" />仅看提取失败 ({failedTotal})</button></div></section>
      <section className="panel table-panel"><div className="panel-heading"><div><p className="eyebrow">已收集作品</p><h3>{works.length} 条当前页资料</h3></div><BookOpenText size={20} aria-hidden="true" /></div>{error ? <StatusMessage tone="error">{error}</StatusMessage> : works.length === 0 ? <p className="empty-copy">还没有匹配的资料。先同步主播或收藏，再回来筛选。</p> : <div className="table-scroll"><table><thead><tr><th>内容</th><th>主播</th><th>主题</th><th>处理状态</th><th aria-label="操作" /></tr></thead><tbody>{works.map((work) => <tr key={work.id}><td><strong>{work.description || "无标题作品"}</strong><span className="cell-sub">{work.note_preview || work.transcript_preview || `aweme ${work.aweme_id}`}</span>{work.failure && <span className="failure-reason">{stageLabels[work.failure.stage] || work.failure.stage}：{work.failure.reason}</span>}</td><td>{work.creator_nickname || "未知主播"}</td><td><div className="topic-list">{work.topics.slice(0, 3).map((topic) => <span key={topic}>{topic}</span>)}</div></td><td><span className={`state state-${work.extraction_status}`}>{statusLabels[work.extraction_status] || work.extraction_status}</span><span className="cell-sub">笔记：{statusLabels[work.note_status] || work.note_status}</span></td><td className="library-actions">{work.note_available ? <button className="icon-button" type="button" aria-label={`阅读${work.description || "无标题作品"}的学习笔记`} title="阅读学习笔记" disabled={detailBusy} onClick={() => void openWork(work)}><FileText size={16} aria-hidden="true" /></button> : work.transcript_status === "succeeded" ? <button className="icon-button" type="button" aria-label={`阅读${work.description || "无标题作品"}的转写`} title="阅读转写" disabled={detailBusy} onClick={() => void openWork(work)}><BookOpenText size={16} aria-hidden="true" /></button> : <span className="cell-sub">无可读内容</span>}{work.transcript_status === "succeeded" && <button className="button subtle single-note-button" type="button" aria-label={`生成${work.description || "无标题作品"}的完整笔记`} disabled={generateBusy === work.id} onClick={() => void generateWorkNote(work)}>{generateBusy === work.id ? <LoaderCircle className="spin" size={14} aria-hidden="true" /> : <FileText size={14} aria-hidden="true" />}生成完整笔记</button>}</td></tr>)}</tbody></table></div>}<div className="pagination" aria-label="资料分页"><button className="icon-button" type="button" aria-label="上一页" title="上一页" disabled={page === 0} onClick={() => setPage((current) => Math.max(0, current - 1))}><ChevronLeft size={17} aria-hidden="true" /></button><span>第 {page + 1} / {pageCount} 页</span><button className="icon-button" type="button" aria-label="下一页" title="下一页" disabled={page + 1 >= pageCount} onClick={() => setPage((current) => current + 1)}><ChevronRight size={17} aria-hidden="true" /></button></div></section>
      {selected && <div className="note-modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setSelected(null) }}><section className="note-modal" role="dialog" aria-modal="true" aria-labelledby="note-modal-title"><NoteReader title={selected.description || (selectedIsNote ? "学习笔记" : "作品转写")} markdown={selectedContent || ""} onClose={() => setSelected(null)} subtitle={selectedIsNote ? "learning_note.md · 基于 ASR 转写生成" : selected.transcript?.model || "尚无转写"} provenance={selectedIsNote ? { variant: selected.transcript_variant, sourceKind: selected.transcript_source_kind, cleanerModel: selected.cleaner_model, fallback: selected.cleaning_fallback } : selected.transcript ? { variant: selected.transcript.variant, sourceKind: selected.transcript.source_kind, cleanerModel: selected.transcript.cleaner_model, fallback: selected.transcript.cleaning_fallback } : null} provider={selected.note_provider} model={selected.note_model} qualityStatus={selected.note_quality_status} coverage={selected.note_coverage} segmentCoverage={selected.note_segment_coverage} atomCoverage={selected.note_atom_coverage} warning={selected.note_quality_status === "fallback" ? "这是规则回退笔记，未通过语义完整性审查。" : selected.failure ? `处理失败：${stageLabels[selected.failure.stage] || selected.failure.stage}：${selected.failure.reason}` : !selectedContent ? "这条作品尚无可读内容。" : null} /></section></div>}
    </div>
  )
}

function SettingsView() {
  const [diagnostics, setDiagnostics] = useState<Record<string, { status: string; path?: string }> | null>(null)
  const [cookie, setCookie] = useState("")
  const [cookieConfigured, setCookieConfigured] = useState<boolean | null>(null)
  const [cookieBusy, setCookieBusy] = useState(false)
  const [cookieMessage, setCookieMessage] = useState<{ tone: "success" | "error"; text: string } | null>(null)

  useEffect(() => {
    void apiJson<Record<string, { status: string; path?: string }>>("/api/settings/diagnostics")
      .then(setDiagnostics)
      .catch(() => setDiagnostics(null))
    void apiJson<CookieStatus>("/api/settings/douyin-cookie")
      .then((result) => setCookieConfigured(result.configured))
      .catch(() => setCookieConfigured(null))
  }, [])

  const saveCookie = async () => {
    setCookieBusy(true)
    setCookieMessage(null)
    try {
      const result = await apiJson<CookieSaveResult>("/api/settings/douyin-cookie", {
        method: "PUT",
        body: JSON.stringify({ cookie }),
      })
      setCookie("")
      setCookieConfigured(result.configured)
      setCookieMessage({
        tone: "success",
        text: result.account_nickname ? `抖音账号已连接：${result.account_nickname}` : "抖音账号已连接",
      })
    } catch (error) {
      setCookieMessage({ tone: "error", text: error instanceof Error ? error.message : "Cookie 保存失败" })
    } finally {
      setCookieBusy(false)
    }
  }

  const removeCookie = async () => {
    setCookieBusy(true)
    setCookieMessage(null)
    try {
      const result = await apiJson<CookieStatus>("/api/settings/douyin-cookie", { method: "DELETE" })
      setCookie("")
      setCookieConfigured(result.configured)
      setCookieMessage({ tone: "success", text: "抖音凭据已移除" })
    } catch (error) {
      setCookieMessage({ tone: "error", text: error instanceof Error ? error.message : "凭据移除失败" })
    } finally {
      setCookieBusy(false)
    }
  }

  return (
    <div className="content-grid settings-grid">
      <section className="intro-panel">
        <p className="section-kicker">SYSTEM</p>
        <h2>本地优先，数据留在你的机器上。</h2>
        <p className="muted-copy">Cookie 通过本地凭据存储读取；运行数据默认写入 D 盘，不会静默回退到 C 盘。</p>
      </section>
      <div className="settings-stack">
        <section className="panel credential-panel" aria-labelledby="cookie-title">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">抖音授权</p>
              <h3 id="cookie-title">账号凭据</h3>
            </div>
            <KeyRound size={20} aria-hidden="true" />
          </div>
          <div className="credential-status-row">
            <p className="credential-status">
              <span className={`status-dot ${cookieConfigured ? "is-ok" : "is-idle"}`} aria-hidden="true" />
              {cookieConfigured === null ? "凭据状态未知" : cookieConfigured ? "已配置抖音 Cookie" : "尚未配置抖音 Cookie"}
            </p>
            {cookieConfigured ? (
              <button className="button subtle" type="button" disabled={cookieBusy} onClick={removeCookie}>
                <Trash2 size={15} aria-hidden="true" />
                移除凭据
              </button>
            ) : null}
          </div>
          <label htmlFor="douyin-cookie">抖音 Cookie</label>
          <div className="input-row">
            <input
              id="douyin-cookie"
              type="password"
              autoComplete="new-password"
              value={cookie}
              onChange={(event) => setCookie(event.target.value)}
              placeholder="粘贴当前登录会话的 Cookie"
            />
            <button className="button primary" type="button" disabled={!cookie.trim() || cookieBusy} onClick={saveCookie}>
              {cookieBusy ? <LoaderCircle className="spin" size={16} aria-hidden="true" /> : <CheckCircle2 size={16} aria-hidden="true" />}
              {cookieBusy ? "验证中" : "验证并保存"}
            </button>
          </div>
          {cookieMessage && <StatusMessage tone={cookieMessage.tone}>{cookieMessage.text}</StatusMessage>}
        </section>
        <section className="panel diagnostics-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">运行诊断</p><h3>处理环境</h3></div>
            <Settings size={20} aria-hidden="true" />
          </div>
          {diagnostics ? Object.entries(diagnostics).map(([name, item]) => <div className="diagnostic-row" key={name}><span>{name}</span><strong className={item.status === "ok" ? "ok" : "warning"}>{item.status === "ok" ? "正常" : "未就绪"}</strong>{item.path && <small>{item.path}</small>}</div>) : <p className="empty-copy">诊断信息暂不可用。</p>}
        </section>
      </div>
    </div>
  )
}

export function App() {
  const [activeView, setActiveView] = useState<ViewKey>("creator")
  const [connectionState, setConnectionState] = useState<ConnectionState>("connecting")
  const [processingProgress, setProcessingProgress] = useState<ProcessingProgress | null>(null)
  const [noteProgress, setNoteProgress] = useState<NoteProgress | null>(null)
  const [noteBusy, setNoteBusy] = useState(false)
  const [noteMessage, setNoteMessage] = useState<{ tone: "success" | "error"; text: string } | null>(null)
  const [providerStatus, setProviderStatus] = useState<ProviderStatus | null>(null)

  useEffect(() => {
    let active = true
    let timer: number | undefined
    const checkHealth = async () => {
      try {
        const result = await apiJson<{ status: string }>("/api/health")
        if (active) {
          setConnectionState(result.status === "ok" ? "connected" : result.status === "degraded" ? "degraded" : "disconnected")
        }
      } catch {
        if (active) setConnectionState("disconnected")
      } finally {
        if (active) timer = window.setTimeout(() => void checkHealth(), 5000)
      }
    }
    void checkHealth()
    return () => {
      active = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  useEffect(() => {
    void apiJson<ProviderStatus>("/api/notes/provider-status").then(setProviderStatus).catch(() => setProviderStatus(null))
  }, [])

  useEffect(() => {
    let active = true
    let timer: number | undefined
    const loadNoteProgress = async () => {
      try {
        const result = await apiJson<NoteProgress>("/api/notes/progress")
        if (active && typeof result?.total === "number") setNoteProgress(result)
      } catch {
        // Keep the latest valid snapshot visible during a brief service interruption.
      } finally {
        if (active) timer = window.setTimeout(() => void loadNoteProgress(), 5000)
      }
    }
    void loadNoteProgress()
    return () => {
      active = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  const generateMissingNotes = async () => {
    setNoteBusy(true)
    setNoteMessage(null)
    try {
      const result = await apiJson<NoteQueueResult>("/api/notes/generate", {
        method: "POST",
        body: JSON.stringify({ force: false }),
      })
      setNoteMessage({
        tone: "success",
        text: result.queued > 0
          ? `已加入 ${result.queued} 条笔记任务${result.already_queued > 0 ? `，已有 ${result.already_queued} 条在队列中` : ""}。`
          : result.already_queued > 0
            ? `缺失笔记已有 ${result.already_queued} 条在队列中。`
            : "当前没有缺失笔记。",
      })
      setNoteProgress((current) => current ? {
        ...current,
        pending: current.pending + result.queued,
        not_queued: Math.max(0, current.not_queued - result.queued),
      } : current)
    } catch (reason) {
      setNoteMessage({ tone: "error", text: reason instanceof Error ? reason.message : "无法创建笔记任务" })
    } finally {
      setNoteBusy(false)
    }
  }

  const upgradeLegacyNotes = async () => {
    setNoteBusy(true)
    setNoteMessage(null)
    try {
      const result = await apiJson<NoteQueueResult>("/api/notes/generate", { method: "POST", body: JSON.stringify({ legacy_only: true, limit: 3, provider_policy: "local" }) })
      setNoteMessage({
        tone: "success",
        text: result.queued > 0
          ? `新加入 ${result.queued} 条${result.already_queued > 0 ? `，已有 ${result.already_queued} 条在队列中` : ""}。`
          : result.already_queued > 0
            ? `已有 ${result.already_queued} 条在队列中。`
            : "当前没有可升级的旧笔记。",
      })
      setNoteProgress((current) => {
        if (!current) return current
        const currentUpgrade = current.upgrade
        if (!currentUpgrade) return current
        const currentTotal = currentUpgrade.total ?? currentUpgrade.eligible ?? result.eligible ?? 0
        const currentCompleted = currentUpgrade.completed ?? 0
        const currentActive = currentUpgrade.active ?? ((currentUpgrade.pending ?? 0) + (currentUpgrade.running ?? 0))
        const currentRemaining = currentUpgrade.remaining ?? Math.max(0, currentTotal - currentCompleted)
        const nextActive = currentActive + result.queued
        return {
          ...current,
          upgrade: {
            ...currentUpgrade,
            total: currentTotal,
            eligible: currentUpgrade.eligible ?? result.eligible,
            queued: (currentUpgrade.queued ?? 0) + result.queued,
            pending: (currentUpgrade.pending ?? 0) + result.queued,
            active: nextActive,
            remaining: currentRemaining,
            percent: currentUpgrade.percent ?? (currentTotal ? currentCompleted / currentTotal * 100 : 0),
          },
        }
      })
    } catch (reason) {
      setNoteMessage({ tone: "error", text: reason instanceof Error ? reason.message : "无法升级旧笔记" })
    } finally {
      setNoteBusy(false)
    }
  }

  useEffect(() => {
    let active = true
    let timer: number | undefined
    const loadProgress = async () => {
      try {
        const result = await apiJson<ProcessingProgress>("/api/jobs/progress")
        if (active && result?.works && result?.transcription) setProcessingProgress(result)
      } catch {
        // Keep the latest valid snapshot visible during a brief service interruption.
      } finally {
        if (active) timer = window.setTimeout(() => void loadProgress(), 5000)
      }
    }
    void loadProgress()
    return () => {
      active = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  const content = { creator: <CreatorView />, favorites: <FavoritesView />, jobs: <JobsView />, library: <LibraryView />, settings: <SettingsView /> }[activeView]
  const connectionLabel = {
    connecting: "正在连接本地服务",
    connected: "本地服务已连接",
    degraded: "本地服务已连接（存储暂不可用）",
    disconnected: "本地服务未连接",
  }[connectionState]
  const semanticProviderReady = Boolean(providerStatus?.ollama?.ready || (noteProgress?.provider_summary?.provider === "ollama" && noteProgress.provider_summary.ready))
  return <div className="app-shell"><aside className="sidebar"><div className="brand-mark" aria-hidden="true">知</div><nav aria-label="主要导航">{navigation.map(({ key, label, icon: Icon }) => <button className={activeView === key ? "nav-item is-active" : "nav-item"} key={key} type="button" onClick={() => setActiveView(key)}><Icon aria-hidden="true" size={18} /><span>{label}</span></button>)}</nav></aside><main className="workspace"><header className="topbar"><div><p className="topbar-kicker">{viewTitles[activeView]}</p><h1>抖音知识资料库</h1></div><span className={`connection-status is-${connectionState}`} aria-live="polite"><span aria-hidden="true" className="status-dot" />{connectionLabel}</span></header><GlobalProgress progress={processingProgress} notes={noteProgress} noteBusy={noteBusy} noteMessage={noteMessage} onGenerateNotes={() => void generateMissingNotes()} onUpgradeNotes={() => void upgradeLegacyNotes()} providerReady={semanticProviderReady} providerStatus={providerStatus} /><div className="page-content">{content}</div></main></div>
}
