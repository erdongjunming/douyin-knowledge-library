import {
  Archive,
  BookOpenText,
  CheckCircle2,
  CircleAlert,
  KeyRound,
  ListChecks,
  LoaderCircle,
  Radio,
  RefreshCw,
  Search,
  Settings,
  Trash2,
} from "lucide-react"
import { useEffect, useMemo, useState } from "react"

import "./styles.css"

type ViewKey = "creator" | "favorites" | "jobs" | "library" | "settings"

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

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  })
  const body = (await response.json()) as T & { detail?: string }
  if (!response.ok) {
    throw new Error(body.detail || `请求失败 (${response.status})`)
  }
  return body
}

function StatusMessage({ tone, children }: { tone: "success" | "error"; children: string }) {
  return (
    <p className={`status-message ${tone}`} role="status">
      {tone === "success" ? <CheckCircle2 size={16} aria-hidden="true" /> : <CircleAlert size={16} aria-hidden="true" />}
      {children}
    </p>
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
      const result = await apiJson<{ status: string; discovered_count: number }>("/api/creators/sync", {
        method: "POST",
        body: JSON.stringify({ reference, confirmed: true }),
      })
      setMessage({ tone: "success", text: `同步${result.status === "complete" ? "完成" : "未完整完成"}，发现 ${result.discovered_count} 条作品` })
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
      const result = await apiJson<{ status: string; unique_work_count: number }>("/api/favorites/sync", { method: "POST" })
      setMessage({ tone: "success", text: `收藏同步${result.status === "complete" ? "完成" : "未完整完成"}，共 ${result.unique_work_count} 条去重作品` })
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
  analyze: "整理主题",
  complete: "已完成",
}

function JobsView() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [error, setError] = useState<string | null>(null)
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
  return (
    <div className="content-grid wide-grid">
      <section className="intro-panel compact-intro"><p className="section-kicker">PROCESSING</p><h2>每一步都留下可追踪状态。</h2><p className="muted-copy">下载、抽取、转写和主题整理在同一个任务队列里串联。</p></section>
      <section className="panel table-panel" aria-labelledby="jobs-title">
        <div className="panel-heading"><div><p className="eyebrow">实时队列</p><h3 id="jobs-title">最近任务</h3></div><ListChecks size={20} aria-hidden="true" /></div>
        {error && <StatusMessage tone="error">{error}</StatusMessage>}
        {jobs.length === 0 && !error ? <p className="empty-copy">还没有处理任务。同步主播或收藏后，任务会显示在这里。</p> : (
          <div className="table-scroll"><table><thead><tr><th>任务</th><th>作品</th><th>阶段</th><th>进度</th><th>状态</th></tr></thead><tbody>
            {jobs.map((job) => <tr key={job.id}><td><strong>{job.kind}</strong><span className="cell-sub">#{job.id}</span></td><td>{job.work_id ?? "批量同步"}</td><td>{stageLabels[job.stage] || job.stage}</td><td><div className="progress-wrap"><div className="progress-track"><span style={{ width: `${Math.max(0, Math.min(100, job.progress * 100))}%` }} /></div><span>{Math.round(job.progress * 100)}%</span></div></td><td><span className={`state state-${job.status}`}>{job.status}</span></td></tr>)}
          </tbody></table></div>
        )}
      </section>
    </div>
  )
}

function LibraryView() {
  const [works, setWorks] = useState<Work[]>([])
  const [query, setQuery] = useState("")
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    void apiJson<{ items: Work[] }>("/api/works?limit=100").then((result) => setWorks(result.items)).catch((reason) => setError(reason instanceof Error ? reason.message : "无法读取资料库"))
  }, [])
  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) return works
    return works.filter((work) => `${work.description || ""} ${work.creator_nickname || ""} ${work.topics.join(" ")}`.toLowerCase().includes(normalized))
  }, [query, works])
  return (
    <div className="content-grid wide-grid">
      <section className="library-toolbar"><div><p className="section-kicker">LIBRARY</p><h2>从转写里找到知识。</h2></div><label className="search-field"><Search size={17} aria-hidden="true" /><input aria-label="筛选资料" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="筛选主播、主题或关键词" /></label></section>
      <section className="panel table-panel"><div className="panel-heading"><div><p className="eyebrow">已收集作品</p><h3>{filtered.length} 条资料</h3></div><BookOpenText size={20} aria-hidden="true" /></div>{error ? <StatusMessage tone="error">{error}</StatusMessage> : filtered.length === 0 ? <p className="empty-copy">还没有匹配的资料。先同步主播或收藏，再回来筛选。</p> : <div className="table-scroll"><table><thead><tr><th>内容</th><th>主播</th><th>主题</th><th>资产</th></tr></thead><tbody>{filtered.map((work) => <tr key={work.id}><td><strong>{work.description || "无标题作品"}</strong><span className="cell-sub">aweme {work.aweme_id}</span></td><td>{work.creator_nickname || "未知主播"}</td><td><div className="topic-list">{work.topics.slice(0, 3).map((topic) => <span key={topic}>{topic}</span>)}</div></td><td>{work.asset_kinds.length ? work.asset_kinds.join(" · ") : "待处理"}</td></tr>)}</tbody></table></div>}</section>
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
  const content = { creator: <CreatorView />, favorites: <FavoritesView />, jobs: <JobsView />, library: <LibraryView />, settings: <SettingsView /> }[activeView]
  return <div className="app-shell"><aside className="sidebar"><div className="brand-mark" aria-hidden="true">知</div><nav aria-label="主要导航">{navigation.map(({ key, label, icon: Icon }) => <button className={activeView === key ? "nav-item is-active" : "nav-item"} key={key} type="button" onClick={() => setActiveView(key)}><Icon aria-hidden="true" size={18} /><span>{label}</span></button>)}</nav></aside><main className="workspace"><header className="topbar"><div><p className="topbar-kicker">{viewTitles[activeView]}</p><h1>抖音知识资料库</h1></div><span className="connection-status"><span aria-hidden="true" className="status-dot" />本地服务已连接</span></header><div className="page-content">{content}</div></main></div>
}
