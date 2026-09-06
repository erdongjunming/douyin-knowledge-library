import { BookOpenText, FileText, List, ShieldAlert, Sparkles, X } from "lucide-react"
import { useMemo } from "react"
import ReactMarkdown from "react-markdown"
import type React from "react"

export type NoteReaderProps = {
  title: string
  markdown: string
  onClose?: () => void
  provider?: string | null
  model?: string | null
  qualityStatus?: string | null
  coverage?: { segment_total?: number | null; segment_disposition_count?: number | null; evidence_atom_total?: number | null; evidence_atom_included?: number | null } | null
  segmentCoverage?: { covered?: number | null; total?: number | null; percent?: number | null } | null
  atomCoverage?: { covered?: number | null; total?: number | null; percent?: number | null } | null
  warning?: string | null
  subtitle?: string | null
  provenance?: { variant?: string | null; sourceKind?: string | null; cleanerModel?: string | null; fallback?: string | null } | null
}

const sectionLabels = [
  { heading: "主题复盘", label: "主题复盘", icon: FileText },
  { heading: "方法与步骤", label: "方法与步骤", icon: Sparkles },
  { heading: "案例与反例", label: "案例与反例", icon: BookOpenText },
  { heading: "前提、边界与风险", label: "前提、边界与风险", icon: ShieldAlert },
  { heading: "完整逐段时间线", label: "逐段时间线", icon: List },
]

function headingId(text: string) {
  return `note-section-${text.trim().toLowerCase().replace(/\s+/g, "-").replace(/[^\w\u4e00-\u9fff-]/g, "")}`
}

export function NoteReader({ title, markdown, onClose, provider, model, qualityStatus, coverage, segmentCoverage, atomCoverage, warning, subtitle, provenance }: NoteReaderProps) {
  const availableSections = useMemo(() => sectionLabels.filter(({ heading }) => new RegExp(`^##\\s+${heading}\\s*$`, "m").test(markdown)), [markdown])
  const components = useMemo(() => ({
    h2: ({ children, ...props }: React.ComponentProps<"h2">) => {
      const text = String(children ?? "")
      return <h2 id={headingId(text)} {...props}>{children}</h2>
    },
    a: ({ children, href, ...props }: React.ComponentProps<"a">) => <a href={href} target="_blank" rel="noopener noreferrer" {...props}>{children}</a>,
  }), [])
  const scrollTo = (heading: string) => document.getElementById(headingId(heading))?.scrollIntoView({ behavior: "smooth", block: "start" })

  return <div className="note-reader">
      <div className="reader-heading">
      <div><p className="eyebrow">LEARNING NOTE</p><h3 id="note-modal-title">{title}</h3>{subtitle ? <span>{subtitle}</span> : (provider || model) && <span>{provider || "unknown"}{model ? ` / ${model}` : ""}</span>}</div>
      {onClose && <button className="icon-button" type="button" aria-label="关闭资料" title="关闭" onClick={onClose}><X size={16} aria-hidden="true" /></button>}
    </div>
    {(qualityStatus || coverage || segmentCoverage || atomCoverage || warning || provenance) && <div className="note-quality" aria-label="笔记质量信息">
      {provenance && (provenance.variant || provenance.sourceKind) && <span className="transcript-provenance">转写：{provenance.variant === "cleaned" ? "cleaned" : `${provenance.sourceKind || "asr"}/${provenance.variant || "raw"}`}{provenance.cleanerModel ? ` · ${provenance.cleanerModel}` : ""}</span>}
      {provenance?.fallback && <span className="note-warning">清洗回退：{provenance.fallback}</span>}
      {qualityStatus && <span className={`quality-badge quality-${qualityStatus}`}>质量：{qualityStatus}</span>}
      {(coverage || segmentCoverage || atomCoverage) && <span>覆盖 {segmentCoverage ? `${segmentCoverage.covered ?? "-"}/${segmentCoverage.total ?? "-"}` : `${coverage?.segment_disposition_count ?? "-"}/${coverage?.segment_total ?? "-"}`} 段 · 证据 {atomCoverage ? `${atomCoverage.covered ?? "-"}/${atomCoverage.total ?? "-"}` : `${coverage?.evidence_atom_included ?? "-"}/${coverage?.evidence_atom_total ?? "-"}`}{segmentCoverage?.percent != null ? ` · ${segmentCoverage.percent}%` : ""}</span>}
      {warning && <span className="note-warning" role="alert">{warning}</span>}
    </div>}
    {availableSections.length > 0 && <nav className="note-section-nav" aria-label="笔记章节导航">{availableSections.map(({ heading, label, icon: Icon }) => <button className="button subtle" type="button" key={heading} onClick={() => scrollTo(heading)}><Icon size={14} aria-hidden="true" />{label}</button>)}</nav>}
    <div className="note-reader-body"><article className="note-content"><ReactMarkdown components={components}>{markdown}</ReactMarkdown></article></div>
  </div>
}
