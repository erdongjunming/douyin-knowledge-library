import { Archive, BookOpenText, ListChecks, Radio, Settings } from "lucide-react"

import "./styles.css"

const navigation = [
  { label: "主播", icon: Radio },
  { label: "我的收藏", icon: Archive },
  { label: "任务中心", icon: ListChecks },
  { label: "资料库", icon: BookOpenText },
  { label: "设置", icon: Settings },
]

export function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark" aria-hidden="true">
          知
        </div>
        <nav aria-label="主要导航">
          {navigation.map(({ label, icon: Icon }, index) => (
            <button className={index === 0 ? "nav-item is-active" : "nav-item"} key={label}>
              <Icon aria-hidden="true" size={18} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
      </aside>
      <main className="workspace">
        <header className="topbar">
          <h1>抖音知识资料库</h1>
          <span className="connection-status">
            <span aria-hidden="true" className="status-dot" />
            本地服务已连接
          </span>
        </header>
        <section className="empty-workspace" aria-labelledby="workspace-title">
          <h2 id="workspace-title">主播资料</h2>
          <p>添加主播主页后，这里会显示同步状态与内容资料。</p>
        </section>
      </main>
    </div>
  )
}

