import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

import { App } from "./App"

afterEach(() => {
  cleanup()
})

beforeEach(() => {
  vi.restoreAllMocks()
})

it("shows the local knowledge library identity and a successful health check", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  )

  render(<App />)

  expect(screen.getByRole("heading", { name: "抖音知识资料库" })).toBeVisible()
  expect(await screen.findByText("本地服务已连接")).toBeVisible()
})

it("shows connecting while the health check is pending", () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(() => new Promise<Response>(() => undefined))

  render(<App />)

  expect(screen.getByText("正在连接本地服务")).toBeVisible()
})

it("shows disconnected when the health check fails", async () => {
  vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("fetch failed"))

  render(<App />)

  expect(await screen.findByText("本地服务未连接")).toBeVisible()
})

it("shows the global transcription progress across views", async () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    if (String(input) === "/api/jobs/progress") {
      return new Response(JSON.stringify({
        works: { collected: 2508, audio_ready: 157 },
        transcription: {
          total: 157,
          completed: 89,
          running: 1,
          pending: 63,
          failed: 4,
          not_queued: 0,
          percent: 56.7,
        },
      }), { status: 200, headers: { "Content-Type": "application/json" } })
    }
    return new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })

  render(<App />)

  const progress = await screen.findByRole("progressbar", { name: "音频转写进度" })
  expect(progress).toHaveAttribute("aria-valuenow", "56.7")
  expect(screen.getByText("89 / 157")).toBeVisible()
  expect(screen.getByText("处理中 1")).toBeVisible()
  expect(screen.getByText("待处理 63")).toBeVisible()
  expect(screen.getByText("失败 4")).toBeVisible()

  fireEvent.click(screen.getByRole("button", { name: "资料库" }))
  expect(screen.getByRole("progressbar", { name: "音频转写进度" })).toBeVisible()
})

it("shows note progress and queues all missing notes", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input)
    if (path === "/api/jobs/progress") {
      return new Response(JSON.stringify({
        works: { collected: 10, audio_ready: 8 },
        audio_extraction: { total: 10, completed: 8, running: 0, pending: 0, failed: 0, not_queued: 2, percent: 80 },
        transcription: { total: 8, completed: 6, running: 0, pending: 0, failed: 0, not_queued: 2, percent: 75 },
      }), { status: 200, headers: { "Content-Type": "application/json" } })
    }
    if (path === "/api/notes/progress") {
      return new Response(JSON.stringify({
        total: 6,
        completed: 2,
        running: 1,
        pending: 0,
        failed: 0,
        skipped: 2,
        not_queued: 3,
        percent: 33.3,
        recent_aweme_id: "731",
        failures: [],
      }), { status: 200, headers: { "Content-Type": "application/json" } })
    }
    if (path === "/api/notes/generate" && init?.method === "POST") {
      return new Response(JSON.stringify({ queued: 3, already_queued: 0, skipped: 2 }), {
        status: 202,
        headers: { "Content-Type": "application/json" },
      })
    }
    return new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })

  render(<App />)

  const progress = await screen.findByRole("progressbar", { name: "知识笔记生成进度" })
  expect(progress).toHaveAttribute("aria-valuenow", "33.3")
  expect(screen.getByText("最近处理 731")).toBeVisible()
  fireEvent.click(screen.getByRole("button", { name: "生成缺失笔记" }))
  expect(await screen.findByText("已加入 3 条笔记任务。")).toBeVisible()
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/notes/generate",
    expect.objectContaining({ method: "POST", body: JSON.stringify({ force: false }) }),
  )
})

it("uses an actionable message for an empty non-json error", async () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(null, { status: 502 }))

  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "任务中心" }))

  expect(await screen.findByText("本地服务请求失败 (502)")).toBeVisible()
})

it("previews a creator and enables confirmed synchronization", async () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input)
    if (path === "/api/creators/preview") {
      return new Response(
        JSON.stringify({
          sec_uid: "sec-731",
          nickname: "知识主播",
          unique_id: "knowledge731",
          profile_url: "https://www.douyin.com/user/sec-731",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      )
    }
    if (path === "/api/creators/sync" && init?.method === "POST") {
      return new Response(
        JSON.stringify({ id: 9, kind: "sync_creator", status: "pending" }),
        { status: 202, headers: { "Content-Type": "application/json" } },
      )
    }
    return new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })

  render(<App />)
  fireEvent.change(screen.getByLabelText("主播主页或 sec_uid"), {
    target: { value: "https://www.douyin.com/user/sec-731" },
  })
  fireEvent.click(screen.getByRole("button", { name: "预览主播" }))

  expect(await screen.findByText("知识主播")).toBeVisible()
  const syncButton = screen.getByRole("button", { name: "确认并同步主播" })
  expect(syncButton).toBeEnabled()
  fireEvent.click(syncButton)
  expect(await screen.findByText("主播同步任务已加入队列，可前往任务中心查看进度。")).toBeVisible()
})

it("queues a favorites sync instead of claiming it completed inline", async () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input)
    if (path === "/api/favorites/sync" && init?.method === "POST") {
      return new Response(
        JSON.stringify({ id: 10, kind: "sync_favorites", status: "pending" }),
        { status: 202, headers: { "Content-Type": "application/json" } },
      )
    }
    return new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })

  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "我的收藏" }))
  fireEvent.click(screen.getByRole("button", { name: "同步全部收藏" }))

  expect(await screen.findByText("收藏同步任务已加入队列，可前往任务中心查看进度。")).toBeVisible()
})

it("switches to the jobs view and loads progress records", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
    async (input, init) => new Response(
      JSON.stringify({
        ...(String(input) === "/api/jobs/7/cancel" && init?.method === "POST" ? {
          id: 7,
          kind: "download_work",
          work_id: 731,
          status: "cancelled",
          stage: "extract",
          progress: 0.5,
          attempts: 0,
          error_code: "cancelled_by_user",
        } : { items: [
          {
            id: 7,
            kind: "download_work",
            work_id: 731,
            status: "running",
            stage: "extract",
            progress: 0.5,
            attempts: 0,
            error_code: null,
          },
        ] }),
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ),
  )

  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "任务中心" }))

  await waitFor(() => expect(screen.getByText("download_work")).toBeVisible())
  expect(screen.getByText("提取音频")).toBeVisible()
  fireEvent.click(screen.getByRole("button", { name: "取消任务 7" }))
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
    "/api/jobs/7/cancel",
    expect.objectContaining({ method: "POST" }),
  ))
})

it("selects jobs, pauses them in bulk, and refreshes stale work", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input)
    if (path === "/api/jobs/refresh" && init?.method === "POST") {
      return new Response(JSON.stringify({ recovered: 1, items: [{ id: 7, kind: "generate_note", work_id: 731, status: "pending", stage: "analyze", progress: 0, attempts: 0, error_code: null }] }), { status: 200 })
    }
    if (path === "/api/jobs/bulk-pause" && init?.method === "POST") {
      return new Response(JSON.stringify({ paused: 1, items: [{ id: 7, kind: "generate_note", work_id: 731, status: "paused", stage: "analyze", progress: 0, attempts: 0, error_code: "paused_by_user" }] }), { status: 200 })
    }
    if (path === "/api/jobs") {
      return new Response(JSON.stringify({ items: [{ id: 7, kind: "generate_note", work_id: 731, status: "pending", stage: "analyze", progress: 0, attempts: 0, error_code: null }] }), { status: 200 })
    }
    return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
  })

  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "任务中心" }))
  await screen.findByText("generate_note")
  fireEvent.click(screen.getByRole("button", { name: "多选" }))
  fireEvent.click(screen.getByRole("checkbox", { name: "选择任务 7" }))
  fireEvent.click(screen.getByRole("button", { name: "暂停 (1)" }))
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/jobs/bulk-pause", expect.objectContaining({ method: "POST", body: JSON.stringify({ job_ids: [7] }) })))
  fireEvent.click(screen.getByRole("button", { name: "刷新任务" }))
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/jobs/refresh", expect.objectContaining({ method: "POST" })))
  expect(await screen.findByText("已刷新任务，并重新排队 1 个卡住任务。")).toBeVisible()
})

it("shows separate lanes and reports bounded refresh results", async () => {
  const jobs = [
    { id: 7, kind: "generate_note", queue_lane: "new_note", lane_label: "新笔记录入", work_id: 731, status: "pending", stage: "analyze", progress: 0, attempts: 0, error_code: null },
    { id: 8, kind: "generate_note", queue_lane: "legacy_upgrade", lane_label: "旧笔记升级", work_id: 732, status: "paused", stage: "analyze", progress: 0, attempts: 0, error_code: "paused_by_user" },
  ]
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input)
    if (path === "/api/jobs/refresh" && init?.method === "POST") {
      return new Response(JSON.stringify({ items: jobs, recovered: 1, recovered_by_lane: { new_note: 0, legacy_upgrade: 1 }, legacy_refill: { queued: 2, reason: "new_queue_empty" } }), { status: 200 })
    }
    if (path === "/api/jobs") return new Response(JSON.stringify({ items: jobs }), { status: 200 })
    return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
  })

  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "任务中心" }))
  expect(await screen.findByText("新笔记录入")).toBeVisible()
  expect(screen.getByText("旧笔记升级")).toBeVisible()
  fireEvent.click(screen.getByRole("button", { name: "刷新任务" }))
  expect(await screen.findByText("已刷新：恢复 1 个卡住任务，旧升级补入 2 条。")).toBeVisible()
})

it("keeps skipped bulk selections visible with their result", async () => {
  const jobs = [
    { id: 7, kind: "generate_note", work_id: 731, status: "pending", stage: "analyze", progress: 0, attempts: 0, error_code: null },
    { id: 8, kind: "generate_note", work_id: 732, status: "paused", stage: "analyze", progress: 0, attempts: 0, error_code: "paused_by_user" },
  ]
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input)
    if (path === "/api/jobs/bulk-pause" && init?.method === "POST") return new Response(JSON.stringify({ paused: 1, skipped: 1, items: [{ ...jobs[0], status: "paused" }], results: [{ id: 7, status: "paused", reason: null }, { id: 8, status: "skipped", reason: "invalid_transition:paused->paused" }] }), { status: 200 })
    if (path === "/api/jobs") return new Response(JSON.stringify({ items: jobs }), { status: 200 })
    return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
  })

  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "任务中心" }))
  await screen.findAllByText("generate_note")
  fireEvent.click(screen.getByRole("button", { name: "多选" }))
  fireEvent.click(screen.getByRole("checkbox", { name: "全选任务" }))
  fireEvent.click(screen.getByRole("button", { name: "暂停 (2)" }))
  expect(await screen.findByText("已暂停 1 个任务，跳过 1 个。#8：已暂停")).toBeVisible()
  expect(screen.getByRole("checkbox", { name: "选择任务 8" })).toBeChecked()
})

it("validates and saves a douyin cookie without retaining the secret", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input)
    if (path === "/api/settings/diagnostics") {
      return new Response(JSON.stringify({ storage: { status: "ok", path: "D:\\DouyinKnowledgeLibrary" } }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    }
    if (path === "/api/settings/douyin-cookie" && init?.method === "PUT") {
      return new Response(
        JSON.stringify({
          configured: true,
          valid: true,
          account_nickname: "知识主播",
          expires_or_unknown: "unknown",
          risk_controlled: false,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      )
    }
    if (path === "/api/settings/douyin-cookie") {
      return new Response(JSON.stringify({ configured: false }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    }
    return new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })

  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "设置" }))
  const input = await screen.findByLabelText("抖音 Cookie")
  fireEvent.change(input, { target: { value: "sessionid=super-secret" } })
  fireEvent.click(screen.getByRole("button", { name: "验证并保存" }))

  expect(await screen.findByText("抖音账号已连接：知识主播")).toBeVisible()
  expect(input).toHaveValue("")
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/settings/douyin-cookie",
    expect.objectContaining({
      method: "PUT",
      body: JSON.stringify({ cookie: "sessionid=super-secret" }),
    }),
  )
})

it("explains when the Windows credential store is unavailable", async () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input)
    if (path === "/api/settings/douyin-cookie" && init?.method === "PUT") {
      return new Response(JSON.stringify({ detail: "douyin_cookie_store_unavailable" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      })
    }
    if (path === "/api/settings/douyin-cookie") {
      return new Response(JSON.stringify({ configured: false }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    }
    return new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })

  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "设置" }))
  fireEvent.change(await screen.findByLabelText("抖音 Cookie"), {
    target: { value: "sessionid=super-secret" },
  })
  fireEvent.click(screen.getByRole("button", { name: "验证并保存" }))

  expect(await screen.findByText("Windows 凭据库当前不可用，请在普通用户会话中启动后端后重试。")).toBeVisible()
  expect(screen.queryByText(/super-secret/)).not.toBeInTheDocument()
})

it("removes a configured douyin cookie", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input)
    if (path === "/api/settings/diagnostics") {
      return new Response(JSON.stringify({ storage: { status: "ok", path: "D:\\DouyinKnowledgeLibrary" } }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    }
    if (path === "/api/settings/douyin-cookie" && init?.method === "DELETE") {
      return new Response(JSON.stringify({ configured: false }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    }
    if (path === "/api/settings/douyin-cookie") {
      return new Response(JSON.stringify({ configured: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    }
    return new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })

  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "设置" }))
  expect(await screen.findByText("已配置抖音 Cookie")).toBeVisible()
  fireEvent.click(screen.getByRole("button", { name: "移除凭据" }))

  expect(await screen.findByText("尚未配置抖音 Cookie")).toBeVisible()
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/settings/douyin-cookie",
    expect.objectContaining({ method: "DELETE" }),
  )
})

it("opens and reads a completed transcript from the library", async () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    if (path === "/api/works?limit=100") {
      return new Response(JSON.stringify({ items: [{
        id: 1,
        aweme_id: "731",
        description: "知识口述",
        media_type: "video",
        availability: "available",
        asset_kinds: ["audio", "transcript_text"],
        creator_nickname: "知识主播",
        published_at: null,
        topics: [],
        transcript_status: "succeeded",
        transcript_preview: "这是完整转写正文。",
      }] }), { status: 200, headers: { "Content-Type": "application/json" } })
    }
    if (path === "/api/works/1") {
      return new Response(JSON.stringify({
        id: 1,
        transcript: {
          engine: "qwen3-asr",
          model: "Qwen/Qwen3-ASR-0.6B",
          language: "Chinese",
          text: "这是完整转写正文。第二段也在这里。",
          segments: [],
        },
      }), { status: 200, headers: { "Content-Type": "application/json" } })
    }
    return new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })

  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "资料库" }))
  fireEvent.click(await screen.findByRole("button", { name: "阅读知识口述的转写" }))

  expect(await screen.findByText("这是完整转写正文。第二段也在这里。")).toBeVisible()
  expect(screen.getByText("Qwen/Qwen3-ASR-0.6B")).toBeVisible()
})

it("paginates the full library and opens the learning note content", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    if (path === "/api/works?limit=100") {
      return new Response(JSON.stringify({
        items: [{
          id: 1,
          aweme_id: "note-731",
          description: "可复用的学习笔记",
          media_type: "video",
          availability: "available",
          asset_kinds: ["audio", "transcript_text"],
          creator_nickname: "知识主播",
          published_at: null,
          topics: ["学习"],
          transcript_status: "succeeded",
          transcript_preview: "转写预览",
          note_status: "succeeded",
          note_available: true,
          note_preview: "核心观点预览",
          extraction_status: "succeeded",
          failure: null,
        }],
        total: 201,
        limit: 100,
        offset: 0,
        has_more: true,
        failed_total: 4,
      }), { status: 200, headers: { "Content-Type": "application/json" } })
    }
    if (path === "/api/works?limit=100&offset=100") {
      return new Response(JSON.stringify({ items: [], total: 201, limit: 100, offset: 100, has_more: true, failed_total: 4 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    }
    if (path === "/api/works/1") {
      return new Response(JSON.stringify({
        id: 1,
        description: "可复用的学习笔记",
        note: { path: "D:\\DouyinKnowledgeLibrary\\dy-note\\note-731\\learning_note.md", text: "# 可复用的学习笔记\n\n## 核心观点\n\n- 先做再复盘。" },
        note_status: "succeeded",
        extraction_status: "succeeded",
        failure: null,
        transcript: null,
      }), { status: 200, headers: { "Content-Type": "application/json" } })
    }
    return new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })

  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "资料库" }))
  fireEvent.click(await screen.findByRole("button", { name: "阅读可复用的学习笔记的学习笔记" }))

  expect(await screen.findByText(/先做再复盘/)).toBeVisible()
  expect(screen.getByRole("dialog")).toBeVisible()
  expect(screen.getByText("learning_note.md · 基于 ASR 转写生成")).toBeVisible()
  fireEvent.click(screen.getByRole("button", { name: "关闭资料" }))
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: "下一页" }))
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/works?limit=100&offset=100", expect.anything()))
})

it("keeps a stale library response from replacing a newer filter result", async () => {
  let resolveOld!: (response: Response) => void
  let resolveNew!: (response: Response) => void
  const oldResponse = new Promise<Response>((resolve) => { resolveOld = resolve })
  const newResponse = new Promise<Response>((resolve) => { resolveNew = resolve })
  const jsonResponse = (body: unknown) => new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
    const path = String(input)
    if (path === "/api/works?limit=100&query=old") return oldResponse
    if (path === "/api/works?limit=100&query=new") return newResponse
    if (path.startsWith("/api/works?")) return Promise.resolve(jsonResponse({ items: [], total: 0, failed_total: 0 }))
    return Promise.resolve(jsonResponse({ status: "ok" }))
  })

  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "资料库" }))
  const input = await screen.findByLabelText("筛选资料")
  fireEvent.change(input, { target: { value: "old" } })
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/works?limit=100&query=old", expect.anything()))
  fireEvent.change(input, { target: { value: "new" } })
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/works?limit=100&query=new", expect.anything()))

  resolveNew(jsonResponse({
    items: [{
      id: 2,
      aweme_id: "new",
      description: "新结果",
      media_type: "video",
      availability: "available",
      asset_kinds: [],
      creator_nickname: "主播",
      published_at: null,
      topics: [],
      transcript_status: "succeeded",
      transcript_preview: "",
      note_status: "missing",
      note_available: false,
      note_preview: null,
      extraction_status: "succeeded",
      failure: null,
    }],
    total: 1,
    failed_total: 0,
  }))
  expect(await screen.findByText("新结果")).toBeVisible()

  resolveOld(jsonResponse({
    items: [{
      id: 1,
      aweme_id: "old",
      description: "旧结果",
      media_type: "video",
      availability: "available",
      asset_kinds: [],
      creator_nickname: "主播",
      published_at: null,
      topics: [],
      transcript_status: "succeeded",
      transcript_preview: "",
      note_status: "missing",
      note_available: false,
      note_preview: null,
      extraction_status: "succeeded",
      failure: null,
    }],
    total: 1,
    failed_total: 0,
  }))
  await waitFor(() => expect(screen.queryByText("旧结果")).not.toBeInTheDocument())
  expect(screen.getByText("新结果")).toBeVisible()
})

it.each([false, true])("upgrades legacy notes locally when Ollama is ready with external configured %s", async (configured) => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input)
    if (path === "/api/jobs/progress") return new Response(JSON.stringify({ works: { collected: 1, audio_ready: 1 }, audio_extraction: { total: 1, completed: 1, running: 0, pending: 0, failed: 0, not_queued: 0, percent: 100 }, transcription: { total: 1, completed: 1, running: 0, pending: 0, failed: 0, not_queued: 0, percent: 100 } }), { status: 200 })
    if (path === "/api/notes/progress") return new Response(JSON.stringify({ total: 1, completed: 1, running: 0, pending: 0, failed: 0, skipped: 1, not_queued: 0, percent: 100, recent_aweme_id: null, failures: [], quality_counts: { semantic_complete: 0, fallback: 0, legacy: 1, unknown: 0 }, upgrade: { eligible: 1 }, }), { status: 200 })
    if (path === "/api/notes/provider-status") return new Response(JSON.stringify({ ollama: { ready: true, model: "qwen3:4b" }, external: { configured, ready: null } }), { status: 200 })
    if (path === "/api/notes/generate" && init?.method === "POST") return new Response(JSON.stringify({ queued: 1, already_queued: 0, skipped: 0 }), { status: 202 })
    return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
  })

  render(<App />)
  fireEvent.click(await screen.findByRole("button", { name: "升级旧笔记" }))
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/notes/generate", expect.objectContaining({ method: "POST", body: JSON.stringify({ legacy_only: true, limit: 3, provider_policy: "local" }) })))
})

it("uses the legacy upgrade eligible count when quality counts are absent", async () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    if (path === "/api/jobs/progress") {
      return new Response(JSON.stringify({
        works: { collected: 1, audio_ready: 1 },
        audio_extraction: { total: 1, completed: 1, running: 0, pending: 0, failed: 0, not_queued: 0, percent: 100 },
        transcription: { total: 1, completed: 1, running: 0, pending: 0, failed: 0, not_queued: 0, percent: 100 },
      }), { status: 200 })
    }
    if (path === "/api/notes/progress") {
      return new Response(JSON.stringify({
        total: 1,
        completed: 1,
        running: 0,
        pending: 0,
        failed: 0,
        skipped: 1,
        not_queued: 0,
        percent: 100,
        recent_aweme_id: null,
        failures: [],
        upgrade: { eligible: 2 },
      }), { status: 200 })
    }
    if (path === "/api/notes/provider-status") {
      return new Response(JSON.stringify({ ollama: { ready: true, model: "qwen3:8b" } }), { status: 200 })
    }
    return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
  })

  render(<App />)

  expect(await screen.findByRole("button", { name: "升级旧笔记" })).toBeVisible()
})

it("shows queued and already-active upgrade counts and updates local progress", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input)
    if (path === "/api/jobs/progress") {
      return new Response(JSON.stringify({
        works: { collected: 1, audio_ready: 1 },
        audio_extraction: { total: 1, completed: 1, running: 0, pending: 0, failed: 0, not_queued: 0, percent: 100 },
        transcription: { total: 1, completed: 1, running: 0, pending: 0, failed: 0, not_queued: 0, percent: 100 },
      }), { status: 200 })
    }
    if (path === "/api/notes/progress") {
      return new Response(JSON.stringify({
        total: 1,
        completed: 1,
        running: 0,
        pending: 0,
        failed: 0,
        skipped: 1,
        not_queued: 0,
        percent: 100,
        recent_aweme_id: null,
        failures: [],
        upgrade: { total: 3, completed: 0, remaining: 3, active: 0, eligible: 3, percent: 0 },
      }), { status: 200 })
    }
    if (path === "/api/notes/provider-status") {
      return new Response(JSON.stringify({ ollama: { ready: true, model: "qwen3:8b" } }), { status: 200 })
    }
    if (path === "/api/notes/generate" && init?.method === "POST") {
      return new Response(JSON.stringify({ queued: 2, already_queued: 1, skipped: 0 }), { status: 202 })
    }
    return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
  })

  render(<App />)
  fireEvent.click(await screen.findByRole("button", { name: "升级旧笔记" }))

  await waitFor(() => expect(screen.getByText(/新加入 2 条，已有 1 条在队列中/)).toBeVisible())
  expect(screen.getByText("处理中 2")).toBeVisible()
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/notes/generate",
    expect.objectContaining({ method: "POST", body: JSON.stringify({ legacy_only: true, limit: 3, provider_policy: "local" }) }),
  )
})

it("explains failed legacy upgrades and directs explicit retry to the task center", async () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    const data = path.startsWith("/api/notes/progress") ? {
      total: 1, completed: 1, running: 0, pending: 0, failed: 0, skipped: 1,
      not_queued: 0, percent: 100, recent_aweme_id: null, failures: [],
      upgrade: { total: 1, completed: 0, remaining: 1, failed: 1, eligible: 1 },
    } : path === "/api/jobs/progress" ? {
      works: { collected: 1, audio_ready: 1 },
      transcription: { total: 1, completed: 1, running: 0, pending: 0, failed: 0, not_queued: 0, percent: 100 },
    } : path === "/api/notes/provider-status" ? { ollama: { ready: true } }
      : path === "/api/notes/generate" ? {
        queued: 0, already_queued: 0, skipped: 0, blocked_failed: 1,
        legacy_scheduler: { reason: "no_candidates", blocked_failed: 1 },
      } : { status: "ok" }
    return new Response(JSON.stringify(data))
  })
  render(<App />)
  fireEvent.click(await screen.findByRole("button", { name: "升级旧笔记" }))
  expect(await screen.findByText(/1 条旧升级曾失败，请在任务中心查看原因并重试/)).toBeVisible()
  expect(screen.queryByText("当前没有可升级的旧笔记。")).not.toBeInTheDocument()
})

it.each([true, false])("disables upgrades without Ollama even when external is ready %s", async (configured) => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    if (path === "/api/jobs/progress") {
      return new Response(JSON.stringify({
        works: { collected: 2, audio_ready: 2 },
        audio_extraction: { total: 2, completed: 2, running: 0, pending: 0, failed: 0, not_queued: 0, percent: 100 },
        transcription: { total: 2, completed: 2, running: 0, pending: 0, failed: 0, not_queued: 0, percent: 100 },
      }), { status: 200 })
    }
    if (path === "/api/notes/progress") {
      return new Response(JSON.stringify({
        total: 2,
        completed: 2,
        running: 0,
        pending: 0,
        failed: 0,
        skipped: 2,
        not_queued: 0,
        percent: 100,
        recent_aweme_id: null,
        failures: [],
        quality_counts: { semantic_complete: 0, fallback: 1, legacy: 0, unknown: 1, missing: 0 },
        upgrade: { total: 2, completed: 0, remaining: 2, active: 0, failed: 0, eligible: 2, percent: 0 },
      }), { status: 200 })
    }
    if (path === "/api/notes/provider-status") {
      return new Response(JSON.stringify({
        ready: configured,
        ollama: { ready: false, model: "qwen3:8b" },
        external: { configured, ready: configured },
        external_configured: configured,
      }), { status: 200 })
    }
    return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
  })

  render(<App />)

  const button = await screen.findByRole("button", { name: "升级旧笔记" })
  expect(button).toBeDisabled()
  fireEvent.click(button)
  expect(screen.getByText(/回退 1 · 旧 0 · 未知 1/)).toBeVisible()
  expect(screen.getByRole("progressbar", { name: "旧笔记语义升级进度" })).toHaveAttribute("aria-valuetext", "0 / 2，0%")
  expect(fetchMock).not.toHaveBeenCalledWith("/api/notes/generate", expect.anything())
})

it.each(["ollama", "external"])("uses only local readiness from the cached %s provider summary", async (provider) => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    if (path === "/api/jobs/progress") return new Response(JSON.stringify({
      works: { collected: 1, audio_ready: 1 },
      audio_extraction: { total: 1, completed: 1, running: 0, pending: 0, failed: 0, not_queued: 0, percent: 100 },
      transcription: { total: 1, completed: 1, running: 0, pending: 0, failed: 0, not_queued: 0, percent: 100 },
    }), { status: 200 })
    if (path === "/api/notes/provider-status") return new Response("unavailable", { status: 503 })
    if (path === "/api/notes/progress") return new Response(JSON.stringify({
      total: 1, completed: 1, running: 0, pending: 0, failed: 0, skipped: 1,
      not_queued: 0, percent: 100, recent_aweme_id: null, failures: [],
      upgrade: { eligible: 1 },
      provider_summary: { provider, ready: true, external_configured: true },
    }), { status: 200 })
    return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
  })

  render(<App />)
  const button = await screen.findByRole("button", { name: "升级旧笔记" })
  if (provider === "ollama") expect(button).toBeEnabled()
  else {
    expect(button).toBeDisabled()
    fireEvent.click(button)
    expect(fetchMock).not.toHaveBeenCalledWith("/api/notes/generate", expect.anything())
  }
})
