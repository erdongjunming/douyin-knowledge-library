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
  vi.spyOn(globalThis, "fetch").mockImplementation(
    async () => new Response(
      JSON.stringify({
        items: [
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
        ],
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ),
  )

  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "任务中心" }))

  await waitFor(() => expect(screen.getByText("download_work")).toBeVisible())
  expect(screen.getByText("提取音频")).toBeVisible()
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
