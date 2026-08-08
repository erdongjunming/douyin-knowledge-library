import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

import { App } from "./App"

afterEach(() => {
  cleanup()
})

beforeEach(() => {
  vi.restoreAllMocks()
})

it("shows the local knowledge library identity", () => {
  render(<App />)

  expect(screen.getByRole("heading", { name: "抖音知识资料库" })).toBeVisible()
  expect(screen.getByText("本地服务已连接")).toBeVisible()
})

it("previews a creator and enables confirmed synchronization", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(
      JSON.stringify({
        sec_uid: "sec-731",
        nickname: "知识主播",
        unique_id: "knowledge731",
        profile_url: "https://www.douyin.com/user/sec-731",
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ),
  )

  render(<App />)
  fireEvent.change(screen.getByLabelText("主播主页或 sec_uid"), {
    target: { value: "https://www.douyin.com/user/sec-731" },
  })
  fireEvent.click(screen.getByRole("button", { name: "预览主播" }))

  expect(await screen.findByText("知识主播")).toBeVisible()
  expect(screen.getByRole("button", { name: "确认并同步主播" })).toBeEnabled()
})

it("switches to the jobs view and loads progress records", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(
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
