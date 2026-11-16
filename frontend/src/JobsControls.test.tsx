import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"

import { App } from "./App"

afterEach(() => { cleanup(); vi.restoreAllMocks() })

const oldJob = { id: 1, kind: "generate_note", work_id: null, queue_lane: "legacy_upgrade", lane_label: "旧笔记升级", status: "paused", stage: "analyze", progress: 0, attempts: 1, error_code: null }

it("filters older paused tasks, resumes and cancels them individually", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = new URL(String(input), "http://test")
    if (url.pathname === "/api/jobs" && url.searchParams.get("status") === "paused") return new Response(JSON.stringify({ items: [oldJob], total: 1 }))
    if (url.pathname === "/api/jobs") return new Response(JSON.stringify({ items: [], total: 110 }))
    if (url.pathname === "/api/jobs/1/retry") return new Response(JSON.stringify({ ...oldJob, status: "pending" }))
    if (url.pathname === "/api/jobs/1/cancel") return new Response(JSON.stringify({ ...oldJob, status: "cancelled" }))
    return new Response(JSON.stringify({ status: "ok" }))
  })
  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "任务中心" }))
  fireEvent.change(screen.getByRole("combobox", { name: "任务状态" }), { target: { value: "paused" } })
  await screen.findByText("#1")
  fireEvent.click(screen.getByRole("button", { name: "恢复任务 1" }))
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/jobs/1/retry", expect.objectContaining({ method: "POST" })))
  fireEvent.click(screen.getByRole("button", { name: "取消任务 1" }))
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/jobs/1/cancel", expect.objectContaining({ method: "POST" })))
})

it("shows terminal legacy failure details with an explicit retry control", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    if (path === "/api/jobs") return new Response(JSON.stringify({ items: [{ ...oldJob, status: "failed", error_detail: "quality_gate_failed" }] }))
    if (path === "/api/jobs/1/retry") return new Response(JSON.stringify({ ...oldJob, status: "pending" }))
    return new Response(JSON.stringify({ status: "ok" }))
  })
  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "任务中心" }))
  expect(await screen.findByText("quality_gate_failed")).toBeVisible()
  fireEvent.click(screen.getByRole("button", { name: "重试任务 1" }))
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/jobs/1/retry", expect.objectContaining({ method: "POST" })))
})

it("only offers failed retries for the supported legacy upgrade lane", async () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    if (String(input) === "/api/jobs") return new Response(JSON.stringify({ items: [
      { ...oldJob, id: 2, status: "failed", queue_lane: "new_note" },
      { ...oldJob, id: 3, status: "failed", kind: "download_work", queue_lane: "pipeline" },
    ] }))
    return new Response(JSON.stringify({ status: "ok" }))
  })
  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "任务中心" }))
  await screen.findByText("#2")
  expect(screen.queryByRole("button", { name: "重试任务 2" })).not.toBeInTheDocument()
  expect(screen.queryByRole("button", { name: "重试任务 3" })).not.toBeInTheDocument()
})
