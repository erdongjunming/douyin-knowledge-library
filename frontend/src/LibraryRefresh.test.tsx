import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"

import { App } from "./App"

const work = {
  id: 17, aweme_id: "legacy-17", description: "Pending lesson", media_type: "video",
  availability: "available", asset_kinds: ["audio"], creator_nickname: "Creator",
  published_at: null, topics: [], transcript_status: "succeeded", transcript_preview: "Source text",
  note_status: "pending", note_available: false, note_preview: null,
  extraction_status: "succeeded", failure: null,
}
const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200 })

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

it("makes a finished note readable without leaving the library", async () => {
  vi.useFakeTimers()
  let ready = false
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    if (path.startsWith("/api/works?")) return json({ items: [{ ...work, note_available: ready, note_status: ready ? "succeeded" : "pending" }] })
    if (path === "/api/works/17") return json({ ...work, note: { path: "learning_note.md", text: "## Completed lesson\n\nPublished content." }, transcript: null })
    return json({ status: "ok" })
  })
  render(<App />)
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "资料库" })) })
  expect(screen.getByRole("button", { name: "阅读Pending lesson的转写" })).toBeVisible()
  ready = true
  await act(async () => { await vi.advanceTimersByTimeAsync(5000) })

  fireEvent.click(screen.getByRole("button", { name: "阅读Pending lesson的学习笔记" }))
  await act(async () => { await vi.advanceTimersByTimeAsync(0) })
  expect(screen.getByText("Published content.")).toBeVisible()
})

it("refreshes an open reader after publication and keeps content during a failed refresh", async () => {
  vi.useFakeTimers()
  let ready = false
  let unavailable = false
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    if (path.startsWith("/api/works?")) return json({ items: [work] })
    if (path === "/api/works/17") {
      if (unavailable) return new Response("unavailable", { status: 503 })
      return json({ ...work, note: ready ? { path: "learning_note.md", text: "## Complete lesson\n\nFresh content." } : null, transcript: { text: "Original transcript.", model: "local-asr" } })
    }
    return json({ status: "ok" })
  })
  render(<App />)
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "资料库" })) })
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "阅读Pending lesson的转写" })) })
  expect(screen.getByText("Original transcript.")).toBeVisible()
  ready = true
  await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
  expect(screen.getByText("Fresh content.")).toBeVisible()
  unavailable = true
  await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
  expect(screen.getByText("Fresh content.")).toBeVisible()
  fireEvent.click(screen.getByRole("button", { name: "关闭资料" }))
  await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
})

it("keeps the open note when a refresh briefly returns an empty note", async () => {
  vi.useFakeTimers()
  let empty = false
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    if (path.startsWith("/api/works?")) return json({ items: [{ ...work, note_available: true, note_status: "succeeded" }] })
    if (path === "/api/works/17") {
      return json({
        ...work,
        note: empty ? null : { path: "learning_note.md", text: "## Stable note\n\nKeep this content." },
        transcript: { text: "Fallback transcript.", model: "local-asr" },
      })
    }
    return json({ status: "ok" })
  })
  render(<App />)
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "资料库" })) })
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "阅读Pending lesson的学习笔记" })) })
  expect(screen.getByText("Keep this content.")).toBeVisible()
  empty = true
  await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
  expect(screen.getByText("Keep this content.")).toBeVisible()
  expect(screen.queryByText("Fallback transcript.")).not.toBeInTheDocument()
})

it("uses a guarded local upgrade when generating a complete note for an existing legacy note", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    if (String(input).startsWith("/api/works?")) return json({ items: [{ ...work, note_available: true, note_status: "succeeded", note_quality_status: "legacy" }] })
    if (String(input) === "/api/notes/generate") return json({ queued: 1, already_queued: 0, skipped: 0 })
    return json({ status: "ok" })
  })
  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "资料库" }))
  fireEvent.click(await screen.findByRole("button", { name: "生成Pending lesson的完整笔记" }))
  expect(fetchMock).toHaveBeenCalledWith("/api/notes/generate", expect.objectContaining({
    method: "POST", body: JSON.stringify({ aweme_ids: ["legacy-17"], provider_policy: "local", force: false, legacy_only: true }),
  }))
})

it.each([
  { blocked_failed: 1 },
  { legacy_scheduler: { blocked_failed: 1 } },
])("explains a blocked single-note upgrade and directs retry to the task center: %j", async (failure) => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    if (path.startsWith("/api/works?")) return json({ items: [{ ...work, note_available: true, note_status: "succeeded", note_quality_status: "legacy" }] })
    if (path === "/api/notes/generate") return json({ queued: 0, already_queued: 0, skipped: 0, ...failure })
    return json({ status: "ok" })
  })
  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "资料库" }))
  fireEvent.click(await screen.findByRole("button", { name: "生成Pending lesson的完整笔记" }))

  const message = await screen.findByText("该笔记的旧升级曾失败，请在任务中心查看原因并重试。")
  expect(message).toHaveClass("error")
  expect(screen.queryByText("当前没有可生成的完整笔记。")).not.toBeInTheDocument()
})
