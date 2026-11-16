import { act, cleanup, render } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"

import { App } from "./App"

afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks() })

const response = (body: unknown = { status: "ok" }) => new Response(JSON.stringify(body))
const settle = async () => { await act(async () => { await vi.advanceTimersByTimeAsync(0) }) }

it("retains a single progress polling chain after repeated refresh events", async () => {
  vi.useFakeTimers()
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => response())
  const calls = (path: string) => fetchMock.mock.calls.filter(([input]) => String(input).split("?")[0] === path).length
  const view = render(<App />)
  await settle()
  for (let index = 0; index < 3; index++) {
    act(() => window.dispatchEvent(new Event("dkl:refresh-progress")))
    await settle()
  }
  await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
  expect(calls("/api/notes/progress")).toBe(5)
  expect(calls("/api/jobs/progress")).toBe(5)
  view.unmount()
  const before = fetchMock.mock.calls.length
  await act(async () => { await vi.advanceTimersByTimeAsync(30000) })
  expect(fetchMock.mock.calls.length).toBe(before)
})

it("coalesces refresh events during an in-flight request and aborts on unmount", async () => {
  vi.useFakeTimers()
  let release!: (value: Response) => void
  let signal: AbortSignal | undefined
  let requests = 0
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    if (!String(input).startsWith("/api/notes/progress")) return response()
    requests++
    signal = init?.signal as AbortSignal
    if (requests === 1) return await new Promise<Response>((resolve) => { release = resolve })
    return response()
  })
  const view = render(<App />)
  await settle()
  act(() => {
    for (let index = 0; index < 4; index++) window.dispatchEvent(new Event("dkl:refresh-progress"))
  })
  expect(requests).toBe(1)
  await act(async () => { release(response()); await vi.advanceTimersByTimeAsync(0) })
  expect(requests).toBe(2)
  view.unmount()
  expect(signal?.aborted).toBe(true)
})

it("re-probes provider readiness periodically and on refresh", async () => {
  vi.useFakeTimers()
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => response())
  const calls = () => fetchMock.mock.calls.filter(([input]) => String(input) === "/api/notes/provider-status").length
  render(<App />)
  await settle()
  await act(async () => { await vi.advanceTimersByTimeAsync(30000) })
  expect(calls()).toBe(2)
  act(() => window.dispatchEvent(new Event("dkl:refresh-progress")))
  await settle()
  expect(calls()).toBe(3)
})

it("bypasses the inventory cache on an explicit progress refresh", async () => {
  vi.useFakeTimers()
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => response())
  render(<App />)
  await settle()
  act(() => window.dispatchEvent(new Event("dkl:refresh-progress")))
  await settle()
  expect(fetchMock).toHaveBeenCalledWith("/api/notes/progress?refresh=true", expect.anything())
})
