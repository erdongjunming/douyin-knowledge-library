import { useCallback, useEffect, useRef } from "react"

type PollingOptions = {
  interval?: number
  refreshEvent?: string
  enabled?: boolean
  immediate?: boolean
  key?: string
}

// One request and one timer per poller. Refresh bursts request one trailing read.
export function usePolling(
  load: (signal: AbortSignal, context: { refresh: boolean }) => Promise<void>,
  { interval = 5000, refreshEvent, enabled = true, immediate = true, key = "" }: PollingOptions = {},
) {
  const loadRef = useRef(load)
  const refreshRef = useRef<() => Promise<void>>(async () => {})
  useEffect(() => { loadRef.current = load })

  useEffect(() => {
    if (!enabled) return
    const controller = new AbortController()
    let timer: number | undefined
    let inFlight: Promise<void> | null = null
    let queued = false
    let queuedRefresh = false
    const run = (refresh = false): Promise<void> => {
      if (controller.signal.aborted) return Promise.resolve()
      if (timer !== undefined) { window.clearTimeout(timer); timer = undefined }
      if (inFlight) { queued = true; queuedRefresh ||= refresh; return inFlight }
      inFlight = (async () => {
        do {
          const forceRefresh = refresh || queuedRefresh
          refresh = false
          queuedRefresh = false
          queued = false
          try { await loadRef.current(controller.signal, { refresh: forceRefresh }) } catch { /* Callers display actionable errors. */ }
        } while (queued && !controller.signal.aborted)
      })().finally(() => {
        inFlight = null
        if (!controller.signal.aborted) timer = window.setTimeout(() => void run(), interval)
      })
      return inFlight
    }
    const requestRefresh = () => run(true)
    refreshRef.current = requestRefresh
    const onRefresh = () => { void requestRefresh() }
    if (refreshEvent) window.addEventListener(refreshEvent, onRefresh)
    if (immediate) void run()
    else timer = window.setTimeout(() => void run(), interval)
    return () => {
      controller.abort()
      if (timer !== undefined) window.clearTimeout(timer)
      if (refreshEvent) window.removeEventListener(refreshEvent, onRefresh)
      if (refreshRef.current === requestRefresh) refreshRef.current = async () => {}
    }
  }, [enabled, immediate, interval, key, refreshEvent])

  return useCallback(() => refreshRef.current(), [])
}
