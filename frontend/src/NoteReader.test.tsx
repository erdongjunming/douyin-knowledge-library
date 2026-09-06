import { render, screen, fireEvent } from "@testing-library/react"
import { expect, it, vi } from "vitest"

import { NoteReader } from "./NoteReader"

it("renders complete theme and timeline markdown and scrolls to sections", () => {
  const scrollIntoView = vi.fn()
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: scrollIntoView })
  render(<NoteReader title="示例笔记" markdown={"# 示例\n\n## 主题复盘\n\n主题内容\n\n## 完整逐段时间线\n\n时间线内容"} onClose={vi.fn()} />)

  expect(screen.getByRole("heading", { name: "主题复盘" })).toBeVisible()
  expect(screen.getByRole("heading", { name: "完整逐段时间线" })).toBeVisible()
  fireEvent.click(screen.getByRole("button", { name: "逐段时间线" }))
  expect(scrollIntoView).toHaveBeenCalled()
})

it("does not execute raw html in markdown", () => {
  render(<NoteReader title="安全笔记" markdown={'## 主题复盘\n\n<script>alert("x")</script>'} onClose={vi.fn()} />)
  expect(document.querySelector("script")).toBeNull()
})

it("exposes all teaching sections and degraded quality warnings", () => {
  render(
    <NoteReader
      title="完整笔记"
      qualityStatus="failed"
      warning="输入不足，不能宣称语义完整。"
      markdown={[
        "## 主题复盘",
        "主题内容",
        "## 方法与步骤",
        "步骤内容",
        "## 案例与反例",
        "案例内容",
        "## 前提、边界与风险",
        "边界内容",
        "## 完整逐段时间线",
        "时间线内容",
      ].join("\n\n")}
      onClose={vi.fn()}
    />,
  )

  expect(screen.getByRole("button", { name: "方法与步骤" })).toBeVisible()
  expect(screen.getByRole("button", { name: "案例与反例" })).toBeVisible()
  expect(screen.getByRole("button", { name: "前提、边界与风险" })).toBeVisible()
  expect(screen.getByRole("alert")).toHaveTextContent("输入不足")
})
