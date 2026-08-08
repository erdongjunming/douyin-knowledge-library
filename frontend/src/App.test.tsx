import { render, screen } from "@testing-library/react"
import { expect, it } from "vitest"

import { App } from "./App"

it("shows the local knowledge library identity", () => {
  render(<App />)

  expect(screen.getByRole("heading", { name: "抖音知识资料库" })).toBeVisible()
  expect(screen.getByText("本地服务已连接")).toBeVisible()
})
