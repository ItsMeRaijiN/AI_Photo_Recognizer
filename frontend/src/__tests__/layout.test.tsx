import React from "react";
import { render } from "@testing-library/react";

jest.mock("next/font/google", () => ({
  Geist: () => ({ variable: "--font-geist-sans" }),
  Geist_Mono: () => ({ variable: "--font-geist-mono" }),
}));

jest.mock("@/components/ThemeProvider", () => ({
  ThemeProvider: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="theme-provider">{children}</div>
  ),
}));

import RootLayout from "../app/layout";

const consoleErrorSpy = jest.spyOn(console, "error").mockImplementation(() => {});
afterAll(() => {
  consoleErrorSpy.mockRestore();
});

describe("RootLayout", () => {
  it("should wrap children with ThemeProvider", () => {
    const { getByTestId, getByText } = render(
      <RootLayout>
        <div>Test content</div>
      </RootLayout>
    );

    expect(getByTestId("theme-provider")).toBeInTheDocument();
    expect(getByText("Test content")).toBeInTheDocument();
  });
});