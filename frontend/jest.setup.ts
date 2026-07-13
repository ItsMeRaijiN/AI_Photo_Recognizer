import "@testing-library/jest-dom";
import { TextEncoder, TextDecoder } from "util";

Object.defineProperty(window, "alert", {
  value: jest.fn(),
  writable: true,
});

Object.defineProperty(window, "confirm", {
  value: jest.fn(() => true),
  writable: true,
});

if (!URL.createObjectURL) {
  URL.createObjectURL = jest.fn(() => "blob://created");
}
if (!URL.revokeObjectURL) {
  URL.revokeObjectURL = jest.fn();
}

class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
global.ResizeObserver = global.ResizeObserver || MockResizeObserver;

const globalPolyfills = globalThis as unknown as {
  TextEncoder?: typeof TextEncoder;
  TextDecoder?: typeof TextDecoder;
};
if (!globalPolyfills.TextEncoder) globalPolyfills.TextEncoder = TextEncoder;
if (!globalPolyfills.TextDecoder) globalPolyfills.TextDecoder = TextDecoder;
