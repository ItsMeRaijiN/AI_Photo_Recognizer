import "@testing-library/jest-dom";

// JSDOM nie implementuje confirm/alert w pełni (czasem rzuca "Not implemented")
Object.defineProperty(window, "alert", {
  value: jest.fn(),
  writable: true,
});

Object.defineProperty(window, "confirm", {
  value: jest.fn(() => true),
  writable: true,
});

// createObjectURL bywa używane do blobów (np. obrazki)
if (!URL.createObjectURL) {
  // @ts-ignore
  URL.createObjectURL = jest.fn(() => "blob://created");
}
if (!URL.revokeObjectURL) {
  // @ts-ignore
  URL.revokeObjectURL = jest.fn();
}

// (opcjonalnie) jeśli gdzieś pojawi się ResizeObserver
class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
// @ts-ignore
global.ResizeObserver = global.ResizeObserver || MockResizeObserver;

// (opcjonalnie) TextEncoder/TextDecoder (czasem wymagane przez libs)
import { TextEncoder, TextDecoder } from "util";
if (!(global as any).TextEncoder) (global as any).TextEncoder = TextEncoder;
// @ts-ignore
if (!(global as any).TextDecoder) (global as any).TextDecoder = TextDecoder;
