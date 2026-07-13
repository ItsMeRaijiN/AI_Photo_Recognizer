import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/components/ThemeProvider";

export const metadata: Metadata = {
  title: "AI Photo Recognizer — weryfikacja obrazów",
  description: "Analiza obrazów i wykrywanie treści wygenerowanych przez AI.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="pl" suppressHydrationWarning>
      <body>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          storageKey="aipr-theme"
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
