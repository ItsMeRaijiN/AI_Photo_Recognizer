import 'react';

declare module 'react' {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    /** Non-standard Chromium attribute enabling directory selection in file inputs. */
    webkitdirectory?: string;
  }
}
