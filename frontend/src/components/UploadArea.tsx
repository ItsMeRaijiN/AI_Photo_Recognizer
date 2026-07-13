'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  FileImage, FolderOpen, ImagePlus, Link as LinkIcon,
  Loader2, ShieldCheck, UploadCloud,
} from 'lucide-react';

interface UploadAreaProps {
  onFileSelect: (files: File[]) => void;
  onBatchUpload?: (files: File[]) => Promise<void>;
  isLoading: boolean;
  progress: number;
  progressText?: string;
}

const MAX_FILES = 100;
const MAX_FILE_BYTES = 20 * 1024 * 1024;
const MAX_BATCH_BYTES = 200 * 1024 * 1024;
const SUPPORTED_IMAGE_EXTENSION = /\.(jpg|jpeg|png|webp|bmp|tiff|heic)$/i;
const MIME_TO_EXTENSION: Record<string, string> = {
  'image/jpeg': '.jpg',
  'image/jpg': '.jpg',
  'image/png': '.png',
  'image/webp': '.webp',
  'image/bmp': '.bmp',
  'image/x-ms-bmp': '.bmp',
  'image/tiff': '.tiff',
  'image/heic': '.heic',
  'image/heif': '.heic',
};

export default function UploadArea({
  onFileSelect,
  onBatchUpload,
  isLoading,
  progress,
  progressText,
}: UploadAreaProps) {
  const [url, setUrl] = useState('');
  const [isUrlLoading, setIsUrlLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const processFiles = useCallback((files: File[]) => {
    if (files.length === 0) return;

    const imageFiles = files.filter(file => SUPPORTED_IMAGE_EXTENSION.test(file.name));

    if (imageFiles.length === 0) {
      alert('Nie znaleziono plików obrazów w wybranej lokalizacji.');
      return;
    }
    if (imageFiles.length > MAX_FILES) {
      alert(`Możesz przesłać maksymalnie ${MAX_FILES} obrazów jednocześnie.`);
      return;
    }

    const oversized = imageFiles.find(file => file.size > MAX_FILE_BYTES);
    if (oversized) {
      alert(`Plik „${oversized.name}” przekracza limit 20 MB.`);
      return;
    }
    const totalBytes = imageFiles.reduce((sum, file) => sum + file.size, 0);
    if (totalBytes > MAX_BATCH_BYTES) {
      alert('Łączny rozmiar plików przekracza limit 200 MB. Podziel analizę na mniejsze partie.');
      return;
    }

    if (onBatchUpload && imageFiles.length > 1) {
      void onBatchUpload(imageFiles);
    } else {
      onFileSelect(imageFiles);
    }
  }, [onFileSelect, onBatchUpload]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (isLoading) return;
      const target = event.target as HTMLElement | null;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target?.isContentEditable) return;

      if (event.shiftKey && event.key.toLowerCase() === 't') {
        event.preventDefault();
        fileInputRef.current?.click();
      } else if (event.shiftKey && event.key.toLowerCase() === 'r') {
        event.preventDefault();
        folderInputRef.current?.click();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isLoading]);

  useEffect(() => {
    const handlePaste = (event: ClipboardEvent) => {
      if (isLoading) return;
      const files = Array.from(event.clipboardData?.items || [])
        .filter(item => item.type.startsWith('image/'))
        .map(item => item.getAsFile())
        .filter((file): file is File => Boolean(file));
      if (files.length) processFiles(files);
    };
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [isLoading, processFiles]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: processFiles,
    accept: { 'image/*': [] },
    multiple: true,
    maxFiles: MAX_FILES,
    maxSize: MAX_FILE_BYTES,
    disabled: isLoading,
    noClick: true,
    noKeyboard: true,
  });

  const handleInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    processFiles(Array.from(event.target.files || []));
    event.target.value = '';
  };

  const handleUrlSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!url) return;
    setIsUrlLoading(true);

    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const declaredLength = Number(response.headers?.get?.('content-length') || 0);
      if (declaredLength > MAX_FILE_BYTES) throw new Error('plik przekracza limit 20 MB');

      const blob = await response.blob();
      if (blob.size > MAX_FILE_BYTES) throw new Error('plik przekracza limit 20 MB');
      if (blob.type && !blob.type.startsWith('image/')) throw new Error('adres nie wskazuje na obraz');

      const pathExtension = (() => {
        try {
          return new URL(url).pathname.match(SUPPORTED_IMAGE_EXTENSION)?.[0]?.toLowerCase();
        } catch {
          return undefined;
        }
      })();
      const ext = MIME_TO_EXTENSION[blob.type] || pathExtension;
      if (!ext) throw new Error('nieobsługiwany format obrazu');

      onFileSelect([new File([blob], `url_image_${Date.now()}${ext}`, { type: blob.type || 'image/jpeg' })]);
      setUrl('');
    } catch (error) {
      console.error('URL fetch error:', error);
      const message = error instanceof Error && error.message ? error.message : 'CORS lub niedostępny';
      alert(`Błąd pobierania z URL: ${message}`);
    } finally {
      setIsUrlLoading(false);
    }
  };

  return (
    <section className="overflow-hidden rounded-3xl border border-zinc-200 bg-white shadow-[0_18px_60px_-40px_rgba(24,24,27,0.45)] dark:border-white/10 dark:bg-[#111318]">
      <div className="flex flex-col gap-2 border-b border-zinc-200 px-5 py-5 dark:border-white/8 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-indigo-600 dark:text-indigo-400">Nowa analiza</p>
          <h2 className="mt-1 text-lg font-semibold tracking-tight">Dodaj obraz do weryfikacji</h2>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-zinc-500">
          <ShieldCheck size={14} className="text-emerald-500" /> do 100 plików · 20 MB każdy
        </div>
      </div>

      <div className="space-y-4 p-4 sm:p-6">
        <input ref={fileInputRef} type="file" accept="image/*" multiple onChange={handleInput} className="hidden" disabled={isLoading} />
        <input ref={folderInputRef} type="file" webkitdirectory="" multiple onChange={handleInput} className="hidden" disabled={isLoading} />

        <div
          {...getRootProps()}
          onClick={() => !isLoading && fileInputRef.current?.click()}
          className={`group relative flex min-h-64 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border border-dashed px-5 py-9 text-center transition-all duration-200 sm:min-h-72 ${
            isDragActive
              ? 'border-indigo-500 bg-indigo-50 ring-4 ring-indigo-500/10 dark:bg-indigo-500/10'
              : 'border-zinc-300 bg-zinc-50/70 hover:border-indigo-400 hover:bg-indigo-50/40 dark:border-zinc-700 dark:bg-black/20 dark:hover:border-indigo-500/70 dark:hover:bg-indigo-500/[0.06]'
          } ${isLoading ? 'cursor-not-allowed opacity-65' : ''}`}
        >
          <input {...getInputProps()} />
          <div className="pointer-events-none absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-indigo-500/[0.06] to-transparent" />
          <span className={`relative mb-5 flex h-14 w-14 items-center justify-center rounded-2xl border shadow-sm transition ${
            isDragActive
              ? 'border-indigo-300 bg-indigo-600 text-white'
              : 'border-zinc-200 bg-white text-indigo-600 group-hover:-translate-y-0.5 dark:border-zinc-700 dark:bg-zinc-900 dark:text-indigo-400'
          }`}>
            {isLoading ? <Loader2 size={25} className="animate-spin" /> : <UploadCloud size={25} />}
          </span>

          <p className="relative text-base font-semibold text-zinc-800 dark:text-zinc-100">
            {isDragActive ? 'Upuść pliki tutaj!' : 'Przeciągnij, wybierz lub wklej plik (lub pliki)'}
          </p>
          <p className="relative mt-2 max-w-md text-sm leading-6 text-zinc-500">
            JPG, PNG, WebP i pozostałe popularne formaty. Wiele plików zostanie przeanalizowanych jako jedna partia.
          </p>

          <div className="relative mt-5 flex flex-wrap justify-center gap-2">
            <button
              type="button"
              disabled={isLoading}
              onClick={event => { event.stopPropagation(); fileInputRef.current?.click(); }}
              className="inline-flex items-center gap-2 rounded-lg bg-zinc-900 px-3.5 py-2 text-xs font-medium text-white transition hover:bg-zinc-700 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200"
            >
              <FileImage size={14} /> Wybierz pliki <kbd className="opacity-60">⇧T</kbd>
            </button>
            <button
              type="button"
              disabled={isLoading}
              onClick={event => { event.stopPropagation(); folderInputRef.current?.click(); }}
              className="inline-flex items-center gap-2 rounded-lg border border-zinc-300 bg-white px-3.5 py-2 text-xs font-medium text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
            >
              <FolderOpen size={14} /> Wybierz folder <kbd className="opacity-60">⇧R</kbd>
            </button>
          </div>

          {isLoading && (
            <div className="absolute inset-x-0 bottom-0 border-t border-zinc-200 bg-white/95 px-4 py-3 backdrop-blur dark:border-zinc-800 dark:bg-zinc-900/95">
              <div className="mb-2 flex justify-between gap-4 text-xs text-zinc-500">
                <span className="truncate">{progressText || 'Przetwarzanie…'}</span><span>{progress}%</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
                <div className="h-full rounded-full bg-indigo-500 transition-all" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}
        </div>

        <form onSubmit={handleUrlSubmit} className="flex flex-col gap-2 sm:flex-row">
          <label className="relative flex-1">
            <span className="sr-only">Adres obrazu</span>
            <LinkIcon className="absolute left-3.5 top-3 text-zinc-400" size={16} />
            <input
              type="url"
              placeholder="Wklej link do obrazka..."
              className="h-10 w-full rounded-xl border border-zinc-300 bg-white pl-10 pr-4 text-sm outline-none transition placeholder:text-zinc-400 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/10 dark:border-zinc-700 dark:bg-black/20"
              value={url}
              onChange={event => setUrl(event.target.value)}
              disabled={isLoading || isUrlLoading}
            />
          </label>
          <button
            type="submit"
            disabled={!url || isLoading || isUrlLoading}
            className="inline-flex h-10 items-center justify-center gap-2 rounded-xl border border-zinc-300 bg-zinc-50 px-4 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:bg-zinc-700"
          >
            {isUrlLoading ? <Loader2 size={15} className="animate-spin" /> : <ImagePlus size={15} />} Analizuj
          </button>
        </form>
      </div>
    </section>
  );
}
