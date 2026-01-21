'use client';
import { useCallback, useState, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  UploadCloud, Loader2, Link as LinkIcon, ImagePlus,
  FolderOpen, Layers
} from 'lucide-react';

interface UploadAreaProps {
  onFileSelect: (files: File[]) => void;
  onBatchUpload?: (files: File[]) => Promise<void>;
  onFolderAnalyze?: (path: string, recursive: boolean) => Promise<void>;
  isLoading: boolean;
  progress: number;
  progressText?: string;
}

export default function UploadArea({
  onFileSelect,
  onBatchUpload,
  onFolderAnalyze,
  isLoading,
  progress,
  progressText
}: UploadAreaProps) {
  const [url, setUrl] = useState('');
  const [isUrlLoading, setIsUrlLoading] = useState(false);
  const [folderPath, setFolderPath] = useState('');
  const [recursive, setRecursive] = useState(false);
  const [showFolderInput, setShowFolderInput] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (isLoading) return;

      if (e.shiftKey && (e.key === 'T' || e.key === 't')) {
        e.preventDefault();
        fileInputRef.current?.click();
      } else if (e.shiftKey && (e.key === 'R' || e.key === 'r')) {
        e.preventDefault();
        folderInputRef.current?.click();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isLoading]);

  const processFiles = useCallback((files: File[]) => {
    if (files.length === 0) return;

    const imageFiles = files.filter(f =>
      f.type.startsWith('image/') ||
      /\.(jpg|jpeg|png|webp|gif|bmp|tiff|heic)$/i.test(f.name)
    );

    if (imageFiles.length === 0) {
      alert('Nie znaleziono plików obrazów w wybranej lokalizacji.');
      return;
    }

    if (onBatchUpload && imageFiles.length > 1) {
      onBatchUpload(imageFiles);
    } else if (imageFiles.length === 1) {
      onFileSelect(imageFiles);
    } else {
      onFileSelect(imageFiles);
    }
  }, [onFileSelect, onBatchUpload]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    processFiles(acceptedFiles);
  }, [processFiles]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': [] },
    multiple: true,
    disabled: isLoading,
    noClick: true,
    noKeyboard: true
  });

  const handleDropzoneClick = () => {
    if (!isLoading) {
      fileInputRef.current?.click();
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    processFiles(files);
    e.target.value = '';
  };

  const handleFolderInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    processFiles(files);
    e.target.value = '';
  };

  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      if (isLoading) return;
      const items = e.clipboardData?.items;
      if (!items) return;
      const files: File[] = [];
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) files.push(file);
        }
      }
      if (files.length > 0) {
        processFiles(files);
      }
    };
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [processFiles, isLoading]);

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url) return;
    setIsUrlLoading(true);
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const blob = await response.blob();

      const mimeToExt: Record<string, string> = {
        'image/jpeg': '.jpg',
        'image/jpg': '.jpg',
        'image/png': '.png',
        'image/webp': '.webp',
        'image/bmp': '.bmp',
        'image/tiff': '.tiff',
        'image/heic': '.heic',
      };

      const ext = mimeToExt[blob.type] || '.jpg';

      const timestamp = Date.now();
      const filename = `url_image_${timestamp}${ext}`;

      console.log('URL image:', { mime: blob.type, ext, filename, size: blob.size });

      const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
      onFileSelect([file]);
      setUrl('');
    } catch (error: any) {
      console.error('URL fetch error:', error);
      alert(`Błąd pobierania z URL: ${error.message || 'CORS lub niedostępny'}`);
    } finally {
      setIsUrlLoading(false);
    }
  };

  const handleFolderPathSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!folderPath.trim() || !onFolderAnalyze) return;
    await onFolderAnalyze(folderPath.trim(), recursive);
  };

  return (
    <div className="space-y-4">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        onChange={handleFileInputChange}
        className="hidden"
        disabled={isLoading}
      />
      <input
        ref={folderInputRef}
        type="file"
        // @ts-ignore - webkitdirectory is not in types but works
        webkitdirectory=""
        multiple
        onChange={handleFolderInputChange}
        className="hidden"
        disabled={isLoading}
      />

      <div
        {...getRootProps()}
        onClick={handleDropzoneClick}
        className={`
          relative overflow-hidden rounded-2xl border-2 border-dashed p-8 text-center cursor-pointer transition-all duration-300
          flex flex-col items-center justify-center gap-4 group bg-zinc-50/50 dark:bg-zinc-900/50
          ${isDragActive 
            ? 'border-indigo-500 bg-indigo-500/10 scale-[1.01]' 
            : 'border-zinc-300 dark:border-zinc-700 hover:border-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800/80'}
          ${isLoading ? 'opacity-50 cursor-not-allowed grayscale' : ''}
        `}
      >
        <input {...getInputProps()} />

        <div className={`p-4 rounded-full bg-zinc-100 dark:bg-zinc-800 transition-all duration-300 ${
          isDragActive 
            ? 'bg-indigo-500/20 text-indigo-400' 
            : 'text-zinc-400 group-hover:text-zinc-600 dark:group-hover:text-zinc-200'
        }`}>
          {isLoading
            ? <Loader2 size={32} className="animate-spin text-indigo-400" />
            : <UploadCloud size={32} />
          }
        </div>

        <div className="z-10">
          <p className="text-lg font-medium text-zinc-700 dark:text-zinc-200">
            {isDragActive ? "Upuść pliki tutaj!" : "Przeciągnij, wybierz lub wklej plik (lub pliki)"}
          </p>
          <div className="flex items-center justify-center gap-4 mt-2 text-xs text-zinc-500">
            <span className="flex items-center gap-1">
              <span className="bg-zinc-200 dark:bg-zinc-800 px-1.5 py-0.5 rounded border border-zinc-300 dark:border-zinc-700 font-mono">
                Shift + T
              </span>
              pliki
            </span>
            <span className="flex items-center gap-1">
              <span className="bg-zinc-200 dark:bg-zinc-800 px-1.5 py-0.5 rounded border border-zinc-300 dark:border-zinc-700 font-mono">
                Shift + R
              </span>
              folder
            </span>
          </div>
        </div>

        {isLoading && (
          <div className="absolute bottom-0 left-0 w-full">
            <div className="h-1.5 bg-zinc-200 dark:bg-zinc-800">
              <div
                className="h-full bg-indigo-500 transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
            {progressText && (
              <div className="bg-zinc-100 dark:bg-zinc-800 px-3 py-1 text-xs text-zinc-600 dark:text-zinc-400 font-mono">
                {progressText}
              </div>
            )}
          </div>
        )}
      </div>

      {onFolderAnalyze && (
        <div className="flex gap-2">
          <button
            onClick={() => setShowFolderInput(!showFolderInput)}
            disabled={isLoading}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition border
              ${showFolderInput 
                ? 'bg-indigo-500 text-white border-indigo-500' 
                : 'bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-600 dark:text-zinc-400 border-zinc-200 dark:border-zinc-700'}
              disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <FolderOpen size={14} />
            Analiza ścieżki
          </button>
        </div>
      )}

      {showFolderInput && onFolderAnalyze && (
        <form onSubmit={handleFolderPathSubmit} className="space-y-3 p-4 bg-zinc-100 dark:bg-zinc-900/50 rounded-xl border border-zinc-200 dark:border-zinc-800">
          <p className="text-xs text-zinc-500 mb-2">
            Podaj ścieżkę do <strong>folderu</strong> lub <strong>pliku</strong>
          </p>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <FolderOpen className="absolute left-3 top-2.5 text-zinc-500" size={16} />
              <input
                type="text"
                placeholder="C:\\Users\\User\\Pictures lub C:\\Users\\User\\image.jpg"
                className="w-full bg-white dark:bg-black/50 border border-zinc-300 dark:border-zinc-700 rounded-lg py-2 pl-10 pr-4 text-sm text-zinc-900 dark:text-white focus:border-indigo-500 outline-none transition font-mono"
                value={folderPath}
                onChange={e => setFolderPath(e.target.value)}
                disabled={isLoading}
              />
            </div>
            <button
              type="submit"
              disabled={!folderPath.trim() || isLoading}
              className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50 transition flex items-center gap-2"
            >
              <Layers size={16} />
              Analizuj
            </button>
          </div>
          <label className="flex items-center gap-2 text-xs text-zinc-500 cursor-pointer">
            <input
              type="checkbox"
              checked={recursive}
              onChange={e => setRecursive(e.target.checked)}
              className="rounded border-zinc-300 text-indigo-500 focus:ring-indigo-500"
              disabled={isLoading}
            />
            Skanuj podfoldery rekurencyjnie
          </label>
        </form>
      )}

      <form onSubmit={handleUrlSubmit} className="flex gap-2">
        <div className="relative flex-1">
          <LinkIcon className="absolute left-3 top-2.5 text-zinc-500" size={16} />
          <input
            type="url"
            placeholder="Wklej link do obrazka..."
            className="w-full bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-lg py-2 pl-9 pr-4 text-sm text-zinc-900 dark:text-white focus:border-indigo-500 outline-none transition"
            value={url}
            onChange={e => setUrl(e.target.value)}
            disabled={isLoading || isUrlLoading}
          />
        </div>
        <button
          type="submit"
          disabled={!url || isLoading || isUrlLoading}
          className="bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-900 dark:text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50 transition flex items-center gap-2 border border-zinc-200 dark:border-zinc-700"
        >
          {isUrlLoading ? <Loader2 size={14} className="animate-spin" /> : <ImagePlus size={16} />}
          Analizuj
        </button>
      </form>
    </div>
  );
}