'use client';
import { useState, useEffect, useCallback, useRef } from 'react';
import { useTheme } from "next-themes";
import { api, endpoints } from '@/lib/api';
import { getErrorMessage } from '@/lib/errors';
import { AnalysisResult, BatchUploadResponse, ModelInfo } from '@/lib/types';
import UploadArea from './UploadArea';
import AnalysisResultCard from './AnalysisResult';
import AdminPanel from './AdminPanel';
import AuthImage from './AuthImage';
import {
  LogOut, History, RefreshCcw, User, Ghost, Sun, Moon,
  ShieldAlert, FileDown, Trash2, Cpu, Layers, AlertCircle, Clock,
  ScanSearch, CircleCheck, Gauge
} from 'lucide-react';

interface DashboardProps {
  onLogoutAction: () => void;
  isGuest: boolean;
}

export default function Dashboard({ onLogoutAction, isGuest }: DashboardProps) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  const [dbHistory, setDbHistory] = useState<AnalysisResult[]>([]);
  const [localHistory, setLocalHistory] = useState<AnalysisResult[]>([]);
  const [currentResult, setCurrentResult] = useState<AnalysisResult | null>(null);
  const [batchResults, setBatchResults] = useState<BatchUploadResponse | null>(null);

  const [loading, setLoading] = useState(false);
  const [username, setUsername] = useState('');
  const [isAdmin, setIsAdmin] = useState(false);
  const [showAdminPanel, setShowAdminPanel] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const ownedPreviewUrls = useRef(new Set<string>());

  const [progressPercent, setProgressPercent] = useState(0);
  const [progressText, setProgressText] = useState('');

  const displayHistory = isGuest ? localHistory : dbHistory;

  const checkAdmin = useCallback(async () => {
    try {
      await api.get(endpoints.adminStats);
      setIsAdmin(true);
    } catch {
      setIsAdmin(false);
    }
  }, []);

  const fetchHistory = useCallback(async () => {
    if (isGuest) return;
    try {
      const res = await api.get(endpoints.history);
      setDbHistory(res.data);
    } catch (e) {
      console.error(e);
    }
  }, [isGuest]);

  const fetchModelInfo = useCallback(async () => {
    try {
      const res = await api.get(endpoints.modelInfo);
      if (typeof res.data?.loaded === 'boolean') setModelInfo(res.data);
    } catch {
      setModelInfo(null);
    }
  }, []);

  useEffect(() => {
    setMounted(true);
    fetchModelInfo();
    if (isGuest) {
      setUsername('Gość');
      try {
        const saved = sessionStorage.getItem('guestHistory');
        if (saved) {
          setLocalHistory(JSON.parse(saved));
        }
      } catch (e) {
        console.error("Błąd odczytu historii gościa", e);
      }
    } else {
      setUsername(localStorage.getItem('username') || 'Użytkownik');
      fetchHistory();
      checkAdmin();
    }
  }, [isGuest, fetchHistory, checkAdmin, fetchModelInfo]);

  useEffect(() => {
    const urls = ownedPreviewUrls.current;
    return () => {
      urls.forEach(url => URL.revokeObjectURL(url));
      urls.clear();
    };
  }, []);

  useEffect(() => {
    if (loading) return;
    const referencedUrls = new Set<string>();
    const collect = (result: AnalysisResult | null | undefined) => {
      if (result?.previewUrl) referencedUrls.add(result.previewUrl);
    };
    collect(currentResult);
    localHistory.forEach(collect);
    dbHistory.forEach(collect);
    batchResults?.results.forEach(collect);

    ownedPreviewUrls.current.forEach(url => {
      if (!referencedUrls.has(url)) {
        URL.revokeObjectURL(url);
        ownedPreviewUrls.current.delete(url);
      }
    });
  }, [currentResult, localHistory, dbHistory, batchResults, loading]);

  useEffect(() => {
    if (isGuest && mounted) {
      const historyToSave = localHistory.map(item => {
        const copy = { ...item };
        delete copy.previewUrl;
        return copy;
      });
      sessionStorage.setItem('guestHistory', JSON.stringify(historyToSave));
    }
  }, [localHistory, isGuest, mounted]);

  const handleClearHistory = () => {
    if (isGuest) {
      if (confirm("Wyczyścić historię?")) {
        setLocalHistory([]);
        sessionStorage.removeItem('guestHistory');
      }
    } else {
      fetchHistory();
    }
  };

  const handleSelectHistoryItem = (item: AnalysisResult) => {
    setCurrentResult(item);
    setBatchResults(null);
    setShowAdminPanel(false);
  };

  const handleUpload = async (files: File[]) => {
    if (files.length > 1) {
      return handleBatchUpload(files);
    }

    setLoading(true);
    setCurrentResult(null);
    setBatchResults(null);
    setProgressPercent(0);
    setShowAdminPanel(false);

    const file = files[0];
    setProgressText(`Analizuję: ${file.name}`);

    const blobUrl = URL.createObjectURL(file);
    ownedPreviewUrls.current.add(blobUrl);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await api.post(endpoints.predict, formData, {
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percent = Math.round((progressEvent.loaded / progressEvent.total) * 90);
            setProgressPercent(Math.min(percent, 90));
          }
        }
      });

      const newResult: AnalysisResult = {
        ...res.data,
        previewUrl: blobUrl
      };

      setCurrentResult(newResult);

      if (isGuest) {
        setLocalHistory(prev => [newResult, ...prev].slice(0, 20));
      } else {
        setDbHistory(prev => [newResult, ...prev].slice(0, 100));
      }

      setProgressPercent(100);
    } catch (e) {
      URL.revokeObjectURL(blobUrl);
      ownedPreviewUrls.current.delete(blobUrl);
      console.error(`Błąd przy pliku ${file.name}`, e);
      alert(`Błąd analizy: ${file.name}`);
    }

    if (!isGuest) await fetchHistory();
    setLoading(false);
    setProgressText('');
    setProgressPercent(0);
  };

  const handleBatchUpload = async (files: File[]) => {
    setLoading(true);
    setCurrentResult(null);
    setBatchResults(null);
    setProgressPercent(0);
    setShowAdminPanel(false);

    setProgressText(`Przygotowuję ${files.length} plików...`);

    const blobUrls = new Map<number, string>();
    files.forEach((file, index) => {
      const objectUrl = URL.createObjectURL(file);
      blobUrls.set(index, objectUrl);
      ownedPreviewUrls.current.add(objectUrl);
    });

    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    setProgressText(`Wysyłam ${files.length} plików...`);
    setProgressPercent(30);

    try {
      const res = await api.post<BatchUploadResponse>(endpoints.predictBatch, formData, {
        timeout: 300000,  // 5 minutes
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percent = Math.round((progressEvent.loaded / progressEvent.total) * 50) + 30;
            setProgressPercent(Math.min(percent, 80));
          }
        }
      });

      setProgressPercent(90);
      setProgressText('Przetwarzam wyniki...');

      const response = res.data;

      const claimedIndexes = new Set<number>();
      const resultsWithPreviews = response.results.map(result => {
        const fallbackIndex = files.findIndex((file, index) =>
          !claimedIndexes.has(index) && file.name === result.filename
        );
        const sourceIndex = result.source_index ?? fallbackIndex;
        if (sourceIndex >= 0) claimedIndexes.add(sourceIndex);
        return { ...result, previewUrl: blobUrls.get(sourceIndex) };
      });

      blobUrls.forEach((objectUrl, index) => {
        if (!claimedIndexes.has(index)) {
          URL.revokeObjectURL(objectUrl);
          ownedPreviewUrls.current.delete(objectUrl);
        }
      });

      setBatchResults({
        ...response,
        results: resultsWithPreviews
      });

      if (isGuest) {
        setLocalHistory(prev => [...resultsWithPreviews, ...prev].slice(0, 20));
      } else {
        await fetchHistory();
      }

      if (resultsWithPreviews.length > 0) {
        setCurrentResult(resultsWithPreviews[0]);
      }

      setProgressPercent(100);
    } catch (e) {
      blobUrls.forEach(objectUrl => {
        URL.revokeObjectURL(objectUrl);
        ownedPreviewUrls.current.delete(objectUrl);
      });
      console.error('Batch upload error:', e);
      alert(`Błąd: ${getErrorMessage(e, 'Błąd analizy wielu plików')}`);
    } finally {
      setLoading(false);
      setProgressText('');
      setProgressPercent(0);
    }
  };

  const exportCsv = () => {
    if (!displayHistory.length) return;

    const escapeCsv = (value: string | number): string => {
      const s = String(value);
      return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };

    const headers = ["filename", "is_ai", "score", "confidence", "backbone", "model_type", "inference_ms", "timestamp"];
    const rows = displayHistory.map(item => [
      item.filename,
      item.is_ai ? "AI" : "REAL",
      item.score.toFixed(4),
      item.confidence.toFixed(4),
      item.backbone_name,
      item.model_type,
      item.inference_time_ms.toFixed(0),
      new Date(item.created_at).toISOString()
    ].map(escapeCsv));

    const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\r\n");
    const blob = new Blob(["﻿" + csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "historia_analiz.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  if (!mounted) return null;

  return (
    <div className="min-h-screen bg-[#f4f6f8] text-zinc-900 transition-colors duration-300 dark:bg-[#080a0f] dark:text-zinc-100">
      <nav className="sticky top-0 z-50 border-b border-zinc-200/80 bg-white/85 backdrop-blur-xl dark:border-white/8 dark:bg-[#0b0d12]/85">
        <div className="mx-auto flex h-[68px] max-w-[1440px] items-center justify-between px-4 sm:px-6 lg:px-8">
          <div
            className="flex cursor-pointer items-center gap-3"
            onClick={() => { setShowAdminPanel(false); setCurrentResult(null); setBatchResults(null); }}
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-md shadow-indigo-600/20">
              <ScanSearch size={19} />
            </span>
            <div>
              <h1 className="text-sm font-semibold tracking-tight sm:text-base">AI Photo Recognizer</h1>
            </div>
          </div>

          <div className="flex items-center gap-1.5 sm:gap-3">
            <button
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              title={theme === 'dark' ? 'Przełącz na jasny motyw' : 'Przełącz na ciemny motyw'}
              className="rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-100 dark:hover:bg-zinc-800"
            >
              {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
            </button>
            <div className="mx-1 h-5 w-px bg-zinc-200 dark:bg-zinc-800" />

            {isAdmin && (
              <button
                onClick={() => setShowAdminPanel(!showAdminPanel)}
                title="Panel administratora"
                className={`flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold transition
                  ${showAdminPanel
                    ? 'bg-indigo-600 text-white'
                    : 'bg-zinc-100 text-zinc-600 hover:text-indigo-600 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:text-indigo-400'}`}
              >
                <ShieldAlert size={14} /> Admin
              </button>
            )}

            <span className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-zinc-500">
              {isGuest ? <Ghost size={14} /> : <User size={14} />}
              <span className="hidden md:inline">{username}</span>
            </span>
            <button
              onClick={onLogoutAction}
              title="Wyloguj"
              className="flex items-center gap-2 rounded-lg p-2 text-sm text-zinc-500 transition hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-500/10 dark:hover:text-red-400 sm:px-3"
            >
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </nav>

      <main className="mx-auto max-w-[1440px] px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <section className="mb-6 flex flex-col gap-5 rounded-3xl border border-zinc-200 bg-white px-5 py-5 shadow-sm dark:border-white/8 dark:bg-[#111318] sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <div>
            <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em] sm:text-2xl">
              {showAdminPanel ? 'Centrum administracyjne' : 'Weryfikacja autentyczności obrazu'}
            </h2>
            <p className="mt-1 text-sm text-zinc-500">
              {showAdminPanel ? 'Stan systemu, użytkownicy i konfiguracja modelu.' : 'Prześlij obraz, aby ocenić prawdopodobieństwo wygenerowania przez AI.'}
            </p>
          </div>
          <div className="flex items-center gap-2 self-start sm:self-auto">
            <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium ${
              modelInfo?.loaded
                ? 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-500/20 dark:bg-emerald-500/10 dark:text-emerald-300'
                : 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-500/20 dark:bg-amber-500/10 dark:text-amber-300'
            }`}>
              <span className={`h-1.5 w-1.5 rounded-full ${modelInfo?.loaded ? 'bg-emerald-500' : 'bg-amber-500'}`} />
              {modelInfo?.loaded ? 'Model gotowy' : 'Status modelu nieznany'}
            </span>
            {modelInfo?.loaded && (
              <span className="hidden items-center gap-1.5 rounded-full border border-zinc-200 px-3 py-1.5 text-xs text-zinc-500 dark:border-zinc-700 sm:inline-flex">
                <Gauge size={13} /> {modelInfo.backbone || modelInfo.type}
              </span>
            )}
          </div>
        </section>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3 lg:gap-7">
        <div className="w-full lg:col-span-2">
          {showAdminPanel ? (
            <AdminPanel />
          ) : (
            <>
              <UploadArea
                onFileSelect={handleUpload}
                onBatchUpload={handleBatchUpload}
                isLoading={loading}
                progress={progressPercent}
                progressText={progressText}
              />

              {loading && (
                <div className="mt-6 space-y-2">
                  <div className="flex justify-between text-xs text-zinc-500 font-mono">
                    <span>{progressText}</span>
                    <span>{progressPercent}%</span>
                  </div>
                  <div className="h-2 w-full bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-indigo-500 transition-all duration-300 ease-out"
                      style={{ width: `${progressPercent}%` }}
                    />
                  </div>
                </div>
              )}

              {batchResults && !loading && (
                <div className="mt-6 p-4 bg-zinc-100 dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 flex items-center gap-2">
                      <Layers size={16} className="text-indigo-500" />
                      Wyniki Batch ({batchResults.total} plików)
                    </h3>
                    <span className="text-xs text-zinc-500 flex items-center gap-1">
                      <Clock size={12} />
                      {(batchResults.total_inference_time_ms / 1000).toFixed(2)}s
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div className="text-center p-3 bg-white dark:bg-black/30 rounded-lg">
                      <div className="text-2xl font-bold text-emerald-500">{batchResults.processed}</div>
                      <div className="text-xs text-zinc-500">Przetworzono</div>
                    </div>
                    <div className="text-center p-3 bg-white dark:bg-black/30 rounded-lg">
                      <div className="text-2xl font-bold text-red-500">
                        {batchResults.results.filter(r => r.is_ai).length}
                      </div>
                      <div className="text-xs text-zinc-500">Wykryto AI</div>
                    </div>
                    <div className="text-center p-3 bg-white dark:bg-black/30 rounded-lg">
                      <div className="text-2xl font-bold text-amber-500">{batchResults.failed}</div>
                      <div className="text-xs text-zinc-500">Błędów</div>
                    </div>
                  </div>

                  <div className="max-h-48 overflow-y-auto space-y-2">
                    {batchResults.results.map((result, idx) => (
                      <div
                        key={`${result.source_index ?? idx}-${result.id}-${result.filename}`}
                        onClick={() => setCurrentResult(result)}
                        className={`flex items-center justify-between p-2 rounded-lg cursor-pointer transition
                          ${currentResult?.source_index === result.source_index && currentResult?.filename === result.filename
                            ? 'bg-indigo-100 dark:bg-indigo-900/30 border border-indigo-300 dark:border-indigo-700' 
                            : 'bg-white dark:bg-black/20 hover:bg-zinc-50 dark:hover:bg-zinc-800/50'}`}
                      >
                        <span className="text-sm truncate flex-1 text-zinc-700 dark:text-zinc-300">
                          {result.filename}
                        </span>
                        <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                          result.is_ai 
                            ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400' 
                            : 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400'
                        }`}>
                          {result.is_ai ? 'AI' : 'REAL'}
                        </span>
                      </div>
                    ))}
                  </div>

                  {batchResults.errors.length > 0 && (
                    <div className="mt-3 p-2 bg-red-50 dark:bg-red-900/20 rounded-lg">
                      <div className="text-xs font-semibold text-red-600 dark:text-red-400 mb-1 flex items-center gap-1">
                        <AlertCircle size={12} /> Błędy ({batchResults.errors.length})
                      </div>
                      <div className="text-xs text-red-500 space-y-0.5 max-h-20 overflow-y-auto">
                        {batchResults.errors.slice(0, 5).map((err, i) => (
                          <div key={i} className="truncate">{err}</div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {currentResult && !loading && (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 mt-8">
                  <div className="mb-6 flex justify-center bg-zinc-100 dark:bg-zinc-900/50 rounded-xl p-4 border border-zinc-200 dark:border-zinc-800">
                    <AuthImage
                      filePath={currentResult.file_path}
                      analysisId={currentResult.id > 0 ? currentResult.id : undefined}
                      fallbackSrc={currentResult.previewUrl}
                      alt="Analyzed"
                      className="max-h-[400px] w-auto object-contain rounded-lg shadow-sm"
                    />
                  </div>
                  <AnalysisResultCard data={currentResult} />
                  {isGuest && (
                    <p className="text-center text-xs text-zinc-400 mt-4 opacity-70">
                      * Tryb Gościa: Wyniki są tymczasowe.
                    </p>
                  )}
                </div>
              )}

            </>
          )}
        </div>

        {!showAdminPanel && (
          <aside className="flex flex-col rounded-3xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-white/8 dark:bg-[#111318] lg:sticky lg:top-24 lg:h-[calc(100vh-124px)]">
            <div className="flex items-center justify-between mb-4 pb-4 border-b border-zinc-200 dark:border-zinc-800">
              <h3 className="font-semibold flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300">
                <History className="text-indigo-500" size={18} />
                Historia
              </h3>
              <div className="flex gap-1">
                <button
                  onClick={exportCsv}
                  title="Pobierz CSV"
                  className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-full transition text-zinc-400 hover:text-indigo-500"
                >
                  <FileDown size={16} />
                </button>
                <button
                  onClick={handleClearHistory}
                  title={isGuest ? "Wyczyść historię" : "Odśwież"}
                  className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-full transition text-zinc-400 hover:text-red-500"
                >
                  {isGuest ? <Trash2 size={16} /> : <RefreshCcw size={16} />}
                </button>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
              {displayHistory.map((item, idx) => (
                <div
                  key={isGuest ? `guest-${idx}` : `${item.id}-${idx}`}
                  onClick={() => handleSelectHistoryItem(item)}
                  className={`
                    flex gap-3 p-3 rounded-lg border cursor-pointer transition-all hover:bg-zinc-50 dark:hover:bg-zinc-800/80 group
                    ${currentResult?.id === item.id && currentResult?.filename === item.filename
                      ? 'bg-zinc-100 dark:bg-zinc-800 border-zinc-300 dark:border-zinc-600 ring-1 ring-zinc-300 dark:ring-zinc-600'
                      : 'bg-white dark:bg-black/20 border-zinc-200 dark:border-zinc-800'}
                  `}
                >
                  <div className="w-14 h-14 rounded-md bg-zinc-100 dark:bg-zinc-900 flex-shrink-0 overflow-hidden border border-zinc-200 dark:border-zinc-700 flex items-center justify-center relative">
                    <AuthImage
                      filePath={item.file_path}
                      analysisId={item.id > 0 ? item.id : undefined}
                      fallbackSrc={item.previewUrl}
                      alt="thumb"
                      className="w-full h-full object-cover"
                    />
                  </div>

                  <div className="flex-1 min-w-0 flex flex-col justify-between py-0.5">
                    <div className="flex justify-between items-start">
                      <span
                        className="text-xs font-medium truncate w-24 text-zinc-700 dark:text-zinc-300 group-hover:text-zinc-900 dark:group-hover:text-white"
                        title={item.filename}
                      >
                        {item.filename}
                      </span>
                      <span className="text-[10px] text-zinc-400 font-mono">
                        {new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>

                    <div className="flex items-center justify-between mt-1">
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider ${
                        item.is_ai 
                          ? 'bg-red-100 dark:bg-red-500/20 text-red-600 dark:text-red-400' 
                          : 'bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400'
                      }`}>
                        {item.is_ai ? 'AI' : 'REAL'}
                      </span>
                      <span className={`text-xs font-bold font-mono ${
                        item.is_ai ? 'text-red-500' : 'text-emerald-500'
                      }`}>
                        {(item.score * 100).toFixed(0)}%
                      </span>
                    </div>

                    <div className="flex items-center gap-1 mt-1">
                      <Cpu size={10} className="text-zinc-400" />
                      <span className="text-[9px] text-zinc-400 truncate">
                        {item.backbone_name || 'unknown'}
                      </span>
                    </div>
                  </div>
                </div>
              ))}

              {displayHistory.length === 0 && (
                <div className="text-center mt-10">
                  <span className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-zinc-100 text-zinc-400 dark:bg-zinc-800"><CircleCheck size={18} /></span>
                  <p className="text-zinc-400 text-sm">Brak wyników.</p>
                  <p className="mx-auto mt-1 max-w-[180px] text-xs leading-5 text-zinc-400">Pierwsza ukończona analiza pojawi się w tym miejscu.</p>
                </div>
              )}
            </div>
          </aside>
        )}
        </div>
      </main>
    </div>
  );
}
