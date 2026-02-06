'use client';
import { useState, useEffect } from 'react';
import { useTheme } from "next-themes";
import { api, endpoints } from '@/lib/api';
import { AnalysisResult, BatchUploadResponse } from '@/lib/types';
import UploadArea from './UploadArea';
import AnalysisResultCard from './AnalysisResult';
import AdminPanel from './AdminPanel';
import AuthImage from './AuthImage';
import {
  LogOut, History, RefreshCcw, User, Ghost, Sun, Moon,
  ShieldAlert, FileDown, Trash2, Cpu, Layers, AlertCircle, Clock
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

  const [progressPercent, setProgressPercent] = useState(0);
  const [progressText, setProgressText] = useState('');

  const displayHistory = isGuest ? localHistory : dbHistory;

  useEffect(() => {
    setMounted(true);
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
  }, [isGuest]);

  useEffect(() => {
    if (isGuest && mounted) {
      const historyToSave = localHistory.map(({ previewUrl, ...rest }) => rest);
      sessionStorage.setItem('guestHistory', JSON.stringify(historyToSave));
    }
  }, [localHistory, isGuest, mounted]);

  const checkAdmin = async () => {
    try {
      await api.get(endpoints.adminStats);
      setIsAdmin(true);
    } catch {
      setIsAdmin(false);
    }
  };

  const fetchHistory = async () => {
    if (isGuest) return;
    try {
      const res = await api.get(endpoints.history);
      setDbHistory(res.data);
    } catch (e) {
      console.error(e);
    }
  };

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
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await api.post(endpoints.predict, formData);

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

    const blobUrls = new Map<string, string>();
    files.forEach(file => {
      blobUrls.set(file.name, URL.createObjectURL(file));
    });

    const formData = new FormData();
    files.forEach(file => {
      const filename = file.name;
      const cleanFile = new File([file], filename, { type: file.type });
      formData.append('files', cleanFile);
    });

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

      const resultsWithPreviews = response.results.map(result => ({
        ...result,
        previewUrl: blobUrls.get(result.filename)
      }));

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
    } catch (e: any) {
      console.error('Batch upload error:', e);
      let message = 'Błąd batch upload';
      if (e.response?.data?.detail) {
        message = e.response.data.detail;
      } else if (e.code === 'ECONNABORTED') {
        message = 'Timeout - za dużo plików lub wolne połączenie. Spróbuj mniejszą porcję.';
      } else if (e.message) {
        message = e.message;
      }
      alert(`Błąd: ${message}`);
    } finally {
      setLoading(false);
      setProgressText('');
      setProgressPercent(0);
    }
  };

  const handleFolderAnalyze = async (path: string, recursive: boolean) => {
    if (isGuest) {
      alert('Analiza folderów wymaga zalogowania.');
      return;
    }

    setLoading(true);
    setCurrentResult(null);
    setBatchResults(null);
    setProgressPercent(0);
    setShowAdminPanel(false);

    setProgressText(`Analizuję folder: ${path}`);
    setProgressPercent(20);

    try {
      const res = await api.post<BatchUploadResponse>(endpoints.analyzeFolder, {
        path,
        recursive,
        max_images: 100
      });

      setProgressPercent(90);

      const response = res.data;
      setBatchResults(response);

      await fetchHistory();

      if (response.results.length > 0) {
        setCurrentResult(response.results[0]);
      }

      setProgressPercent(100);
    } catch (e: any) {
      console.error('Folder analysis error:', e);
      const message = e.response?.data?.detail || 'Błąd analizy folderu';
      alert(`Błąd: ${message}`);
    } finally {
      setLoading(false);
      setProgressText('');
      setProgressPercent(0);
    }
  };

  const exportCsv = () => {
    if (!displayHistory.length) return;

    const headers = ["filename", "is_ai", "score", "confidence", "backbone", "model_type", "inference_ms", "timestamp"];
    const rows = displayHistory.map(item => [
      item.filename,
      item.is_ai ? "AI" : "REAL",
      item.score.toFixed(4),
      item.confidence.toFixed(4),
      item.backbone_name,
      item.model_type,
      item.inference_time_ms.toFixed(0),
      new Date(item.created_at).toLocaleString()
    ]);

    const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
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
    <div className="min-h-screen bg-zinc-50 dark:bg-black text-zinc-900 dark:text-zinc-100 font-sans transition-colors duration-300">
      <nav className="border-b border-zinc-200 dark:border-zinc-800 bg-white/80 dark:bg-zinc-950/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div
            className="flex items-center gap-2 cursor-pointer"
            onClick={() => { setShowAdminPanel(false); setCurrentResult(null); setBatchResults(null); }}
          >
            <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
            <h1 className="text-lg font-bold tracking-tight">AI Photo Recognizer</h1>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              title="Zmień motyw"
              className="p-2 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors text-zinc-500"
            >
              {theme === 'dark' ? <Moon size={20} /> : <Sun size={20} />}
            </button>
            <div className="h-4 w-[1px] bg-zinc-300 dark:bg-zinc-700"></div>

            {isAdmin && (
              <button
                onClick={() => setShowAdminPanel(!showAdminPanel)}
                title="Panel administratora"
                className={`text-xs font-bold uppercase tracking-wider px-3 py-1.5 rounded transition flex items-center gap-2 
                  ${showAdminPanel 
                    ? 'bg-indigo-500 text-white' 
                    : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 hover:text-indigo-400'}`}
              >
                <ShieldAlert size={14} /> Admin
              </button>
            )}

            <span className="text-sm text-zinc-500 flex items-center gap-2">
              {isGuest ? <Ghost size={14} /> : <User size={14} />}
              <span className="hidden md:inline">{username}</span>
            </span>
            <button
              onClick={onLogoutAction}
              title="Wyloguj"
              className="text-sm text-zinc-500 hover:text-red-500 flex items-center gap-2 transition hover:bg-zinc-100 dark:hover:bg-zinc-800 px-3 py-1.5 rounded-md"
            >
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className={isGuest ? "lg:col-span-3 max-w-3xl mx-auto w-full" : "lg:col-span-2"}>
          {showAdminPanel ? (
            <AdminPanel />
          ) : (
            <>
              <UploadArea
                onFileSelect={handleUpload}
                onBatchUpload={handleBatchUpload}
                onFolderAnalyze={isGuest ? undefined : handleFolderAnalyze}
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
                        key={idx}
                        onClick={() => setCurrentResult(result)}
                        className={`flex items-center justify-between p-2 rounded-lg cursor-pointer transition
                          ${currentResult?.filename === result.filename 
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
                  <AnalysisResultCard data={currentResult} isGuest={isGuest} />
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
          <aside className="lg:h-[calc(100vh-120px)] lg:sticky lg:top-24 flex flex-col bg-white dark:bg-zinc-900/20 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-5 shadow-sm">
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
                  <p className="text-zinc-400 text-sm">Brak wyników.</p>
                </div>
              )}
            </div>
          </aside>
        )}
      </main>
    </div>
  );
}