'use client';
import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { AnalysisResult, getModelDisplayName, isMetricSuspicious, MetricResult } from '@/lib/types';
import { api, endpoints } from '@/lib/api';
import {
  AlertTriangle, CheckCircle, Eye, Activity, Clock, Zap, Loader2,
  Ban, Download, ChevronDown, ChevronUp, AlertCircle, Cpu, Check
} from 'lucide-react';
import AuthImage from './AuthImage';

interface Props {
  data: AnalysisResult;
  isGuest?: boolean;
}

export default function AnalysisResultCard({ data, isGuest = false }: Props) {
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);
  const [loadingHeatmap, setLoadingHeatmap] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState('');
  const [opacity, setOpacity] = useState(0.6);
  const [showAllMetrics, setShowAllMetrics] = useState(false);

  useEffect(() => {
    setHeatmapUrl(null);
    setError('');
    setLoadingHeatmap(false);
  }, [data.id, data.filename]);

  const fetchHeatmap = async () => {
    if (heatmapUrl) return;
    setLoadingHeatmap(true);
    setError('');
    try {
      const response = await api.get(endpoints.heatmap(data.id), { responseType: 'blob' });
      const url = URL.createObjectURL(response.data);
      setHeatmapUrl(url);

      if (!isGuest && data.id > 0) {
        try {
          await api.post(endpoints.saveHeatmap(data.id));
        } catch (e) {
          console.warn('Could not auto-save heatmap');
        }
      }
    } catch (e: any) {
      if (e.response?.status === 400) {
        setError("Heatmapa niedostępna dla tego typu modelu.");
      } else {
        setError("Błąd generowania heatmapy.");
      }
    } finally {
      setLoadingHeatmap(false);
    }
  };

  const downloadHeatmap = async () => {
    setDownloading(true);
    try {
      const response = await api.get(endpoints.heatmap(data.id, true), { responseType: 'blob' });
      const url = URL.createObjectURL(response.data);
      const link = document.createElement('a');
      link.href = url;
      link.download = `heatmap_${data.filename}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (e) {
      alert("Błąd pobierania heatmapy.");
    } finally {
      setDownloading(false);
    }
  };

  const scorePercent = (data.score * 100).toFixed(1);
  const thresholdPercent = (data.threshold_used * 100).toFixed(0);
  const isAi = data.is_ai;
  const modelName = getModelDisplayName(data);
  const metrics = data.custom_metrics || {};
  const metricEntries = Object.entries(metrics);
  const displayedMetrics = showAllMetrics ? metricEntries : metricEntries.slice(0, 4);
  const hasMoreMetrics = metricEntries.length > 4;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-8 bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl overflow-hidden shadow-lg"
    >
      <div className={`p-6 border-b border-zinc-200 dark:border-zinc-800 ${isAi ? 'bg-red-50 dark:bg-red-950/20' : 'bg-emerald-50 dark:bg-emerald-950/20'}`}>
        <div className="flex flex-col lg:flex-row lg:justify-between lg:items-start gap-6">
          <div className="flex items-start gap-3">
            {isAi ? <AlertTriangle className="text-red-500 flex-shrink-0" size={36} /> : <CheckCircle className="text-emerald-500 flex-shrink-0" size={36} />}
            <div>
              <h2 className={`text-xl sm:text-2xl font-bold whitespace-nowrap ${isAi ? 'text-red-600 dark:text-red-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                {isAi ? "WYKRYTO AI" : "PRAWDZIWE ZDJĘCIE"}
              </h2>

              <div className="flex flex-wrap gap-2 mt-3">
                <span className="flex items-center gap-1 text-xs font-bold bg-white dark:bg-black/30 px-2 py-1 rounded border border-zinc-200 dark:border-zinc-700">
                  <Cpu size={12} className="text-indigo-500" />
                  {modelName}
                </span>
                <span className="flex items-center gap-1 text-xs text-zinc-600 dark:text-zinc-400 bg-white dark:bg-black/30 px-2 py-1 rounded border border-zinc-200 dark:border-zinc-700">
                  <Clock size={12} />
                  {data.inference_time_ms.toFixed(0)}ms
                </span>
              </div>
            </div>
          </div>

          <div className="lg:min-w-[280px]">
            <div className="flex items-baseline justify-between mb-3">
              <span className="text-sm text-zinc-500 dark:text-zinc-400 font-medium">
                Szansa na AI:
              </span>
              <span className={`text-4xl font-mono font-bold ${isAi ? 'text-red-500' : 'text-emerald-500'}`}>
                {scorePercent}%
              </span>
            </div>

            <div className="relative">
              <div className="h-3 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${data.score * 100}%` }}
                  transition={{ duration: 0.8, ease: "easeOut" }}
                  className={`h-full rounded-full ${
                    isAi 
                      ? 'bg-gradient-to-r from-amber-400 to-red-500' 
                      : 'bg-gradient-to-r from-emerald-400 to-emerald-500'
                  }`}
                />
              </div>

              <div
                className="absolute top-0 h-3 flex flex-col items-center"
                style={{ left: `${data.threshold_used * 100}%`, transform: 'translateX(-50%)' }}
              >
                <div className="w-0.5 h-3 bg-zinc-800 dark:bg-zinc-100" />
              </div>
            </div>

            <div className="relative mt-1.5 h-5">
              <span className="absolute left-0 text-[10px] text-zinc-400">
                0%
              </span>

              <span
                className="absolute text-[10px] font-medium text-zinc-600 dark:text-zinc-300 whitespace-nowrap"
                style={{
                  left: `${data.threshold_used * 100}%`,
                  transform: 'translateX(-50%)'
                }}
              >
                próg {thresholdPercent}%
              </span>

              <span className="absolute right-0 text-[10px] text-zinc-400">
                100%
              </span>
            </div>

            <div className="mt-3 text-[11px] text-zinc-500 dark:text-zinc-400 text-center">
              Zdjęcia są klasyfikowane jako AI po przekroczeniu progu
            </div>
          </div>
        </div>
      </div>

      {metricEntries.length > 0 && (
        <div className="p-6 bg-white dark:bg-black/20 border-b border-zinc-200 dark:border-zinc-800">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider flex items-center gap-2">
              <Activity size={14} /> Analiza Szczegółowa
            </h3>
            {hasMoreMetrics && (
              <button
                onClick={() => setShowAllMetrics(!showAllMetrics)}
                className="text-xs text-indigo-500 hover:text-indigo-400 flex items-center gap-1"
              >
                {showAllMetrics ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                {showAllMetrics ? 'Zwiń' : `Pokaż wszystkie (${metricEntries.length})`}
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {displayedMetrics.map(([key, value]) => (
              <MetricCard key={key} name={key} value={value} />
            ))}
          </div>
        </div>
      )}

      <div className="p-6 bg-zinc-50 dark:bg-zinc-900/50">
        {!heatmapUrl ? (
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={fetchHeatmap}
              disabled={loadingHeatmap || data.id < 0}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition 
                ${data.id < 0
                  ? 'bg-zinc-200 dark:bg-zinc-800/50 text-zinc-400 dark:text-zinc-500 cursor-not-allowed' 
                  : 'bg-zinc-800 hover:bg-zinc-700 text-zinc-200 dark:bg-zinc-800 dark:hover:bg-zinc-700'}
              `}
            >
              {loadingHeatmap ? <Loader2 className="animate-spin" size={16}/> : <Eye size={16}/>}
              {loadingHeatmap ? "Generowanie..." : "Pokaż GradCAM"}
            </button>

            {data.id < 0 && (
              <span className="text-xs text-zinc-500 flex items-center gap-1">
                <Ban size={12}/> Niedostępne dla gości
              </span>
            )}
            {error && <span className="text-xs text-red-500">{error}</span>}
          </div>
        ) : (
          <div className="space-y-4 animate-in fade-in zoom-in duration-300">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-2 flex-wrap">
                <p className="text-sm text-zinc-500 dark:text-zinc-400">Heatmapa Uwagi</p>
                <button
                  onClick={downloadHeatmap}
                  disabled={downloading}
                  className="text-xs bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 px-2 py-1 rounded flex items-center gap-1 hover:bg-indigo-200 dark:hover:bg-indigo-900/50 transition disabled:opacity-50"
                >
                  {downloading ? <Loader2 size={12} className="animate-spin" /> : <Download size={12}/>}
                  Pobierz
                </button>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-zinc-500 font-mono">Transparentność: {Math.round(opacity * 100)}%</span>
                <input
                  type="range"
                  min="0" max="1" step="0.05"
                  value={opacity}
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  className="w-24 accent-indigo-500 h-1 bg-zinc-300 dark:bg-zinc-700 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>

            <div className="relative rounded-lg overflow-hidden border border-zinc-300 dark:border-zinc-700 aspect-video bg-black flex items-center justify-center">
              <div className="absolute inset-0 z-0">
                <AuthImage
                  filePath={data.file_path}
                  analysisId={data.id > 0 ? data.id : undefined}
                  fallbackSrc={data.previewUrl}
                  alt="Original"
                  className="w-full h-full object-contain"
                />
              </div>
              <img
                src={heatmapUrl}
                alt="Heatmap"
                className="absolute inset-0 w-full h-full object-contain z-10 transition-opacity duration-100"
                style={{ opacity: opacity }}
              />
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}

function MetricCard({ name, value }: { name: string; value: MetricResult }) {
  const suspicious = isMetricSuspicious(value);

  let displayValue: string;
  let subValues: { key: string; value: string }[] = [];

  if (value === null || value === undefined) {
    displayValue = 'N/A';
  } else if (typeof value === 'object') {
    const entries = Object.entries(value);
    const mainEntry = entries.find(([k, v]) =>
      typeof v === 'number' && !k.includes('suspicious')
    );

    if (mainEntry) {
      displayValue = formatMetricValue(mainEntry[1]);
    } else {
      displayValue = '-';
    }

    subValues = entries
      .filter(([k]) => k !== 'is_suspicious')
      .slice(0, 3)
      .map(([k, v]) => ({
        key: k.replace(/_/g, ' '),
        value: formatMetricValue(v)
      }));
  } else {
    displayValue = formatMetricValue(value);
  }

  return (
    <div
      className={`p-3 rounded-lg border transition-colors
        ${suspicious 
          ? 'bg-amber-50 dark:bg-amber-900/20 border-amber-300 dark:border-amber-700/50' 
          : 'bg-zinc-100 dark:bg-zinc-800/50 border-zinc-200 dark:border-zinc-700/50'}
      `}
      title={subValues.map(sv => `${sv.key}: ${sv.value}`).join('\n')}
    >
      <div className="flex items-start justify-between gap-1">
        <p className="text-[10px] text-zinc-500 uppercase truncate flex-1" title={name}>
          {name.replace(/_/g, ' ')}
        </p>
        {suspicious && (
          <AlertCircle size={12} className="text-amber-500 flex-shrink-0" />
        )}
      </div>
      <p className={`text-lg font-mono mt-1 ${suspicious ? 'text-amber-600 dark:text-amber-400' : 'text-indigo-600 dark:text-indigo-300'}`}>
        {displayValue}
      </p>
      {subValues.length > 1 && (
        <div className="mt-1 space-y-0.5">
          {subValues.slice(1, 3).map(sv => (
            <p key={sv.key} className="text-[9px] text-zinc-400 truncate">
              {sv.key}: {sv.value}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

function formatMetricValue(value: any): string {
  if (value === null || value === undefined) return 'N/A';
  if (typeof value === 'boolean') return value ? 'Tak' : 'Nie';
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toString();
    return value.toFixed(3);
  }
  return String(value);
}