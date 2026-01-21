'use client';
import React, { useState, useEffect, useRef } from 'react';
import { api, endpoints } from '@/lib/api';
import { User, SystemStats } from '@/lib/types';
import {
  Trash2, Users, BarChart, Cpu, HardDrive,
  RefreshCw, Calendar, Loader2, Settings, Database, Zap,
  AlertTriangle, CheckCircle, Upload, Activity, ShieldCheck,
  ShieldOff, UserMinus
} from 'lucide-react';

type TabType = 'dashboard' | 'users' | 'model' | 'settings';

export default function AdminPanel() {
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [users, setUsers] = useState<User[]>([]);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<number | string | null>(null);
  const [apiVersion, setApiVersion] = useState<string>('...');
  const modelInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [usersRes, statsRes, rootRes] = await Promise.all([
        api.get(endpoints.adminUsers),
        api.get(endpoints.adminStats),
        api.get(endpoints.root).catch(() => ({ data: { version: 'unknown' } }))
      ]);
      setUsers(usersRes.data);
      setStats(statsRes.data);
      setApiVersion(rootRes.data.version || 'unknown');
    } catch (e) {
      console.error("Admin fetch error", e);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteUser = async (id: number, username: string) => {
    if (!confirm(`Czy na pewno usunąć użytkownika "${username}"? Ta operacja jest nieodwracalna i usunie wszystkie jego analizy.`)) return;
    setActionLoading(id);
    try {
      await api.delete(endpoints.adminUser(id));
      setUsers(users.filter(u => u.id !== id));
    } catch (e: any) {
      alert(e.response?.data?.detail || "Nie można usunąć użytkownika");
    } finally {
      setActionLoading(null);
    }
  };

  const handleToggleAdmin = async (id: number, currentlyAdmin: boolean) => {
    const action = currentlyAdmin ? 'odebrać uprawnienia admina' : 'nadać uprawnienia admina';
    if (!confirm(`Czy na pewno chcesz ${action}?`)) return;
    setActionLoading(id);
    try {
      await api.patch(endpoints.adminToggleAdmin(id));
      setUsers(users.map(u =>
        u.id === id ? { ...u, is_superuser: !u.is_superuser } : u
      ));
    } catch (e: any) {
      alert(e.response?.data?.detail || "Błąd zmiany uprawnień");
    } finally {
      setActionLoading(null);
    }
  };

  const handleCleanup = async () => {
    if (!confirm("Uruchomić czyszczenie systemu? Usunie stare pliki.")) return;
    setActionLoading('cleanup');
    try {
      const res = await api.post(endpoints.adminCleanup);
      alert(`Wyczyszczono:\n- Pliki temp: ${res.data.deleted_temp_files}\n- Stare zadania: ${res.data.deleted_old_jobs}\n- Osierocone analizy: ${res.data.deleted_orphan_analyses}`);
      await fetchData();
    } catch (e) {
      alert("Błąd czyszczenia");
    } finally {
      setActionLoading(null);
    }
  };

  const handleReloadMetrics = async () => {
    setActionLoading('metrics');
    try {
      await api.post(endpoints.adminReloadMetrics);
      alert("Metryki przeładowane!");
      await fetchData();
    } catch (e) {
      alert("Błąd przeładowania metryk");
    } finally {
      setActionLoading(null);
    }
  };

  const handleOptimizeDb = async () => {
    setActionLoading('optimize');
    try {
      const res = await api.post(endpoints.adminOptimizeDb);
      alert(`Baza zoptymalizowana!\nRozmiar przed: ${res.data.size_before}\nRozmiar po: ${res.data.size_after}`);
    } catch (e: any) {
      alert(e.response?.data?.detail || "Błąd optymalizacji");
    } finally {
      setActionLoading(null);
    }
  };

  const handleModelUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.pt') && !file.name.endsWith('.onnx')) {
      alert('Model musi być plikiem .pt lub .onnx');
      return;
    }

    if (!confirm(`Wgrać nowy model "${file.name}"? Wymaga restartu serwera.`)) {
      e.target.value = '';
      return;
    }

    setActionLoading('model');
    const formData = new FormData();
    formData.append('model', file);

    try {
      const res = await api.post(endpoints.adminUploadModel, formData);
      alert(`Model wgrany do: ${res.data.path}\n\nZrestartuj serwer aby załadować nowy model.`);
      e.target.value = '';
    } catch (err: any) {
      alert(err.response?.data?.detail || "Błąd wgrywania modelu");
    } finally {
      setActionLoading(null);
    }
  };

  const tabs: { id: TabType; label: string; icon: React.ReactNode }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <BarChart size={16} /> },
    { id: 'users', label: 'Użytkownicy', icon: <Users size={16} /> },
    { id: 'model', label: 'Model AI', icon: <Cpu size={16} /> },
    { id: 'settings', label: 'Ustawienia', icon: <Settings size={16} /> },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="animate-spin text-indigo-500" size={32} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex gap-2 flex-wrap">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition ${
              activeTab === tab.id
                ? 'bg-indigo-500 text-white'
                : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700'
            }`}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'dashboard' && stats && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard icon={<Users size={16} />} label="Użytkownicy" value={stats.total_users} subValue={`${stats.active_users} aktywnych`} />
            <StatCard icon={<HardDrive size={16} />} label="Analiz" value={stats.total_analyses} subValue={`${stats.ai_detections} wykrytych AI`} />
            <StatCard icon={<AlertTriangle size={16} />} label="Wykrywalność AI" value={`${stats.total_analyses > 0 ? ((stats.ai_detections / stats.total_analyses) * 100).toFixed(1) : 0}%`} color={stats.ai_detections > 0 ? 'red' : 'default'} />
            <StatCard icon={<CheckCircle size={16} />} label="Model" value={stats.model_info?.loaded ? 'OK' : 'Błąd'} color={stats.model_info?.loaded ? 'green' : 'red'} subValue={stats.model_info?.backbone || 'brak'} />
          </div>

          {stats.model_info && (
            <div className="bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-4">
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <InfoItem label="Backbone" value={stats.model_info.backbone} highlight />
                <InfoItem label="Typ" value={stats.model_info.type.toUpperCase()} />
                <InfoItem label="Próg detekcji" value={`${(stats.model_info.threshold * 100).toFixed(1)}%`} />
                <InfoItem label="Urządzenie" value={stats.model_info.device.toUpperCase()} />
                <InfoItem label="Status" value={stats.model_info.loaded ? 'Załadowany' : 'Błąd'} color={stats.model_info.loaded ? 'green' : 'red'} />
              </div>
            </div>
          )}

          <div className="flex gap-3 flex-wrap">
            <button onClick={fetchData} className="bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition">
              <RefreshCw size={14} /> Odśwież dane
            </button>
            <button onClick={handleCleanup} disabled={actionLoading === 'cleanup'} className="bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 hover:bg-amber-200 dark:hover:bg-amber-900/50 transition disabled:opacity-50">
              {actionLoading === 'cleanup' ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />} Wyczyść system
            </button>
            <button onClick={handleReloadMetrics} disabled={actionLoading === 'metrics'} className="bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400 px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 hover:bg-indigo-200 dark:hover:bg-indigo-900/50 transition disabled:opacity-50">
              {actionLoading === 'metrics' ? <Loader2 size={14} className="animate-spin" /> : <Activity size={14} />} Przeładuj metryki
            </button>
          </div>
        </div>
      )}

      {activeTab === 'users' && (
        <div className="bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl overflow-hidden">
          <div className="p-4 border-b border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950/50 flex items-center justify-between">
            <span className="font-semibold text-zinc-700 dark:text-zinc-300 flex items-center gap-2">
              <Users size={16} /> Zarządzanie Użytkownikami ({users.length})
            </span>
          </div>
          <div className="divide-y divide-zinc-200 dark:divide-zinc-800 max-h-[500px] overflow-y-auto">
            {users.map(u => (
              <div key={u.id} className="p-4 flex items-center justify-between hover:bg-zinc-50 dark:hover:bg-zinc-800/50 transition">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold ${u.is_superuser ? 'bg-indigo-500/20 text-indigo-500 dark:text-indigo-400' : 'bg-zinc-200 dark:bg-zinc-800 text-zinc-500 dark:text-zinc-400'}`}>
                    {u.username[0].toUpperCase()}
                  </div>
                  <div>
                    <div className="text-sm font-medium text-zinc-800 dark:text-zinc-200 flex items-center gap-2">
                      {u.username}
                      {u.is_superuser && <span className="text-[10px] bg-indigo-500/20 text-indigo-500 px-1.5 py-0.5 rounded font-bold">ADMIN</span>}
                    </div>
                    <div className="text-[11px] text-zinc-400 flex items-center gap-2 mt-0.5">
                      <Calendar size={10} />
                      Dołączył: {new Date(u.created_at).toLocaleDateString('pl-PL')}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => handleToggleAdmin(u.id, u.is_superuser)}
                    disabled={actionLoading === u.id}
                    className={`p-2 rounded-lg transition ${
                      u.is_superuser
                        ? 'text-indigo-500 hover:bg-indigo-50 dark:hover:bg-indigo-900/20'
                        : 'text-zinc-400 hover:text-indigo-500 hover:bg-indigo-50 dark:hover:bg-indigo-900/20'
                    }`}
                    title={u.is_superuser ? 'Odbierz uprawnienia admina' : 'Nadaj uprawnienia admina'}
                  >
                    {actionLoading === u.id ? <Loader2 size={16} className="animate-spin" /> : u.is_superuser ? <ShieldOff size={16} /> : <ShieldCheck size={16} />}
                  </button>
                  <button
                    onClick={() => handleDeleteUser(u.id, u.username)}
                    disabled={actionLoading === u.id}
                    className="text-zinc-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 p-2 rounded-lg transition"
                    title="Usuń użytkownika"
                  >
                    {actionLoading === u.id ? <Loader2 size={16} className="animate-spin" /> : <UserMinus size={16} />}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'model' && (
        <div className="space-y-6">
          <div className="bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-5">
            <h4 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-4 flex items-center gap-2">
              <Cpu size={16} className="text-indigo-500" /> Aktualnie załadowany model
            </h4>
            {stats?.model_info ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-white dark:bg-black/30 p-4 rounded-lg border border-zinc-200 dark:border-zinc-700">
                  <div className="text-xs text-zinc-500 uppercase mb-1">Backbone</div>
                  <div className="text-lg font-bold text-indigo-500">{stats.model_info.backbone}</div>
                </div>
                <div className="bg-white dark:bg-black/30 p-4 rounded-lg border border-zinc-200 dark:border-zinc-700">
                  <div className="text-xs text-zinc-500 uppercase mb-1">Typ</div>
                  <div className="text-lg font-bold text-zinc-700 dark:text-zinc-300">{stats.model_info.type}</div>
                </div>
                <div className="bg-white dark:bg-black/30 p-4 rounded-lg border border-zinc-200 dark:border-zinc-700">
                  <div className="text-xs text-zinc-500 uppercase mb-1">Próg AI</div>
                  <div className="text-lg font-bold text-amber-500">{(stats.model_info.threshold * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-white dark:bg-black/30 p-4 rounded-lg border border-zinc-200 dark:border-zinc-700">
                  <div className="text-xs text-zinc-500 uppercase mb-1">GPU/CPU</div>
                  <div className="text-lg font-bold text-emerald-500">{stats.model_info.device}</div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-zinc-500">
                <AlertTriangle size={32} className="mx-auto mb-2 text-red-500" />
                <p>Model nie jest załadowany</p>
              </div>
            )}
          </div>

          <div className="bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-5">
            <h4 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-4 flex items-center gap-2">
              <Upload size={16} className="text-emerald-500" /> Wgraj nowy model
            </h4>
            <input
              ref={modelInputRef}
              type="file"
              accept=".pt,.onnx"
              onChange={handleModelUpload}
              className="hidden"
            />
            <div className="flex items-center gap-4">
              <button
                onClick={() => modelInputRef.current?.click()}
                disabled={actionLoading === 'model'}
                className="bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 hover:bg-emerald-200 dark:hover:bg-emerald-900/50 transition disabled:opacity-50"
              >
                {actionLoading === 'model' ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
                Wybierz plik modelu (.pt lub .onnx)
              </button>
            </div>
            <p className="text-xs text-zinc-500 mt-3">
              Model zostanie zapisany w <code className="bg-zinc-200 dark:bg-zinc-800 px-1 rounded">runs/experiment/</code>.
              Po wgraniu zrestartuj serwer.
            </p>
          </div>
        </div>
      )}

      {activeTab === 'settings' && (
        <div className="space-y-6">
          <div className="bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-5">
            <h4 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-4 flex items-center gap-2">
              <Settings size={16} className="text-indigo-500" /> Akcje systemowe
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <ActionButton
                icon={<Trash2 size={16} />}
                label="Wyczyść pliki tymczasowe"
                description="Usuwa cache, pliki temp i stare zadania batch"
                color="amber"
                onClick={handleCleanup}
                loading={actionLoading === 'cleanup'}
              />
              <ActionButton
                icon={<Activity size={16} />}
                label="Przeładuj metryki"
                description="Odświeża custom metrics z folderu backend/custom_metrics"
                color="indigo"
                onClick={handleReloadMetrics}
                loading={actionLoading === 'metrics'}
              />
              <ActionButton
                icon={<Database size={16} />}
                label="Optymalizuj bazę danych"
                description="VACUUM i reindex bazy SQLite"
                color="emerald"
                onClick={handleOptimizeDb}
                loading={actionLoading === 'optimize'}
              />
            </div>
          </div>
          <div className="bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-5">
            <h4 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-4 flex items-center gap-2">
              <Zap size={16} className="text-indigo-500" /> Informacje API
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div><div className="text-xs text-zinc-500 uppercase">Wersja</div><div className="font-mono text-zinc-700 dark:text-zinc-300">{apiVersion}</div></div>
              <div><div className="text-xs text-zinc-500 uppercase">Backend</div><div className="font-mono text-zinc-700 dark:text-zinc-300">FastAPI</div></div>
              <div><div className="text-xs text-zinc-500 uppercase">Baza danych</div><div className="font-mono text-zinc-700 dark:text-zinc-300">SQLite</div></div>
              <div> <div className="text-xs text-zinc-500 uppercase">Custom Metrics</div> <div className="font-mono text-zinc-700 dark:text-zinc-300"> {stats?.metrics_count ?? 0} aktywnych </div> </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ icon, label, value, subValue, color = 'default' }: { icon: React.ReactNode; label: string; value: string | number; subValue?: string; color?: 'default' | 'red' | 'green'; }) {
  const colorClasses = { default: 'text-zinc-800 dark:text-white', red: 'text-red-500 dark:text-red-400', green: 'text-emerald-500 dark:text-emerald-400' };
  return (
    <div className="bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 p-4 rounded-xl">
      <div className="text-zinc-500 text-xs uppercase font-bold flex items-center gap-2 mb-2">{icon} {label}</div>
      <div className={`text-2xl font-mono font-bold ${colorClasses[color]}`}>{value}</div>
      {subValue && <div className="text-xs text-zinc-400 mt-1">{subValue}</div>}
    </div>
  );
}

function InfoItem({ label, value, highlight = false, color }: { label: string; value: string; highlight?: boolean; color?: 'green' | 'red'; }) {
  const valueColor = color === 'green' ? 'text-emerald-500' : color === 'red' ? 'text-red-500' : highlight ? 'text-indigo-500 dark:text-indigo-400' : 'text-zinc-700 dark:text-zinc-300';
  return (<div><div className="text-[10px] text-zinc-400 uppercase">{label}</div><div className={`text-sm font-bold ${valueColor}`}>{value}</div></div>);
}

function ActionButton({ icon, label, description, color, onClick, loading = false }: { icon: React.ReactNode; label: string; description: string; color: 'amber' | 'indigo' | 'emerald' | 'red'; onClick: () => void; loading?: boolean; }) {
  const colors = {
    amber: 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800 hover:bg-amber-100 dark:hover:bg-amber-900/30',
    indigo: 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800 hover:bg-indigo-100 dark:hover:bg-indigo-900/30',
    emerald: 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800 hover:bg-emerald-100 dark:hover:bg-emerald-900/30',
    red: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 hover:bg-red-100 dark:hover:bg-red-900/30'
  };
  const iconColors = { amber: 'text-amber-500', indigo: 'text-indigo-500', emerald: 'text-emerald-500', red: 'text-red-500' };
  return (
    <button onClick={onClick} disabled={loading} className={`p-4 rounded-xl border text-left transition disabled:opacity-50 ${colors[color]}`}>
      <div className={`${iconColors[color]} mb-2`}>{loading ? <Loader2 size={16} className="animate-spin" /> : icon}</div>
      <div className="text-sm font-medium text-zinc-700 dark:text-zinc-300">{label}</div>
      <div className="text-xs text-zinc-500 mt-1">{description}</div>
    </button>
  );
}