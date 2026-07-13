'use client';

import { useState } from 'react';
import { api, endpoints } from '@/lib/api';
import { getErrorMessage } from '@/lib/errors';
import {
  ArrowRight, Check, Ghost, Loader2, Lock, LogIn,
  ScanSearch, User, UserPlus,
} from 'lucide-react';

interface AuthFormProps {
  onLoginAction: () => void;
  onGuestAction: () => void;
}

export default function AuthForm({ onLoginAction, onGuestAction }: AuthFormProps) {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isRegister) {
        await api.post(endpoints.register, { username, password });
        alert('Konto utworzone! Zaloguj się.');
        setIsRegister(false);
      } else {
        const formData = new FormData();
        formData.append('username', username);
        formData.append('password', password);
        const res = await api.post(endpoints.login, formData);
        localStorage.setItem('token', res.data.access_token);
        localStorage.setItem('username', username);
        onLoginAction();
      }
    } catch (err) {
      setError(getErrorMessage(err, 'Błąd połączenia z serwerem'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#f5f6f8] text-zinc-950 dark:bg-[#080a0f] dark:text-zinc-50">
      <div className="app-grid pointer-events-none absolute inset-0" />
      <div className="pointer-events-none absolute -left-32 top-[-12rem] h-[30rem] w-[30rem] rounded-full bg-indigo-500/10 blur-3xl dark:bg-indigo-500/15" />

      <div className="relative mx-auto grid min-h-screen max-w-[1440px] lg:grid-cols-[1.1fr_0.9fr]">
        <section className="hidden flex-col justify-between px-12 py-10 lg:flex xl:px-20 xl:py-14">
          <div className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-lg shadow-indigo-600/20">
              <ScanSearch size={21} />
            </span>
            <div>
              <p className="font-semibold tracking-tight">AI Photo Recognizer</p>
            </div>
          </div>

          <div className="max-w-xl py-14">
            <h1 className="text-5xl font-semibold leading-[1.08] tracking-[-0.04em] xl:text-6xl">
    Sprawdź, czy obraz został wygenerowany przez AI.
  </h1>
  <p className="mt-6 max-w-lg text-base leading-7 text-zinc-600 dark:text-zinc-400">
    Prześlij obraz, aby otrzymać wynik analizy, metryki jakości
    i mapę obszarów wpływających na decyzję modelu.
  </p>

            <div className="mt-10 grid grid-cols-3 gap-3">
              {[
                ['Wynik', 'czytelna ocena AI'],
                ['GradCAM', 'wizualne uzasadnienie'],
                ['Historia', 'porównanie analiz'],
              ].map(([title, text]) => (
                <div key={title} className="rounded-2xl border border-zinc-200/80 bg-white/60 p-4 backdrop-blur dark:border-white/8 dark:bg-white/[0.03]">
                  <Check size={15} className="mb-7 text-indigo-500" />
                  <p className="text-sm font-semibold">{title}</p>
                  <p className="mt-1 text-xs leading-5 text-zinc-500">{text}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="flex min-h-screen items-center justify-center border-zinc-200/70 px-5 py-10 lg:border-l lg:bg-white/35 dark:border-white/8 dark:lg:bg-black/10 sm:px-8">
          <div className="w-full max-w-[430px]">
            <div className="mb-9 flex items-center gap-3 lg:hidden">
              <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600 text-white">
                <ScanSearch size={21} />
              </span>
              <div>
                <p className="font-semibold">AIPR Workspace</p>
              </div>
            </div>

            <div className="rounded-3xl border border-zinc-200 bg-white p-6 shadow-[0_24px_80px_-36px_rgba(24,24,27,0.35)] dark:border-white/10 dark:bg-[#111318] dark:shadow-black/50 sm:p-8">
              <div className="mb-7">
                <p className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-indigo-600 dark:text-indigo-400">
                  {isRegister ? 'Nowe konto' : 'Logowanie'}
                </p>
                <h2 className="text-3xl font-semibold tracking-[-0.03em]">
                  {isRegister ? 'Rejestracja' : 'Witaj'}
                </h2>
                <p className="mt-2 text-sm leading-6 text-zinc-500 dark:text-zinc-400">
                  {isRegister ? 'Utwórz konto, aby zachować historię analiz.' : 'Zaloguj się, aby wrócić do swoich wyników.'}
                </p>
              </div>

              {error && (
                <div role="alert" className="mb-5 rounded-xl border border-red-200 bg-red-50 px-3.5 py-3 text-sm text-red-700 dark:border-red-500/20 dark:bg-red-500/10 dark:text-red-300">
                  {error}
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-4">
                <label className="block">
                  <span className="mb-1.5 block text-xs font-medium text-zinc-600 dark:text-zinc-300">Nazwa użytkownika</span>
                  <span className="relative block">
                    <User className="absolute left-3.5 top-3.5 text-zinc-400" size={17} />
                    <input
                      className="h-11 w-full rounded-xl border border-zinc-300 bg-white pl-10 pr-4 text-sm outline-none transition placeholder:text-zinc-400 hover:border-zinc-400 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/10 dark:border-zinc-700 dark:bg-black/25 dark:hover:border-zinc-600"
                      placeholder="Login"
                      value={username}
                      onChange={e => setUsername(e.target.value)}
                      disabled={loading}
                      autoComplete="username"
                      minLength={isRegister ? 3 : undefined}
                      maxLength={50}
                    />
                  </span>
                </label>

                <label className="block">
                  <span className="mb-1.5 block text-xs font-medium text-zinc-600 dark:text-zinc-300">Hasło</span>
                  <span className="relative block">
                    <Lock className="absolute left-3.5 top-3.5 text-zinc-400" size={17} />
                    <input
                      type="password"
                      className="h-11 w-full rounded-xl border border-zinc-300 bg-white pl-10 pr-4 text-sm outline-none transition placeholder:text-zinc-400 hover:border-zinc-400 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/10 dark:border-zinc-700 dark:bg-black/25 dark:hover:border-zinc-600"
                      placeholder="Hasło"
                      value={password}
                      onChange={e => setPassword(e.target.value)}
                      disabled={loading}
                      autoComplete={isRegister ? 'new-password' : 'current-password'}
                      minLength={isRegister ? 6 : undefined}
                      maxLength={100}
                    />
                  </span>
                </label>

                <button
                  type="submit"
                  disabled={loading || !username || !password}
                  className="flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 text-sm font-semibold text-white shadow-lg shadow-indigo-600/15 transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {loading ? <Loader2 size={17} className="animate-spin" /> : isRegister ? <UserPlus size={17} /> : <LogIn size={17} />}
                  {isRegister ? 'Zarejestruj się' : 'Zaloguj się'}
                </button>
              </form>

              <div className="my-6 flex items-center gap-3 text-[11px] uppercase tracking-[0.12em] text-zinc-400">
                <span className="h-px flex-1 bg-zinc-200 dark:bg-zinc-800" />
                lub szybki podgląd
                <span className="h-px flex-1 bg-zinc-200 dark:bg-zinc-800" />
              </div>

              <button
                onClick={onGuestAction}
                type="button"
                className="flex h-11 w-full items-center justify-center gap-2 rounded-xl border border-zinc-300 bg-zinc-50 text-sm font-medium text-zinc-700 transition hover:border-zinc-400 hover:bg-zinc-100 dark:border-zinc-700 dark:bg-white/[0.03] dark:text-zinc-200 dark:hover:bg-white/[0.06]"
              >
                <Ghost size={17} /> Kontynuuj jako Gość
              </button>

              <button
                onClick={() => { setIsRegister(!isRegister); setError(''); }}
                className="mx-auto mt-6 flex items-center gap-1.5 text-sm text-zinc-500 transition hover:text-indigo-600 dark:hover:text-indigo-400"
              >
                {isRegister ? 'Masz już konto? Zaloguj się' : 'Nie masz konta? Załóż je'}
                <ArrowRight size={14} />
              </button>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
