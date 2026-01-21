'use client';
import { useState } from 'react';
import { api, endpoints } from '@/lib/api';
import { Lock, User, ArrowRight, UserPlus, LogIn, Ghost } from 'lucide-react';

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
        alert("Konto utworzone! Zaloguj się.");
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
    } catch (err: any) {
      setError(err.response?.data?.detail || "Błąd połączenia z serwerem");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-[80vh] items-center justify-center p-4">
      <div className="w-full max-w-md bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 p-8 rounded-2xl shadow-2xl">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-500 to-indigo-500 bg-clip-text text-transparent">
            {isRegister ? 'Rejestracja' : 'Witaj'}
          </h2>
          <p className="text-zinc-500 dark:text-zinc-400 text-sm mt-2">AI Photo Recognizer</p>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 text-red-500 dark:text-red-400 text-sm rounded-lg text-center">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative group">
            <User className="absolute left-3 top-3 text-zinc-400 group-focus-within:text-indigo-400 transition" size={18} />
            <input
              className="w-full bg-white dark:bg-black/50 border border-zinc-300 dark:border-zinc-700 rounded-lg py-2.5 pl-10 pr-4 text-zinc-900 dark:text-white focus:border-indigo-500 outline-none transition"
              placeholder="Login"
              value={username}
              onChange={e => setUsername(e.target.value)}
              disabled={loading}
            />
          </div>
          <div className="relative group">
            <Lock className="absolute left-3 top-3 text-zinc-400 group-focus-within:text-indigo-400 transition" size={18} />
            <input
              type="password"
              className="w-full bg-white dark:bg-black/50 border border-zinc-300 dark:border-zinc-700 rounded-lg py-2.5 pl-10 pr-4 text-zinc-900 dark:text-white focus:border-indigo-500 outline-none transition"
              placeholder="Hasło"
              value={password}
              onChange={e => setPassword(e.target.value)}
              disabled={loading}
            />
          </div>

          <button
            type="submit"
            disabled={loading || !username || !password}
            className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-2.5 rounded-lg transition-all flex items-center justify-center gap-2"
          >
            {isRegister ? <UserPlus size={18} /> : <LogIn size={18} />}
            {isRegister ? 'Zarejestruj się' : 'Zaloguj się'}
          </button>
        </form>

        <div className="relative my-6">
          <div className="absolute inset-0 flex items-center">
            <span className="w-full border-t border-zinc-300 dark:border-zinc-700" />
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-zinc-50 dark:bg-zinc-900 px-2 text-zinc-500">LUB</span>
          </div>
        </div>

        <button
          onClick={onGuestAction}
          type="button"
          className="w-full bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 font-medium py-2.5 rounded-lg transition-all flex items-center justify-center gap-2 border border-zinc-200 dark:border-zinc-700"
        >
          <Ghost size={18} /> Kontynuuj jako Gość
        </button>

        <div className="mt-6 text-center">
          <button
            onClick={() => setIsRegister(!isRegister)}
            className="text-sm text-zinc-500 hover:text-zinc-900 dark:hover:text-white transition-colors flex items-center justify-center gap-1 mx-auto"
          >
            {isRegister ? 'Masz już konto? Zaloguj się' : 'Nie masz konta? Załóż je'}
            <ArrowRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}