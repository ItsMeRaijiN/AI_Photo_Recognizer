'use client';
import { useState, useEffect, useSyncExternalStore } from 'react';
import AuthForm from '@/components/AuthForm';
import Dashboard from '@/components/Dashboard';

const emptySubscribe = () => () => {};

function useHydrated(): boolean {
  return useSyncExternalStore(emptySubscribe, () => true, () => false);
}

export default function Home() {
  const hydrated = useHydrated();
  const [isLoggedIn, setIsLoggedIn] = useState(
    () => typeof window !== 'undefined' && Boolean(localStorage.getItem('token'))
  );
  const [isGuest, setIsGuest] = useState(false);

  useEffect(() => {
    const handleUnauthorized = () => setIsLoggedIn(false);
    window.addEventListener('auth:unauthorized', handleUnauthorized);
    return () => window.removeEventListener('auth:unauthorized', handleUnauthorized);
  }, []);

  const handleLogin = () => setIsLoggedIn(true);
  const handleGuest = () => setIsGuest(true);

  const handleLogout = () => {
    if (!isGuest) {
      localStorage.removeItem('token');
      localStorage.removeItem('username');
    }
    setIsLoggedIn(false);
    setIsGuest(false);
  };

  if (!hydrated) return null;

  if (isLoggedIn || isGuest) {
    return <Dashboard onLogoutAction={handleLogout} isGuest={isGuest} />;
  }

  return <AuthForm onLoginAction={handleLogin} onGuestAction={handleGuest} />;
}
