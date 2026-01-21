'use client';
import { useState, useEffect } from 'react';
import AuthForm from '@/components/AuthForm';
import Dashboard from '@/components/Dashboard';

export default function Home() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isGuest, setIsGuest] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) setIsLoggedIn(true);
    setIsLoading(false);
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

  if (isLoading) return null;

  if (isLoggedIn || isGuest) {
    return <Dashboard onLogoutAction={handleLogout} isGuest={isGuest} />;
  }

  return <AuthForm onLoginAction={handleLogin} onGuestAction={handleGuest} />;
}