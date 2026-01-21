'use client';
import { useState, useEffect } from 'react';
import { api, endpoints } from '@/lib/api';
import { ImageOff, Loader2 } from 'lucide-react';

interface AuthImageProps {
  filePath: string;
  analysisId?: number;
  fallbackSrc?: string;
  alt: string;
  className?: string;
}

export default function AuthImage({
  filePath,
  analysisId,
  fallbackSrc,
  alt,
  className
}: AuthImageProps) {
  const [src, setSrc] = useState<string | null>(fallbackSrc || null);
  const [loading, setLoading] = useState(!fallbackSrc);
  const [error, setError] = useState(false);

  useEffect(() => {
    setError(false);

    if (fallbackSrc) {
      setSrc(fallbackSrc);
      setLoading(false);
      return;
    }

    if (filePath === 'memory' || filePath === 'upload://memory' || !filePath) {
      setError(true);
      setLoading(false);
      return;
    }

    let isMounted = true;
    let objectUrl: string | null = null;

    const fetchImage = async () => {
      try {
        setLoading(true);

        if (analysisId && analysisId > 0) {
          const response = await api.get(endpoints.analysisImage(analysisId), {
            responseType: 'blob'
          });

          objectUrl = URL.createObjectURL(response.data);

          if (isMounted) {
            setSrc(objectUrl);
            setError(false);
          }
        } else {
          if (isMounted) {
            setError(true);
          }
        }
      } catch (e) {
        console.warn('Image load failed:', e);
        if (isMounted) {
          setError(true);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchImage();

    return () => {
      isMounted = false;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [filePath, analysisId, fallbackSrc]);

  if (loading) {
    return (
      <div className={`flex items-center justify-center bg-zinc-100 dark:bg-zinc-800 ${className}`}>
        <Loader2 className="animate-spin text-zinc-400" size={20} />
      </div>
    );
  }

  if (error || !src) {
    return (
      <div className={`flex flex-col items-center justify-center bg-zinc-100 dark:bg-zinc-800 text-zinc-400 gap-1 ${className}`}>
        <ImageOff size={20} />
        <span className="text-[10px]">Podgląd niedostępny</span>
      </div>
    );
  }

  return <img src={src} alt={alt} className={className} />;
}