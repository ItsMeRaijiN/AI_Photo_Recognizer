interface ApiErrorShape {
  response?: {
    status?: number;
    data?: { detail?: unknown };
  };
  code?: string;
  message?: string;
}

function asApiError(error: unknown): ApiErrorShape {
  if (typeof error === 'object' && error !== null) {
    return error as ApiErrorShape;
  }
  return {};
}

export function getErrorStatus(error: unknown): number | undefined {
  return asApiError(error).response?.status;
}

export function getErrorMessage(error: unknown, fallback: string): string {
  const err = asApiError(error);
  const detail = err.response?.data?.detail;

  if (typeof detail === 'string' && detail) {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map(item =>
        typeof item === 'object' && item !== null && 'msg' in item
          ? String((item as { msg: unknown }).msg)
          : String(item)
      )
      .filter(Boolean);
    if (messages.length) return messages.join('; ');
  }

  if (err.code === 'ECONNABORTED') {
    return 'Przekroczono limit czasu żądania. Spróbuj ponownie lub wyślij mniej plików.';
  }

  if (err.response && err.message) return err.message;

  return fallback;
}
