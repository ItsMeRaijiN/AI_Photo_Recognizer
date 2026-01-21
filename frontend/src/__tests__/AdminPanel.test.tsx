import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AdminPanel from '../components/AdminPanel';
import { User, SystemStats } from '@/lib/types';

const mockGet = jest.fn();
const mockPost = jest.fn();
const mockDelete = jest.fn();
const mockPatch = jest.fn();

jest.mock('../lib/api', () => ({
  api: {
    get: (...args: unknown[]) => mockGet(...args),
    post: (...args: unknown[]) => mockPost(...args),
    delete: (...args: unknown[]) => mockDelete(...args),
    patch: (...args: unknown[]) => mockPatch(...args),
  },
  endpoints: {
    root: '/',
    adminUsers: '/admin/users',
    adminStats: '/admin/stats',
    adminUser: (id: number) => `/admin/users/${id}`,
    adminToggleAdmin: (id: number) => `/admin/users/${id}/toggle-admin`,
    adminCleanup: '/admin/cleanup',
    adminReloadMetrics: '/admin/metrics/reload',
    adminOptimizeDb: '/admin/optimize-db',
    adminUploadModel: '/admin/upload-model',
  },
}));

describe('AdminPanel', () => {
  const mockUsers: User[] = [
    { id: 1, username: 'admin', is_active: true, is_superuser: true, created_at: '2024-01-01T00:00:00Z' },
    { id: 2, username: 'user1', is_active: true, is_superuser: false, created_at: '2024-01-02T00:00:00Z' },
  ];

  const mockStats: SystemStats = {
    total_users: 10,
    active_users: 8,
    total_analyses: 500,
    ai_detections: 200,
    human_detections: 300,
    ai_ratio: 0.4,
    analyses_today: 50,
    storage_used_mb: 1024,
    model_info: {
      type: 'torch',
      backbone: 'effnetv2',
      version: '1.0.0',
      threshold: 0.5,
      device: 'cuda',
      loaded: true,
    },
    metrics_count: 6,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockGet.mockImplementation((url: string) => {
      if (url === '/admin/users') return Promise.resolve({ data: mockUsers });
      if (url === '/admin/stats') return Promise.resolve({ data: mockStats });
      if (url === '/') return Promise.resolve({ data: { version: '2.0.0' } });
      return Promise.resolve({ data: {} });
    });
  });

  describe('loading', () => {
    it('should show spinner while loading', () => {
      mockGet.mockReturnValue(new Promise(() => {}));
      render(<AdminPanel />);
      expect(document.querySelector('.animate-spin')).toBeInTheDocument();
    });

    it('should hide spinner after data loads', async () => {
      render(<AdminPanel />);
      await waitFor(() => {
        expect(document.querySelector('.animate-spin')).not.toBeInTheDocument();
      });
    });
  });

  describe('tab navigation', () => {
    it('should render all tabs and switch between them', async () => {
      const user = userEvent.setup();
      render(<AdminPanel />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Dashboard/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Użytkownicy/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Model AI/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Ustawienia/i })).toBeInTheDocument();
      });

      expect(screen.getByRole('button', { name: /Dashboard/i })).toHaveClass('bg-indigo-500');

      await user.click(screen.getByRole('button', { name: /Użytkownicy/i }));
      expect(screen.getByRole('button', { name: /Użytkownicy/i })).toHaveClass('bg-indigo-500');
      expect(screen.getByText('admin')).toBeInTheDocument();
    });
  });

  describe('dashboard tab', () => {
    it('should display stats from API', async () => {
      render(<AdminPanel />);

      await waitFor(() => {
        expect(screen.getByText('10')).toBeInTheDocument(); // total_users
        expect(screen.getByText('500')).toBeInTheDocument(); // total_analyses
        expect(screen.getByText(/200 wykrytych AI/)).toBeInTheDocument();
        expect(screen.getByText('OK')).toBeInTheDocument(); // model loaded
      });
    });
  });

  describe('user management', () => {
    it('should display user list with admin badges', async () => {
      const user = userEvent.setup();
      render(<AdminPanel />);

      await waitFor(() => screen.getByRole('button', { name: /Użytkownicy/i }));
      await user.click(screen.getByRole('button', { name: /Użytkownicy/i }));

      await waitFor(() => {
        expect(screen.getByText('admin')).toBeInTheDocument();
        expect(screen.getByText('user1')).toBeInTheDocument();
        expect(screen.getByText('ADMIN')).toBeInTheDocument();
      });
    });

    it('should delete user after confirmation', async () => {
      const user = userEvent.setup();
      jest.spyOn(window, 'confirm').mockReturnValue(true);
      mockDelete.mockResolvedValueOnce({ data: {} });

      render(<AdminPanel />);

      await waitFor(() => screen.getByRole('button', { name: /Użytkownicy/i }));
      await user.click(screen.getByRole('button', { name: /Użytkownicy/i }));
      await waitFor(() => screen.getByText('user1'));

      const deleteButtons = screen.getAllByTitle('Usuń użytkownika');
      await user.click(deleteButtons[1]); // user1

      await waitFor(() => {
        expect(mockDelete).toHaveBeenCalledWith('/admin/users/2');
      });
    });

    it('should toggle admin status after confirmation', async () => {
      const user = userEvent.setup();
      jest.spyOn(window, 'confirm').mockReturnValue(true);
      mockPatch.mockResolvedValueOnce({ data: {} });

      render(<AdminPanel />);

      await waitFor(() => screen.getByRole('button', { name: /Użytkownicy/i }));
      await user.click(screen.getByRole('button', { name: /Użytkownicy/i }));
      await waitFor(() => screen.getByText('user1'));

      const toggleButtons = screen.getAllByTitle(/uprawnienia admina/i);
      await user.click(toggleButtons[1]); // user1 - nadaj admina

      await waitFor(() => {
        expect(mockPatch).toHaveBeenCalledWith('/admin/users/2/toggle-admin');
      });
    });

    it('should show error alert on delete failure', async () => {
      const user = userEvent.setup();
      jest.spyOn(window, 'confirm').mockReturnValue(true);
      const alertMock = jest.spyOn(window, 'alert').mockImplementation(() => {});
      mockDelete.mockRejectedValueOnce({ response: { data: { detail: 'Cannot delete admin' } } });

      render(<AdminPanel />);

      await waitFor(() => screen.getByRole('button', { name: /Użytkownicy/i }));
      await user.click(screen.getByRole('button', { name: /Użytkownicy/i }));
      await waitFor(() => screen.getByText('admin'));

      const deleteButtons = screen.getAllByTitle('Usuń użytkownika');
      await user.click(deleteButtons[0]);

      await waitFor(() => {
        expect(alertMock).toHaveBeenCalledWith('Cannot delete admin');
      });

      alertMock.mockRestore();
    });
  });

  describe('model tab', () => {
    it('should display model info', async () => {
      const user = userEvent.setup();
      render(<AdminPanel />);

      await waitFor(() => screen.getByRole('button', { name: /Model AI/i }));
      await user.click(screen.getByRole('button', { name: /Model AI/i }));

      await waitFor(() => {
        expect(screen.getByText('effnetv2')).toBeInTheDocument();
        expect(screen.getByText('cuda')).toBeInTheDocument();
        expect(screen.getByText(/Wybierz plik modelu/i)).toBeInTheDocument();
      });
    });

    it('should reject invalid model file extension', async () => {
      const user = userEvent.setup();
      const alertMock = jest.spyOn(window, 'alert').mockImplementation(() => {});

      render(<AdminPanel />);

      await waitFor(() => screen.getByRole('button', { name: /Model AI/i }));
      await user.click(screen.getByRole('button', { name: /Model AI/i }));
      await waitFor(() => screen.getByText(/Wybierz plik modelu/i));

      const fileInput = document.querySelector('input[type="file"][accept=".pt,.onnx"]')!;
      const invalidFile = new File(['test'], 'model.txt', { type: 'text/plain' });
      fireEvent.change(fileInput, { target: { files: [invalidFile] } });

      await waitFor(() => {
        expect(alertMock).toHaveBeenCalledWith('Model musi być plikiem .pt lub .onnx');
      });

      alertMock.mockRestore();
    });

    it('should upload valid model file', async () => {
      const user = userEvent.setup();
      jest.spyOn(window, 'confirm').mockReturnValue(true);
      jest.spyOn(window, 'alert').mockImplementation(() => {});
      mockPost.mockResolvedValueOnce({ data: { path: '/models/new.pt' } });

      render(<AdminPanel />);

      await waitFor(() => screen.getByRole('button', { name: /Model AI/i }));
      await user.click(screen.getByRole('button', { name: /Model AI/i }));
      await waitFor(() => screen.getByText(/Wybierz plik modelu/i));

      const fileInput = document.querySelector('input[type="file"][accept=".pt,.onnx"]')!;
      const validFile = new File(['model-data'], 'model.pt', { type: 'application/octet-stream' });
      fireEvent.change(fileInput, { target: { files: [validFile] } });

      await waitFor(() => {
        expect(mockPost).toHaveBeenCalledWith('/admin/upload-model', expect.any(FormData));
      });
    });
  });

  describe('settings tab - system actions', () => {
    it('should call cleanup API', async () => {
      const user = userEvent.setup();
      jest.spyOn(window, 'confirm').mockReturnValue(true);
      jest.spyOn(window, 'alert').mockImplementation(() => {});
      mockPost.mockResolvedValueOnce({
        data: { deleted_temp_files: 10, deleted_old_jobs: 5, deleted_orphan_analyses: 2 },
      });

      render(<AdminPanel />);

      await waitFor(() => screen.getByRole('button', { name: /Ustawienia/i }));
      await user.click(screen.getByRole('button', { name: /Ustawienia/i }));
      await waitFor(() => screen.getByText('Wyczyść pliki tymczasowe'));

      await user.click(screen.getByText('Wyczyść pliki tymczasowe'));

      await waitFor(() => {
        expect(mockPost).toHaveBeenCalledWith('/admin/cleanup');
      });
    });

    it('should call reload metrics API', async () => {
      const user = userEvent.setup();
      jest.spyOn(window, 'alert').mockImplementation(() => {});
      mockPost.mockResolvedValueOnce({ data: {} });

      render(<AdminPanel />);

      await waitFor(() => screen.getByRole('button', { name: /Ustawienia/i }));
      await user.click(screen.getByRole('button', { name: /Ustawienia/i }));
      await waitFor(() => screen.getByText('Przeładuj metryki'));

      await user.click(screen.getByText('Przeładuj metryki'));

      await waitFor(() => {
        expect(mockPost).toHaveBeenCalledWith('/admin/metrics/reload');
      });
    });

    it('should call optimize DB API', async () => {
      const user = userEvent.setup();
      jest.spyOn(window, 'alert').mockImplementation(() => {});
      mockPost.mockResolvedValueOnce({ data: { size_before: '100 MB', size_after: '80 MB' } });

      render(<AdminPanel />);

      await waitFor(() => screen.getByRole('button', { name: /Ustawienia/i }));
      await user.click(screen.getByRole('button', { name: /Ustawienia/i }));
      await waitFor(() => screen.getByText('Optymalizuj bazę danych'));

      await user.click(screen.getByText('Optymalizuj bazę danych'));

      await waitFor(() => {
        expect(mockPost).toHaveBeenCalledWith('/admin/optimize-db');
      });
    });

    it('should display API version and metrics count', async () => {
      const user = userEvent.setup();
      render(<AdminPanel />);

      await waitFor(() => screen.getByRole('button', { name: /Ustawienia/i }));
      await user.click(screen.getByRole('button', { name: /Ustawienia/i }));

      await waitFor(() => {
        expect(screen.getByText('2.0.0')).toBeInTheDocument();
        expect(screen.getByText(/6 aktywnych/)).toBeInTheDocument();
      });
    });
  });

  describe('error handling', () => {
    it('should handle API error gracefully', async () => {
      mockGet.mockRejectedValue(new Error('Network error'));
      jest.spyOn(console, 'error').mockImplementation(() => {});

      render(<AdminPanel />);

      await waitFor(() => {
        expect(document.querySelector('.animate-spin')).not.toBeInTheDocument();
      }, { timeout: 3000 });
    });
  });
});