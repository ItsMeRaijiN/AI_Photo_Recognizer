import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Dashboard from '../components/Dashboard';
import { AnalysisResult, BatchUploadResponse } from '@/lib/types';

jest.mock('../components/UploadArea', () => {
  return function MockUploadArea({
    onFileSelect,
    onBatchUpload,
    isLoading,
  }: {
    onFileSelect: (files: File[]) => void;
    onBatchUpload?: (files: File[]) => Promise<void>;
    isLoading: boolean;
  }) {
    return (
      <div data-testid="upload-area">
        <button
          data-testid="upload-single"
          onClick={() => onFileSelect([new File(['x'], 'test.jpg', { type: 'image/jpeg' })])}
          disabled={isLoading}
        >
          Upload
        </button>
        <button
          data-testid="upload-batch"
          onClick={() => onBatchUpload?.([
            new File(['1'], 'a.jpg', { type: 'image/jpeg' }),
            new File(['2'], 'b.jpg', { type: 'image/jpeg' }),
          ])}
          disabled={isLoading}
        >
          Batch
        </button>
        <span data-testid="loading">{isLoading ? 'loading' : 'idle'}</span>
      </div>
    );
  };
});

jest.mock('../components/AnalysisResult', () => {
  return function MockAnalysisResult({ data }: { data: AnalysisResult }) {
    return (
      <div data-testid="analysis-result">
        <span data-testid="result-filename">{data.filename}</span>
        <span data-testid="result-verdict">{data.is_ai ? 'AI' : 'REAL'}</span>
      </div>
    );
  };
});

jest.mock('../components/AdminPanel', () => {
  return function MockAdminPanel() {
    return <div data-testid="admin-panel">Admin Panel</div>;
  };
});

jest.mock('../components/AuthImage', () => {
  return function MockAuthImage({ alt }: { alt: string }) {
    return <img data-testid="auth-image" alt={alt} />;
  };
});

const mockGet = jest.fn();
const mockPost = jest.fn();
jest.mock('../lib/api', () => ({
  api: {
    get: (...args: unknown[]) => mockGet(...args),
    post: (...args: unknown[]) => mockPost(...args),
  },
  endpoints: {
    adminStats: '/admin/stats',
    history: '/analysis/history',
    predict: '/analysis/predict',
    predictBatch: '/analysis/predict/batch',
  },
}));

const localStore: Record<string, string> = {};
const sessionStore: Record<string, string> = {};

const mockLocalStorage = {
  getItem: jest.fn((k: string) => localStore[k] || null),
  setItem: jest.fn((k: string, v: string) => { localStore[k] = v; }),
  removeItem: jest.fn((k: string) => { delete localStore[k]; }),
  clear: jest.fn(() => Object.keys(localStore).forEach(k => delete localStore[k])),
};

const mockSessionStorage = {
  getItem: jest.fn((k: string) => sessionStore[k] || null),
  setItem: jest.fn((k: string, v: string) => { sessionStore[k] = v; }),
  removeItem: jest.fn((k: string) => { delete sessionStore[k]; }),
  clear: jest.fn(() => Object.keys(sessionStore).forEach(k => delete sessionStore[k])),
};

Object.defineProperty(window, 'localStorage', { value: mockLocalStorage });
Object.defineProperty(window, 'sessionStorage', { value: mockSessionStorage });

global.URL.createObjectURL = jest.fn(() => 'blob:test');
global.URL.revokeObjectURL = jest.fn();


const mockResult: AnalysisResult = {
  id: 1,
  filename: 'photo.jpg',
  file_path: '/uploads/photo.jpg',
  is_ai: true,
  score: 0.85,
  confidence: 0.9,
  threshold_used: 0.5,
  inference_time_ms: 120,
  model_type: 'torch',
  backbone_name: 'effnetv2',
  has_heatmap: true,
  created_at: new Date().toISOString(),
};

const mockHistory: AnalysisResult[] = [
  { ...mockResult, id: 1, filename: 'img1.jpg' },
  { ...mockResult, id: 2, filename: 'img2.jpg', is_ai: false, score: 0.15 },
];

describe('Dashboard', () => {
  const mockLogout = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalStorage.clear();
    mockSessionStorage.clear();
    localStore['username'] = 'testuser';

    // Domyślne odpowiedzi API
    mockGet.mockImplementation((url: string) => {
      if (url === '/admin/stats') return Promise.reject(new Error('Not admin'));
      if (url === '/analysis/history') return Promise.resolve({ data: mockHistory });
      return Promise.resolve({ data: {} });
    });

    mockPost.mockImplementation((url: string) => {
      if (url === '/analysis/predict') return Promise.resolve({ data: mockResult });
      if (url === '/analysis/predict/batch') {
        return Promise.resolve({
          data: {
            total: 2, processed: 2, failed: 0,
            results: [
              { ...mockResult, id: 1, filename: 'a.jpg' },
              { ...mockResult, id: 2, filename: 'b.jpg' },
            ],
            errors: [],
            total_inference_time_ms: 240,
          } as BatchUploadResponse,
        });
      }
      return Promise.resolve({ data: {} });
    });
  });


  describe('rendering', () => {
    it('should render header with app name and username', async () => {
      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => {
        expect(screen.getByText('AI Photo Recognizer')).toBeInTheDocument();
        expect(screen.getByText('testuser')).toBeInTheDocument();
      });
    });

    it('should render "Gość" for guest user', async () => {
      render(<Dashboard onLogoutAction={mockLogout} isGuest={true} />);

      await waitFor(() => {
        expect(screen.getByText('Gość')).toBeInTheDocument();
      });
    });

    it('should render upload area and history sidebar', async () => {
      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => {
        expect(screen.getByTestId('upload-area')).toBeInTheDocument();
        expect(screen.getByText('Historia')).toBeInTheDocument();
      });
    });

    it('should render logout button', async () => {
      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => {
        expect(screen.getByTitle(/Wyloguj/i)).toBeInTheDocument();
      });
    });
  });

  describe('history', () => {
    it('should fetch and display history for logged in user', async () => {
      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => {
        expect(mockGet).toHaveBeenCalledWith('/analysis/history');
        expect(screen.getByText('img1.jpg')).toBeInTheDocument();
        expect(screen.getByText('img2.jpg')).toBeInTheDocument();
      });
    });

    it('should NOT fetch history for guest user', async () => {
      render(<Dashboard onLogoutAction={mockLogout} isGuest={true} />);

      await waitFor(() => {
        expect(mockGet).not.toHaveBeenCalledWith('/analysis/history');
      });
    });

    it('should show empty message when no history', async () => {
      mockGet.mockImplementation((url: string) => {
        if (url === '/analysis/history') return Promise.resolve({ data: [] });
        return Promise.reject();
      });

      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => {
        expect(screen.getByText('Brak wyników.')).toBeInTheDocument();
      });
    });

    it('should display AI/REAL badges in history', async () => {
      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => {
        expect(screen.getAllByText('AI').length).toBeGreaterThan(0);
        expect(screen.getAllByText('REAL').length).toBeGreaterThan(0);
      });
    });

    it('should select history item and show analysis result', async () => {
      const user = userEvent.setup();
      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => expect(screen.getByText('img1.jpg')).toBeInTheDocument());

      await user.click(screen.getByText('img1.jpg'));

      await waitFor(() => {
        expect(screen.getByTestId('analysis-result')).toBeInTheDocument();
        expect(screen.getByTestId('result-filename')).toHaveTextContent('img1.jpg');
      });
    });
  });

  describe('single file upload', () => {
    it('should upload file and display result', async () => {
      const user = userEvent.setup();
      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => expect(screen.getByTestId('upload-single')).toBeInTheDocument());

      await user.click(screen.getByTestId('upload-single'));

      await waitFor(() => {
        expect(mockPost).toHaveBeenCalledWith('/analysis/predict', expect.any(FormData));
        expect(screen.getByTestId('analysis-result')).toBeInTheDocument();
      });
    });

    it('should show alert on upload error', async () => {
      const user = userEvent.setup();
      const alertMock = jest.spyOn(window, 'alert').mockImplementation(() => {});
      jest.spyOn(console, 'error').mockImplementation(() => {});
      mockPost.mockRejectedValueOnce(new Error('Upload failed'));

      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => expect(screen.getByTestId('upload-single')).toBeInTheDocument());

      await user.click(screen.getByTestId('upload-single'));

      await waitFor(() => {
        expect(alertMock).toHaveBeenCalled();
      });

      alertMock.mockRestore();
    });
  });

  describe('batch upload', () => {
    it('should upload batch and display results summary', async () => {
      const user = userEvent.setup();
      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => expect(screen.getByTestId('upload-batch')).toBeInTheDocument());

      await user.click(screen.getByTestId('upload-batch'));

      await waitFor(() => {
        expect(mockPost).toHaveBeenCalledWith(
          '/analysis/predict/batch',
          expect.any(FormData),
          expect.anything()
        );
        expect(screen.getByText(/Wyniki Batch/)).toBeInTheDocument();
      });
    });
  });

  describe('guest mode', () => {
    it('should save guest history to sessionStorage', async () => {
      const user = userEvent.setup();
      mockPost.mockResolvedValueOnce({ data: { ...mockResult, id: -1 } });

      render(<Dashboard onLogoutAction={mockLogout} isGuest={true} />);

      await waitFor(() => expect(screen.getByTestId('upload-single')).toBeInTheDocument());

      await user.click(screen.getByTestId('upload-single'));

      await waitFor(() => {
        expect(mockSessionStorage.setItem).toHaveBeenCalledWith(
          'guestHistory',
          expect.any(String)
        );
      });
    });

    it('should clear guest history on button click', async () => {
      const user = userEvent.setup();
      jest.spyOn(window, 'confirm').mockReturnValue(true);
      sessionStore['guestHistory'] = JSON.stringify([{ ...mockResult, id: -1 }]);

      render(<Dashboard onLogoutAction={mockLogout} isGuest={true} />);

      await waitFor(() => expect(screen.getByTitle(/Wyczyść historię/i)).toBeInTheDocument());

      await user.click(screen.getByTitle(/Wyczyść historię/i));

      await waitFor(() => {
        expect(mockSessionStorage.removeItem).toHaveBeenCalledWith('guestHistory');
      });
    });
  });

  describe('admin panel', () => {
    beforeEach(() => {
      mockGet.mockImplementation((url: string) => {
        if (url === '/admin/stats') return Promise.resolve({ data: {} });
        if (url === '/analysis/history') return Promise.resolve({ data: mockHistory });
        return Promise.resolve({ data: {} });
      });
    });

    it('should show admin button for admin user', async () => {
      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => {
        expect(screen.getByTitle('Panel administratora')).toBeInTheDocument();
      });
    });

    it('should toggle admin panel visibility', async () => {
      const user = userEvent.setup();
      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => expect(screen.getByTitle('Panel administratora')).toBeInTheDocument());

      await user.click(screen.getByTitle('Panel administratora'));

      await waitFor(() => {
        expect(screen.getByTestId('admin-panel')).toBeInTheDocument();
        expect(screen.queryByTestId('upload-area')).not.toBeInTheDocument();
      });
    });
  });

  describe('logout', () => {
    it('should call onLogoutAction when logout button is clicked', async () => {
      const user = userEvent.setup();
      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => expect(screen.getByTitle(/Wyloguj/i)).toBeInTheDocument());

      await user.click(screen.getByTitle(/Wyloguj/i));

      expect(mockLogout).toHaveBeenCalled();
    });
  });

  describe('loading state', () => {
    it('should show loading state during upload', async () => {
      let resolve: (v: unknown) => void;
      mockPost.mockReturnValueOnce(new Promise(r => { resolve = r; }));

      const user = userEvent.setup();
      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => expect(screen.getByTestId('upload-single')).toBeInTheDocument());

      fireEvent.click(screen.getByTestId('upload-single'));

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('loading');
      });

      resolve!({ data: mockResult });

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('idle');
      });
    });
  });

  describe('export', () => {
    it('should have CSV export button', async () => {
      render(<Dashboard onLogoutAction={mockLogout} isGuest={false} />);

      await waitFor(() => {
        expect(screen.getByTitle('Pobierz CSV')).toBeInTheDocument();
      });
    });
  });
});