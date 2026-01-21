import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AnalysisResultCard from '../components/AnalysisResult';
import { AnalysisResult } from '@/lib/types';

jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <div {...props}>{children}</div>
    ),
  },
}));

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
    heatmap: (id: number, download = false) => `/analysis/${id}/heatmap${download ? '?download=true' : ''}`,
    saveHeatmap: (id: number) => `/analysis/${id}/heatmap/save`,
  },
}));

global.URL.createObjectURL = jest.fn(() => 'blob:http://localhost/test-heatmap');
global.URL.revokeObjectURL = jest.fn();

describe('AnalysisResultCard', () => {
  const createMockResult = (overrides: Partial<AnalysisResult> = {}): AnalysisResult => ({
    id: 1,
    filename: 'test_image.jpg',
    file_path: '/uploads/test_image.jpg',
    is_ai: false,
    score: 0.3,
    confidence: 0.8,
    threshold_used: 0.5,
    inference_time_ms: 150,
    model_type: 'torch',
    backbone_name: 'effnetv2',
    has_heatmap: true,
    created_at: new Date().toISOString(),
    ...overrides,
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('verdict display', () => {
    it('should show AI verdict for high score', () => {
      const data = createMockResult({ is_ai: true, score: 0.85 });
      render(<AnalysisResultCard data={data} />);

      expect(screen.getByText('WYKRYTO AI')).toBeInTheDocument();
      expect(screen.getByText('85.0%')).toBeInTheDocument();
    });

    it('should show REAL verdict for low score', () => {
      const data = createMockResult({ is_ai: false, score: 0.2 });
      render(<AnalysisResultCard data={data} />);

      expect(screen.getByText('PRAWDZIWE ZDJĘCIE')).toBeInTheDocument();
      expect(screen.getByText('20.0%')).toBeInTheDocument();
    });
  });

  describe('custom metrics', () => {
    it('should display metrics when provided', () => {
      const data = createMockResult({
        custom_metrics: {
          blur_score: 0.123,
          edge_density: 0.456,
        },
      });
      render(<AnalysisResultCard data={data} />);

      expect(screen.getByText('Analiza Szczegółowa')).toBeInTheDocument();
      expect(screen.getByText('blur score')).toBeInTheDocument();
    });

    it('should NOT display metrics section when empty', () => {
      const data = createMockResult({ custom_metrics: {} });
      render(<AnalysisResultCard data={data} />);

      expect(screen.queryByText('Analiza Szczegółowa')).not.toBeInTheDocument();
    });

    it('should expand to show all metrics', async () => {
      const user = userEvent.setup();
      const data = createMockResult({
        custom_metrics: {
          metric1: 0.1,
          metric2: 0.2,
          metric3: 0.3,
          metric4: 0.4,
          metric5: 0.5,
        },
      });
      render(<AnalysisResultCard data={data} />);

      // metric5 should be hidden initially
      expect(screen.queryByText('metric5')).not.toBeInTheDocument();

      await user.click(screen.getByText(/Pokaż wszystkie/));

      expect(screen.getByText('metric5')).toBeInTheDocument();
    });

    it('should format metric values correctly', () => {
      const data = createMockResult({
        custom_metrics: {
          float_metric: 0.12345,
          int_metric: 42,
          bool_metric: true,
        },
      });
      render(<AnalysisResultCard data={data} />);

      expect(screen.getByText('0.123')).toBeInTheDocument();
      expect(screen.getByText('42')).toBeInTheDocument();
      expect(screen.getByText('Tak')).toBeInTheDocument();
    });
  });

  describe('heatmap', () => {
    it('should disable heatmap button for guest (negative id)', () => {
      const data = createMockResult({ id: -1 });
      render(<AnalysisResultCard data={data} isGuest={true} />);

      expect(screen.getByRole('button', { name: /Pokaż GradCAM/i })).toBeDisabled();
      expect(screen.getByText(/Niedostępne dla gości/i)).toBeInTheDocument();
    });

    it('should fetch and display heatmap when clicked', async () => {
      const user = userEvent.setup();
      const mockBlob = new Blob(['heatmap'], { type: 'image/jpeg' });
      mockGet.mockResolvedValueOnce({ data: mockBlob });
      mockPost.mockResolvedValueOnce({ data: { saved: true } });

      const data = createMockResult({ id: 5 });
      render(<AnalysisResultCard data={data} />);

      await user.click(screen.getByRole('button', { name: /Pokaż GradCAM/i }));

      await waitFor(() => {
        expect(mockGet).toHaveBeenCalledWith('/analysis/5/heatmap', { responseType: 'blob' });
        expect(screen.getByText('Heatmapa Uwagi')).toBeInTheDocument();
      });
    });

    it('should show error on fetch failure', async () => {
      const user = userEvent.setup();
      mockGet.mockRejectedValueOnce({ response: { status: 500 } });

      const data = createMockResult({ id: 5 });
      render(<AnalysisResultCard data={data} />);

      await user.click(screen.getByRole('button', { name: /Pokaż GradCAM/i }));

      await waitFor(() => {
        expect(screen.getByText(/Błąd generowania heatmapy/i)).toBeInTheDocument();
      });
    });

    it('should show specific error for unsupported model (400)', async () => {
      const user = userEvent.setup();
      mockGet.mockRejectedValueOnce({ response: { status: 400 } });

      const data = createMockResult({ id: 5 });
      render(<AnalysisResultCard data={data} />);

      await user.click(screen.getByRole('button', { name: /Pokaż GradCAM/i }));

      await waitFor(() => {
        expect(screen.getByText(/Heatmapa niedostępna dla tego typu modelu/i)).toBeInTheDocument();
      });
    });

    it('should auto-save heatmap for logged in users', async () => {
      const user = userEvent.setup();
      const mockBlob = new Blob(['heatmap'], { type: 'image/jpeg' });
      mockGet.mockResolvedValueOnce({ data: mockBlob });
      mockPost.mockResolvedValueOnce({ data: { saved: true } });

      const data = createMockResult({ id: 5 });
      render(<AnalysisResultCard data={data} isGuest={false} />);

      await user.click(screen.getByRole('button', { name: /Pokaż GradCAM/i }));

      await waitFor(() => {
        expect(mockPost).toHaveBeenCalledWith('/analysis/5/heatmap/save');
      });
    });

    it('should NOT auto-save for guests', async () => {
      const user = userEvent.setup();
      mockGet.mockResolvedValueOnce({ data: new Blob(['x']) });

      const data = createMockResult({ id: 5 });
      render(<AnalysisResultCard data={data} isGuest={true} />);

      await user.click(screen.getByRole('button', { name: /Pokaż GradCAM/i }));

      await waitFor(() => {
        expect(mockGet).toHaveBeenCalled();
      });
      expect(mockPost).not.toHaveBeenCalled();
    });

    it('should show controls after heatmap loads', async () => {
      const user = userEvent.setup();
      mockGet.mockResolvedValueOnce({ data: new Blob(['x']) });
      mockPost.mockResolvedValueOnce({ data: {} });

      const data = createMockResult({ id: 5 });
      render(<AnalysisResultCard data={data} />);

      await user.click(screen.getByRole('button', { name: /Pokaż GradCAM/i }));

      await waitFor(() => {
        expect(screen.getByText(/Transparentność/)).toBeInTheDocument();
        expect(screen.getByRole('slider')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Pobierz/i })).toBeInTheDocument();
      });
    });
  });

  describe('state reset', () => {
    it('should reset heatmap when data changes', async () => {
      const user = userEvent.setup();
      mockGet.mockResolvedValue({ data: new Blob(['x']) });
      mockPost.mockResolvedValue({ data: {} });

      const data1 = createMockResult({ id: 1, filename: 'image1.jpg' });
      const { rerender } = render(<AnalysisResultCard data={data1} />);

      await user.click(screen.getByRole('button', { name: /Pokaż GradCAM/i }));
      await waitFor(() => expect(screen.getByText('Heatmapa Uwagi')).toBeInTheDocument());

      const data2 = createMockResult({ id: 2, filename: 'image2.jpg' });
      rerender(<AnalysisResultCard data={data2} />);

      await waitFor(() => {
        expect(screen.queryByText('Heatmapa Uwagi')).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Pokaż GradCAM/i })).toBeInTheDocument();
      });
    });
  });
});