import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import AuthImage from '../components/AuthImage';

const mockGet = jest.fn();
jest.mock('../lib/api', () => ({
  api: {
    get: (...args: unknown[]) => mockGet(...args),
  },
  endpoints: {
    analysisImage: (id: number) => `/analysis/${id}/image`,
  },
}));

const mockCreateObjectURL = jest.fn(() => 'blob:http://localhost/mock-blob');
const mockRevokeObjectURL = jest.fn();
global.URL.createObjectURL = mockCreateObjectURL;
global.URL.revokeObjectURL = mockRevokeObjectURL;

describe('AuthImage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGet.mockReset();
  });


  describe('fallback image', () => {
    it('should render fallback image immediately without API call', () => {
      render(
        <AuthImage
          filePath="/some/path.jpg"
          fallbackSrc="http://example.com/preview.jpg"
          alt="Preview"
        />
      );

      const img = screen.getByAltText('Preview');
      expect(img).toBeInTheDocument();
      expect(img).toHaveAttribute('src', 'http://example.com/preview.jpg');
      expect(mockGet).not.toHaveBeenCalled();
    });

    it('should apply className to fallback image', () => {
      render(
        <AuthImage
          filePath="/path.jpg"
          fallbackSrc="http://example.com/img.jpg"
          alt="Test"
          className="w-full rounded-lg"
        />
      );

      expect(screen.getByAltText('Test')).toHaveClass('w-full', 'rounded-lg');
    });
  });


  describe('special path handling', () => {
    it('should show error for "memory" path without fallback', async () => {
      render(<AuthImage filePath="memory" alt="Test" />);

      await waitFor(() => {
        expect(screen.getByText('Podgląd niedostępny')).toBeInTheDocument();
      });
      expect(mockGet).not.toHaveBeenCalled();
    });

    it('should show error for "upload://memory" path without fallback', async () => {
      render(<AuthImage filePath="upload://memory" alt="Test" />);

      await waitFor(() => {
        expect(screen.getByText('Podgląd niedostępny')).toBeInTheDocument();
      });
      expect(mockGet).not.toHaveBeenCalled();
    });

    it('should show error for empty path without fallback', async () => {
      render(<AuthImage filePath="" alt="Test" />);

      await waitFor(() => {
        expect(screen.getByText('Podgląd niedostępny')).toBeInTheDocument();
      });
      expect(mockGet).not.toHaveBeenCalled();
    });
  });


  describe('API fetch', () => {
    it('should fetch image from API when analysisId is provided', async () => {
      const mockBlob = new Blob(['image-data'], { type: 'image/jpeg' });
      mockGet.mockResolvedValue({ data: mockBlob });

      render(
        <AuthImage
          filePath="/uploads/test.jpg"
          analysisId={42}
          alt="Fetched image"
        />
      );

      await waitFor(() => {
        expect(mockGet).toHaveBeenCalledWith('/analysis/42/image', {
          responseType: 'blob',
        });
      });

      await waitFor(() => {
        const img = screen.getByAltText('Fetched image');
        expect(img).toHaveAttribute('src', 'blob:http://localhost/mock-blob');
      });
    });

    it('should show error when API fetch fails', async () => {
      const consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      mockGet.mockRejectedValue(new Error('Network error'));

      render(
        <AuthImage
          filePath="/uploads/test.jpg"
          analysisId={123}
          alt="Failed image"
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Podgląd niedostępny')).toBeInTheDocument();
      });

      consoleWarnSpy.mockRestore();
    });

    it('should show error when analysisId is not provided and no fallback', async () => {
      render(
        <AuthImage
          filePath="/uploads/test.jpg"
          alt="No ID"
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Podgląd niedostępny')).toBeInTheDocument();
      });
      expect(mockGet).not.toHaveBeenCalled();
    });

    it('should not fetch when analysisId is 0', async () => {
      render(
        <AuthImage filePath="/path.jpg" analysisId={0} alt="Zero ID" />
      );

      await waitFor(() => {
        expect(screen.getByText('Podgląd niedostępny')).toBeInTheDocument();
      });
      expect(mockGet).not.toHaveBeenCalled();
    });

    it('should not fetch when analysisId is negative', async () => {
      render(
        <AuthImage filePath="/path.jpg" analysisId={-1} alt="Negative ID" />
      );

      await waitFor(() => {
        expect(screen.getByText('Podgląd niedostępny')).toBeInTheDocument();
      });
      expect(mockGet).not.toHaveBeenCalled();
    });
  });

  describe('loading state', () => {
    it('should show spinner while fetching', async () => {
      let resolvePromise: (value: unknown) => void;
      mockGet.mockReturnValue(new Promise(resolve => { resolvePromise = resolve; }));

      render(
        <AuthImage
          filePath="/path.jpg"
          analysisId={1}
          alt="Loading"
        />
      );

      expect(document.querySelector('.animate-spin')).toBeInTheDocument();

      resolvePromise!({ data: new Blob(['x'], { type: 'image/png' }) });

      await waitFor(() => {
        expect(screen.getByAltText('Loading')).toBeInTheDocument();
      });
    });

    it('should NOT show spinner when fallbackSrc is provided', () => {
      render(
        <AuthImage
          filePath="/path.jpg"
          fallbackSrc="http://example.com/img.jpg"
          alt="Instant"
        />
      );

      expect(document.querySelector('.animate-spin')).not.toBeInTheDocument();
      expect(screen.getByAltText('Instant')).toBeInTheDocument();
    });
  });

  describe('cleanup', () => {
    it('should revoke object URL on unmount', async () => {
      mockGet.mockResolvedValue({ data: new Blob(['x'], { type: 'image/png' }) });

      const { unmount } = render(
        <AuthImage
          filePath="/path.jpg"
          analysisId={1}
          alt="To be unmounted"
        />
      );

      await waitFor(() => {
        expect(screen.getByAltText('To be unmounted')).toBeInTheDocument();
      });

      unmount();

      expect(mockRevokeObjectURL).toHaveBeenCalled();
    });
  });

  describe('prop changes', () => {
    it('should refetch when analysisId changes', async () => {
      mockGet.mockResolvedValue({ data: new Blob(['x'], { type: 'image/png' }) });

      const { rerender } = render(
        <AuthImage filePath="/path.jpg" analysisId={1} alt="Test" />
      );

      await waitFor(() => {
        expect(mockGet).toHaveBeenCalledWith('/analysis/1/image', expect.anything());
      });

      rerender(
        <AuthImage filePath="/other.jpg" analysisId={2} alt="Test" />
      );

      await waitFor(() => {
        expect(mockGet).toHaveBeenCalledWith('/analysis/2/image', expect.anything());
      });
    });

    it('should switch to fallback when fallbackSrc is added', async () => {
      mockGet.mockResolvedValue({ data: new Blob(['x'], { type: 'image/png' }) });

      const { rerender } = render(
        <AuthImage filePath="/path.jpg" analysisId={1} alt="Test" />
      );

      await waitFor(() => {
        expect(screen.getByAltText('Test')).toHaveAttribute('src', 'blob:http://localhost/mock-blob');
      });

      rerender(
        <AuthImage
          filePath="/path.jpg"
          analysisId={1}
          fallbackSrc="http://new-fallback.jpg"
          alt="Test"
        />
      );

      await waitFor(() => {
        expect(screen.getByAltText('Test')).toHaveAttribute('src', 'http://new-fallback.jpg');
      });
    });
  });
});