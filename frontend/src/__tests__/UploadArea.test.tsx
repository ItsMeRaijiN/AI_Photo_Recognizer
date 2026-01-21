import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import UploadArea from '../components/UploadArea';

const mockOnDrop = jest.fn();

jest.mock('react-dropzone', () => ({
  useDropzone: ({ onDrop, disabled }: { onDrop: (files: File[]) => void; disabled?: boolean }) => {
    mockOnDrop.mockImplementation(onDrop);

    return {
      getRootProps: () => ({
        'data-testid': 'dropzone',
        'data-disabled': disabled ? 'true' : 'false',
      }),
      getInputProps: () => ({
        'data-testid': 'dropzone-input',
      }),
      isDragActive: false,
    };
  },
}));

beforeAll(() => {
  global.fetch = jest.fn();
  (globalThis as any).alert = jest.fn();
});

describe('UploadArea', () => {
  const mockOnFileSelect = jest.fn();
  const mockOnFolderAnalyze = jest.fn().mockResolvedValue(undefined);
  const mockOnBatchUpload = jest.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    jest.clearAllMocks();
    (global.fetch as jest.Mock).mockReset();
  });

  it('renders the basic upload UI', () => {
    render(<UploadArea onFileSelect={mockOnFileSelect} isLoading={false} progress={0} />);

    expect(screen.getByTestId('dropzone')).toBeInTheDocument();
    expect(screen.getByText(/Przeciągnij, wybierz lub wklej/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Wklej link do obrazka/i)).toBeInTheDocument();
  });

  it('disables URL analyze button when URL is empty', () => {
    render(<UploadArea onFileSelect={mockOnFileSelect} isLoading={false} progress={0} />);

    const button = screen.getByRole('button', { name: /Analizuj/i });
    expect(button).toBeDisabled();
  });

  it('calls onFileSelect when a single image file is chosen via hidden file input', async () => {
    const user = userEvent.setup();
    const { container } = render(
      <UploadArea onFileSelect={mockOnFileSelect} isLoading={false} progress={0} />
    );

    const file = new File(['dummy'], 'test.png', { type: 'image/png' });

    const fileInput = container.querySelector('input[type="file"][accept="image/*"]') as HTMLInputElement;
    expect(fileInput).toBeTruthy();

    await user.upload(fileInput, file);

    expect(mockOnFileSelect).toHaveBeenCalledTimes(1);
    expect(mockOnFileSelect).toHaveBeenCalledWith([file]);
  });

  it('calls onBatchUpload when multiple image files are chosen and onBatchUpload is provided', async () => {
    const user = userEvent.setup();
    const { container } = render(
      <UploadArea
        onFileSelect={mockOnFileSelect}
        onBatchUpload={mockOnBatchUpload}
        isLoading={false}
        progress={0}
      />
    );

    const f1 = new File(['a'], 'a.jpg', { type: 'image/jpeg' });
    const f2 = new File(['b'], 'b.png', { type: 'image/png' });

    const fileInput = container.querySelector('input[type="file"][accept="image/*"]') as HTMLInputElement;
    expect(fileInput).toBeTruthy();

    await user.upload(fileInput, [f1, f2]);

    expect(mockOnBatchUpload).toHaveBeenCalledTimes(1);
    expect(mockOnBatchUpload).toHaveBeenCalledWith([f1, f2]);
    expect(mockOnFileSelect).not.toHaveBeenCalled();
  });

it('shows alert when selected folder/files contain no images', async () => {
  const user = userEvent.setup();
  const { container } = render(
    <UploadArea onFileSelect={mockOnFileSelect} isLoading={false} progress={0} />
  );

  const nonImage = new File(['x'], 'notes.txt', { type: 'text/plain' });
  const folderInput = container.querySelector('input[type="file"][webkitdirectory]') as HTMLInputElement;
  expect(folderInput).toBeTruthy();

  await user.upload(folderInput, nonImage);

  expect((globalThis as any).alert).toHaveBeenCalledWith(
    'Nie znaleziono plików obrazów w wybranej lokalizacji.'
  );
  expect(mockOnFileSelect).not.toHaveBeenCalled();
});

  describe('folder analysis', () => {
    it('does not render folder toggle when onFolderAnalyze is not provided', () => {
      render(<UploadArea onFileSelect={mockOnFileSelect} isLoading={false} progress={0} />);
      expect(screen.queryByText(/Analiza ścieżki/i)).not.toBeInTheDocument();
    });

    it('renders folder toggle when onFolderAnalyze is provided', () => {
      render(
        <UploadArea
          onFileSelect={mockOnFileSelect}
          onFolderAnalyze={mockOnFolderAnalyze}
          isLoading={false}
          progress={0}
        />
      );

      expect(screen.getByText(/Analiza ścieżki/i)).toBeInTheDocument();
    });

    it('calls onFolderAnalyze with path and recursive flag', async () => {
      const user = userEvent.setup();

      render(
        <UploadArea
          onFileSelect={mockOnFileSelect}
          onFolderAnalyze={mockOnFolderAnalyze}
          isLoading={false}
          progress={0}
        />
      );

      await user.click(screen.getByText(/Analiza ścieżki/i));

      const folderPathInput = screen.getByPlaceholderText(/Pictures|image\.jpg/i);
      await user.type(folderPathInput, 'C:\\Images');

      const checkbox = screen.getByRole('checkbox');
      await user.click(checkbox);

      const folderAnalyzeButton = screen.getAllByRole('button', { name: /Analizuj/i })[0];
      expect(folderAnalyzeButton).not.toBeDisabled();

      await user.click(folderAnalyzeButton);

      await waitFor(() => {
        expect(mockOnFolderAnalyze).toHaveBeenCalledTimes(1);
        expect(mockOnFolderAnalyze).toHaveBeenCalledWith('C:\\Images', true);
      });
    });

    it('disables folder analyze button when path is empty', async () => {
      const user = userEvent.setup();

      render(
        <UploadArea
          onFileSelect={mockOnFileSelect}
          onFolderAnalyze={mockOnFolderAnalyze}
          isLoading={false}
          progress={0}
        />
      );

      await user.click(screen.getByText(/Analiza ścieżki/i));

      const folderAnalyzeButton = screen.getAllByRole('button', { name: /Analizuj/i })[0];
      expect(folderAnalyzeButton).toBeDisabled();
    });
  });

  describe('URL upload', () => {
    let errSpy: jest.SpyInstance;
    let logSpy: jest.SpyInstance;

    beforeEach(() => {
      errSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    });

    afterEach(() => {
      errSpy.mockRestore();
      logSpy.mockRestore();
    });

    it('fetches image and calls onFileSelect on success', async () => {
      const user = userEvent.setup();

      const blob = new Blob(['img'], { type: 'image/jpeg' });
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        status: 200,
        blob: async () => blob,
      });

      render(<UploadArea onFileSelect={mockOnFileSelect} isLoading={false} progress={0} />);

      await user.type(screen.getByPlaceholderText(/Wklej link do obrazka/i), 'https://example.com/test.jpg');
      await user.click(screen.getAllByRole('button', { name: /Analizuj/i })[0]);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('https://example.com/test.jpg');
        expect(mockOnFileSelect).toHaveBeenCalledWith([expect.any(File)]);
      });
    });

    it('shows alert on network error', async () => {
      const user = userEvent.setup();
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

      render(<UploadArea onFileSelect={mockOnFileSelect} isLoading={false} progress={0} />);

      await user.type(screen.getByPlaceholderText(/Wklej link do obrazka/i), 'https://example.com/test.jpg');
      await user.click(screen.getAllByRole('button', { name: /Analizuj/i })[0]);

      await waitFor(() => {
        expect((globalThis as any).alert).toHaveBeenCalled();
      });
    });

    it('shows alert when response is not ok', async () => {
      const user = userEvent.setup();
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        blob: async () => new Blob(),
      });

      render(<UploadArea onFileSelect={mockOnFileSelect} isLoading={false} progress={0} />);

      await user.type(screen.getByPlaceholderText(/Wklej link do obrazka/i), 'https://example.com/test.jpg');
      await user.click(screen.getAllByRole('button', { name: /Analizuj/i })[0]);

      await waitFor(() => {
        expect((globalThis as any).alert).toHaveBeenCalled();
      });
    });

    it('clears URL input after successful upload', async () => {
      const user = userEvent.setup();

      const blob = new Blob(['img'], { type: 'image/png' });
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        status: 200,
        blob: async () => blob,
      });

      render(<UploadArea onFileSelect={mockOnFileSelect} isLoading={false} progress={0} />);

      const urlInput = screen.getByPlaceholderText(/Wklej link do obrazka/i);
      await user.type(urlInput, 'https://example.com/image.png');
      await user.click(screen.getAllByRole('button', { name: /Analizuj/i })[0]);

      await waitFor(() => {
        expect(urlInput).toHaveValue('');
      });
    });
  });

  it('handles paste event with image file (clipboard) and calls onFileSelect', () => {
    render(<UploadArea onFileSelect={mockOnFileSelect} isLoading={false} progress={0} />);

    const pasted = new File(['img'], 'pasted.png', { type: 'image/png' });

    const pasteEvent = new Event('paste') as any;
    pasteEvent.clipboardData = {
      items: [
        {
          type: 'image/png',
          getAsFile: () => pasted,
        },
      ],
    };

    fireEvent(window, pasteEvent);

    expect(mockOnFileSelect).toHaveBeenCalledWith([pasted]);
  });
});
