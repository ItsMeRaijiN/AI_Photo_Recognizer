jest.mock('axios', () => {
  const mockFns = {
    post: jest.fn(),
    get: jest.fn(),
    delete: jest.fn(),
    patch: jest.fn(),
  };

  return {
    create: jest.fn(() => ({
      interceptors: {
        request: { use: jest.fn() },
        response: { use: jest.fn() },
      },
      ...mockFns,
    })),
    __mocks__: mockFns,
  };
});

import { endpoints, uploadBatch, analyzeFolder } from '@/lib/api';

const axios = jest.requireMock('axios') as { __mocks__: { post: jest.Mock } };
const mockPost = axios.__mocks__.post;

describe('API module', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('endpoints - static paths', () => {
    it('should have correct root and health endpoints', () => {
      expect(endpoints.root).toBe('/');
      expect(endpoints.health).toBe('/health');
    });

    it('should have correct auth endpoints', () => {
      expect(endpoints.login).toBe('/auth/token');
      expect(endpoints.register).toBe('/auth/register');
      expect(endpoints.me).toBe('/auth/me');
      expect(endpoints.myStats).toBe('/auth/me/stats');
    });

    it('should have correct analysis endpoints', () => {
      expect(endpoints.predict).toBe('/analysis/predict');
      expect(endpoints.predictBatch).toBe('/analysis/predict/batch');
      expect(endpoints.analyzeFolder).toBe('/analysis/folder');
      expect(endpoints.history).toBe('/analysis/history');
      expect(endpoints.batchStart).toBe('/analysis/batch/start');
      expect(endpoints.modelInfo).toBe('/analysis/model/info');
      expect(endpoints.availableMetrics).toBe('/analysis/metrics/available');
    });

    it('should have correct admin endpoints', () => {
      expect(endpoints.adminUsers).toBe('/admin/users');
      expect(endpoints.adminStats).toBe('/admin/stats');
      expect(endpoints.adminCleanup).toBe('/admin/cleanup');
      expect(endpoints.adminReloadMetrics).toBe('/admin/metrics/reload');
      expect(endpoints.adminOptimizeDb).toBe('/admin/optimize-db');
      expect(endpoints.adminUploadModel).toBe('/admin/upload-model');
      expect(endpoints.adminBootstrap).toBe('/admin/bootstrap');
    });
  });

  describe('endpoints - dynamic paths', () => {
    it('should generate correct analysis paths with id', () => {
      expect(endpoints.analysis(1)).toBe('/analysis/1');
      expect(endpoints.analysis(999)).toBe('/analysis/999');
      expect(endpoints.analysisImage(42)).toBe('/analysis/42/image');
    });

    it('should generate correct heatmap paths', () => {
      expect(endpoints.heatmap(1)).toBe('/analysis/1/heatmap');
      expect(endpoints.heatmap(1, false)).toBe('/analysis/1/heatmap');
      expect(endpoints.heatmap(1, true)).toBe('/analysis/1/heatmap?download=true');
      expect(endpoints.saveHeatmap(5)).toBe('/analysis/5/heatmap/save');
    });

    it('should generate correct batch paths with jobId', () => {
      expect(endpoints.batchStatus('abc-123')).toBe('/analysis/batch/abc-123');
      expect(endpoints.batchStream('job-uuid')).toBe('/analysis/batch/job-uuid/stream');
    });

    it('should generate correct admin user paths', () => {
      expect(endpoints.adminUser(10)).toBe('/admin/users/10');
      expect(endpoints.adminToggleUser(10)).toBe('/admin/users/10/toggle-active');
      expect(endpoints.adminToggleAdmin(10)).toBe('/admin/users/10/toggle-admin');
    });
  });


  describe('uploadBatch function', () => {
    it('should send files as FormData to predictBatch endpoint', async () => {
      const mockResponse = {
        data: {
          total: 2,
          processed: 2,
          failed: 0,
          results: [],
          errors: [],
          total_inference_time_ms: 200,
        }
      };
      mockPost.mockResolvedValueOnce(mockResponse);

      const files = [
        new File(['content1'], 'test1.jpg', { type: 'image/jpeg' }),
        new File(['content2'], 'test2.jpg', { type: 'image/jpeg' }),
      ];

      const result = await uploadBatch(files);

      expect(mockPost).toHaveBeenCalledWith(
        endpoints.predictBatch,
        expect.any(FormData),
        expect.any(Object)
      );
      expect(result).toEqual(mockResponse.data);
    });

    it('should call onProgress callback with upload percentage', async () => {
      mockPost.mockImplementation((_url: string, _data: FormData, config?: { onUploadProgress?: (e: { loaded: number; total: number }) => void }) => {
        if (config?.onUploadProgress) {
          config.onUploadProgress({ loaded: 50, total: 100 });
          config.onUploadProgress({ loaded: 100, total: 100 });
        }
        return Promise.resolve({ data: { total: 1, processed: 1 } });
      });

      const onProgress = jest.fn();
      const files = [new File(['test'], 'test.jpg', { type: 'image/jpeg' })];

      await uploadBatch(files, { onProgress });

      expect(onProgress).toHaveBeenCalledWith(50);
      expect(onProgress).toHaveBeenCalledWith(100);
    });

    it('should propagate API errors', async () => {
      const error = new Error('Network error');
      mockPost.mockRejectedValueOnce(error);

      const files = [new File(['test'], 'test.jpg', { type: 'image/jpeg' })];

      await expect(uploadBatch(files)).rejects.toThrow('Network error');
    });
  });

  describe('analyzeFolder function', () => {
    it('should send folder path to analyzeFolder endpoint', async () => {
      const mockResponse = {
        data: {
          total: 5,
          processed: 5,
          results: [],
        }
      };
      mockPost.mockResolvedValueOnce(mockResponse);

      const result = await analyzeFolder('/path/to/images');

      expect(mockPost).toHaveBeenCalledWith(
        endpoints.analyzeFolder,
        {
          path: '/path/to/images',
          recursive: false,
          max_images: 100,
        }
      );
      expect(result).toEqual(mockResponse.data);
    });

    it('should pass recursive option when provided', async () => {
      mockPost.mockResolvedValueOnce({ data: {} });

      await analyzeFolder('/path', { recursive: true });

      expect(mockPost).toHaveBeenCalledWith(
        endpoints.analyzeFolder,
        expect.objectContaining({ recursive: true })
      );
    });

    it('should pass maxImages option when provided', async () => {
      mockPost.mockResolvedValueOnce({ data: {} });

      await analyzeFolder('/path', { maxImages: 50 });

      expect(mockPost).toHaveBeenCalledWith(
        endpoints.analyzeFolder,
        expect.objectContaining({ max_images: 50 })
      );
    });

    it('should use default values when options not provided', async () => {
      mockPost.mockResolvedValueOnce({ data: {} });

      await analyzeFolder('/path');

      expect(mockPost).toHaveBeenCalledWith(
        endpoints.analyzeFolder,
        {
          path: '/path',
          recursive: false,
          max_images: 100,
        }
      );
    });

    it('should propagate API errors', async () => {
      mockPost.mockRejectedValueOnce(new Error('Folder not found'));

      await expect(analyzeFolder('/invalid/path')).rejects.toThrow('Folder not found');
    });
  });
});