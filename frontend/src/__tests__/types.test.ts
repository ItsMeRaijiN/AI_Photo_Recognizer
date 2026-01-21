import {
  getModelDisplayName,
  isMetricSuspicious,
  formatFileSize,
  formatDuration,
  AnalysisResult,
  MetricResult,
  MetricObject,
} from '@/lib/types';

const createMockResult = (backbone: string, modelType: string): AnalysisResult => ({
  id: 1,
  filename: 'test.jpg',
  file_path: '/path/test.jpg',
  is_ai: false,
  score: 0.5,
  confidence: 0.75,
  threshold_used: 0.5,
  inference_time_ms: 100,
  model_type: modelType,
  backbone_name: backbone,
  has_heatmap: false,
  created_at: new Date().toISOString(),
});

describe('types.ts helper functions', () => {
  describe('getModelDisplayName', () => {
    it('should return friendly name for EfficientNet variants', () => {
      expect(getModelDisplayName(createMockResult('effnetv2', 'torch')))
        .toBe('EfficientNet V2-S (torch)');
      expect(getModelDisplayName(createMockResult('efficientnet', 'onnx')))
        .toBe('EfficientNet V2-S (onnx)');
    });

    it('should return original name for unknown backbones', () => {
      expect(getModelDisplayName(createMockResult('vit_large', 'torch')))
        .toBe('vit_large (torch)');
      expect(getModelDisplayName(createMockResult('resnet50', 'onnx')))
        .toBe('resnet50 (onnx)');
    });

    it('should handle empty/missing backbone name', () => {
      expect(getModelDisplayName(createMockResult('', 'torch')))
        .toBe('unknown (torch)');
    });
  });

  describe('isMetricSuspicious', () => {
    it('should return false for null/undefined values', () => {
      expect(isMetricSuspicious(null)).toBe(false);
      expect(isMetricSuspicious(undefined as unknown as MetricResult)).toBe(false);
    });

    it('should return true when object has is_suspicious: true', () => {
      const suspicious: MetricObject = { value: 0.8, is_suspicious: true };
      expect(isMetricSuspicious(suspicious)).toBe(true);
    });

    it('should return false when object lacks is_suspicious property', () => {
      const noFlag: MetricObject = { blur_score: 0.1, edge_density: 0.5 };
      expect(isMetricSuspicious(noFlag)).toBe(false);
    });

  });

  describe('formatFileSize', () => {
    it('should return "0 B" for zero bytes', () => {
      expect(formatFileSize(0)).toBe('0 B');
    });

    it('should format kilobytes (1KB - 1MB)', () => {
      expect(formatFileSize(1024)).toBe('1 KB');
      expect(formatFileSize(1536)).toBe('1.5 KB');
      expect(formatFileSize(1024 * 500)).toBe('500 KB');
    });

    it('should format megabytes (1MB - 1GB)', () => {
      expect(formatFileSize(1024 * 1024)).toBe('1 MB');
      expect(formatFileSize(1024 * 1024 * 1.5)).toBe('1.5 MB');
      expect(formatFileSize(1024 * 1024 * 10)).toBe('10 MB');
    });

    it('should format gigabytes (>= 1GB)', () => {
      expect(formatFileSize(1024 * 1024 * 1024)).toBe('1 GB');
      expect(formatFileSize(1024 * 1024 * 1024 * 2.5)).toBe('2.5 GB');
    });
  });

  describe('formatDuration', () => {
    it('should format milliseconds (< 1s)', () => {
      expect(formatDuration(0)).toBe('0ms');
      expect(formatDuration(1)).toBe('1ms');
      expect(formatDuration(500)).toBe('500ms');
      expect(formatDuration(999)).toBe('999ms');
    });

    it('should format seconds (1s - 60s)', () => {
      expect(formatDuration(1000)).toBe('1.0s');
      expect(formatDuration(1500)).toBe('1.5s');
      expect(formatDuration(5000)).toBe('5.0s');
      expect(formatDuration(59999)).toBe('60.0s');
    });

    it('should format minutes (>= 60s)', () => {
      expect(formatDuration(60000)).toBe('1.0min');
      expect(formatDuration(90000)).toBe('1.5min');
      expect(formatDuration(300000)).toBe('5.0min');
    });
  });
});