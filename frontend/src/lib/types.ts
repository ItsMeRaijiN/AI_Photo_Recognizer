export interface AnalysisResult {
  id: number;
  filename: string;
  file_path: string;

  is_ai: boolean;
  score: number;
  confidence: number;
  threshold_used: number;
  inference_time_ms: number;

  model_type: string;
  backbone_name: string;
  model_version?: string | null;

  has_heatmap: boolean;
  heatmap_path?: string | null;

  custom_metrics?: Record<string, MetricResult> | null;

  created_at: string;
  source_index?: number | null;

  previewUrl?: string;

  from_cache?: boolean;
}

export type MetricResult = number | string | boolean | MetricObject | null;

export interface MetricObject {
  [key: string]: number | string | boolean | null | undefined;
  is_suspicious?: boolean;
}

export interface BatchUploadResponse {
  total: number;
  processed: number;
  failed: number;
  results: AnalysisResult[];
  errors: string[];
  total_inference_time_ms: number;
}

export interface User {
  id: number;
  username: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
}

export interface UserStats {
  total_analyses: number;
  ai_detections: number;
  human_detections: number;
  ai_ratio: number;
}

export interface SystemStats {
  total_users: number;
  active_users: number;
  total_analyses: number;
  ai_detections: number;
  human_detections: number;
  ai_ratio: number;
  analyses_today: number;
  storage_used_mb: number;
  model_info: ModelInfo;
  metrics_count?: number;
}

export interface ModelInfo {
  type: string;
  backbone: string;
  version?: string | null;
  threshold: number;
  device: string;
  loaded: boolean;
  image_size?: number;
}

export function getModelDisplayName(result: AnalysisResult): string {
  const backbone = result.backbone_name || 'unknown';
  const type = result.model_type || 'unknown';

  const backboneNames: Record<string, string> = {
    'effnetv2': 'EfficientNet V2-S',
    'efficientnet': 'EfficientNet V2-S',
    'convnext': 'ConvNeXt V2',
    'convnextv2': 'ConvNeXt V2',
  };

  const lowerBackbone = backbone.toLowerCase();
  let friendlyBackbone = backbone;

  for (const [key, name] of Object.entries(backboneNames)) {
    if (lowerBackbone.includes(key)) {
      friendlyBackbone = name;
      break;
    }
  }

  return `${friendlyBackbone} (${type})`;
}

export function isMetricSuspicious(value: MetricResult): boolean {
  if (value === null || value === undefined) return false;
  if (typeof value === 'object' && 'is_suspicious' in value) {
    return value.is_suspicious === true;
  }
  return false;
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}min`;
}
