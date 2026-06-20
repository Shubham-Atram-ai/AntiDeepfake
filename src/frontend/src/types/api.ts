/**
 * api.ts — TypeScript types derived from the AntiDeepfake backend schemas.
 *
 * Source of truth: src/backend/schemas/response.py
 *
 * Notable discrepancy resolved:
 *   MetricsResponse.psnr is Optional[float] in Python → `number | null` here,
 *   because identical images produce infinite PSNR which the backend serialises
 *   as JSON `null`.
 */

// ---------------------------------------------------------------------------
// Success types
// ---------------------------------------------------------------------------

/** Nested image-quality metrics nested inside CloakResponse. */
export interface Metrics {
  /** Structural Similarity Index in [-1.0, 1.0]. Values ≥ 0.90 are imperceptible. */
  ssim: number;
  /**
   * Peak Signal-to-Noise Ratio in dB.  >40 dB is practically imperceptible.
   * `null` is returned for identical images (infinite PSNR).
   */
  psnr: number | null;
}

/** Success response from POST /api/v1/cloak */
export interface CloakResponse {
  /** Always `true` on a successful cloaking request. */
  success: true;
  /** Wall-clock server-side processing time in milliseconds. */
  processing_time_ms: number;
  /** SSIM and PSNR scores comparing original vs cloaked image. */
  metrics: Metrics;
  /**
   * Base64-encoded JPEG of the cloaked image.
   * Display via: `data:image/jpeg;base64,${cloaked_image_base64}`
   */
  cloaked_image_base64: string;
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/** Structured error response for all failure cases from the backend. */
export interface ErrorResponse {
  /** Always `false` for error responses. */
  success: false;
  /**
   * Machine-readable uppercase error code.
   * Known values: NO_FACE_DETECTED | INVALID_IMAGE | PROCESSING_ERROR
   */
  error_code: string;
  /** Human-readable error description. */
  error: string;
}

// ---------------------------------------------------------------------------
// Health check types
// ---------------------------------------------------------------------------

/** Response from GET /health */
export interface HealthResponse {
  /** Overall API health: "healthy" | "degraded" */
  status: 'healthy' | 'degraded' | string;
  /** Semantic version string of the running API. */
  version: string;
  /** True when MTCNN face detector is ready. */
  face_detector_loaded: boolean;
  /** True when InceptionResnetV1 FGSM engine is ready. */
  fgsm_engine_loaded: boolean;
}

// ---------------------------------------------------------------------------
// Frontend state types
// ---------------------------------------------------------------------------

/** Possible backend connectivity states used by useHealthCheck. */
export type BackendStatus =
  | 'checking'
  | 'online'
  | 'degraded'
  | 'offline';

/** Upload + processing state managed in Home.tsx */
export interface AppState {
  /** Currently selected image file, or null if none selected. */
  file: File | null;
  /** Object URL for the original image preview. */
  previewUrl: string | null;
  /** FGSM perturbation budget (0.01 – 0.10). */
  epsilon: number;
  /** True while the API request is in-flight. */
  loading: boolean;
  /** Successful cloaking response, or null. */
  result: CloakResponse | null;
  /** Error state: human-readable message or null. */
  error: string | null;
}

/** Props for the image upload component. */
export interface ImageUploaderProps {
  onFileSelect: (file: File, previewUrl: string) => void;
  disabled: boolean;
}

/** Props for the epsilon slider component. */
export interface EpsilonSliderProps {
  value: number;
  onChange: (value: number) => void;
  disabled: boolean;
}

/** Props for the result viewer component. */
export interface ImageComparisonProps {
  originalUrl: string;
  cloakedBase64: string;
  originalFilename: string;
}

/** Props for the metrics display component. */
export interface MetricsCardProps {
  metrics: Metrics;
  processingTimeMs: number;
}

/** Props for the error message component. */
export interface ErrorMessageProps {
  message: string;
  onDismiss: () => void;
}

/** Props for the loading spinner component. */
export interface LoadingSpinnerProps {
  message?: string;
}

/** Props for the backend status banner. */
export interface BackendStatusProps {
  status: BackendStatus;
  healthData: HealthResponse | null;
}
