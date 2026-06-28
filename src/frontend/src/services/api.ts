/**
 * api.ts — Axios service layer for the AntiDeepfake backend.
 *
 * Endpoint: POST /api/v1/cloak
 * Health:   GET  /health
 *
 * The base URL is read from the VITE_API_URL environment variable.
 * Falls back to http://localhost:8000 for local development.
 */

import axios, { AxiosError, AxiosInstance } from 'axios';
import type { CloakResponse, ErrorResponse, HealthResponse } from '../types/api';

// ---------------------------------------------------------------------------
// Axios instance
// ---------------------------------------------------------------------------

const BASE_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

const apiClient: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000, // 2-minute timeout for ML processing
  headers: {
    Accept: 'application/json',
  },
});

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Union type for the /api/v1/cloak response. */
export type CloakResult =
  | { ok: true; data: CloakResponse }
  | { ok: false; error: string; error_code?: string };

/** Union type for the /health response. */
export type HealthResult =
  | { ok: true; data: HealthResponse }
  | { ok: false };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Extracts a user-friendly error message from an Axios error, preferring
 * the structured `ErrorResponse.error` field when available.
 */
function extractErrorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    const data = err.response?.data as ErrorResponse | undefined;
    if (data?.error) {
      return data.error;
    }
    if (err.code === 'ECONNABORTED') {
      return 'Request timed out. The image may be too large or the server is under load.';
    }
    if (!err.response) {
      return 'Cannot connect to the backend server. Please ensure it is running.';
    }
    return `Server error (${err.response.status}): ${err.message}`;
  }
  if (err instanceof Error) {
    return err.message;
  }
  return 'An unexpected error occurred.';
}

/**
 * Extracts the machine-readable error code from an Axios error response,
 * if one is available.
 */
function extractErrorCode(err: unknown): string | undefined {
  if (err instanceof AxiosError) {
    const data = err.response?.data as ErrorResponse | undefined;
    return data?.error_code;
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

/**
 * Calls GET /health and returns the backend health status.
 *
 * Handles 503 (degraded) as a valid response rather than an error,
 * because the backend returns structured JSON even in degraded state.
 */
export async function checkHealth(): Promise<HealthResult> {
  try {
    const response = await apiClient.get<HealthResponse>('/health', {
      // Don't throw for 503 — we want to read the body
      validateStatus: (status) => status === 200 || status === 503,
    });
    return { ok: true, data: response.data };
  } catch (_err) {
    return { ok: false };
  }
}

/**
 * Calls POST /api/v1/cloak with the provided image file and epsilon value.
 *
 * Sends the request as multipart/form-data with:
 *   - file: the uploaded image
 *   - epsilon: the PGD perturbation budget
 *
 * @param file    - The image File to cloak.
 * @param epsilon - PGD perturbation strength in (0.0, 1.0].
 * @returns       A CloakResult discriminated union (ok/error).
 */
export async function cloakImage(
  file: File,
  epsilon: number
): Promise<CloakResult> {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('epsilon', epsilon.toString());

    const response = await apiClient.post<CloakResponse>(
      '/api/v1/cloak',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return { ok: true, data: response.data };
  } catch (err: unknown) {
    return {
      ok: false,
      error: extractErrorMessage(err),
      error_code: extractErrorCode(err),
    };
  }
}

/**
 * Converts a Base64-encoded JPEG string returned by the backend into a
 * browser-downloadable object URL, then triggers a download.
 *
 * Fully client-side — no backend download endpoint is used.
 *
 * @param base64   - The raw base64 string from `CloakResponse.cloaked_image_base64`.
 * @param filename - Suggested filename for the downloaded file.
 */
export function downloadBase64Image(
  base64: string,
  filename: string = 'cloaked_image.jpg'
): void {
  // Decode base64 → binary → Blob
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: 'image/jpeg' });

  // Create a temporary anchor and trigger download
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);

  // Release the object URL after a short delay
  setTimeout(() => URL.revokeObjectURL(url), 1_000);
}
