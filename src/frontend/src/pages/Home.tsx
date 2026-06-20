/**
 * Home.tsx — Main page of AntiDeepfake.
 *
 * Manages all application state via React hooks:
 *   - File selection + preview URL
 *   - Epsilon value
 *   - Loading state
 *   - API result (CloakResponse)
 *   - Error message
 *
 * Calls the API service and renders the results.
 */

import React, { useCallback, useState } from 'react';
import { cloakImage } from '../services/api';
import { useHealthCheck } from '../hooks/useHealthCheck';
import type { AppState, CloakResponse } from '../types/api';

import BackendStatus from '../components/BackendStatus';
import ImageUploader from '../components/ImageUploader';
import EpsilonSlider from '../components/EpsilonSlider';
import ImageComparison from '../components/ImageComparison';
import MetricsCard from '../components/MetricsCard';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

const INITIAL_STATE: AppState = {
  file: null,
  previewUrl: null,
  epsilon: 0.02,
  loading: false,
  result: null,
  error: null,
};

const Home: React.FC = () => {
  const [state, setState] = useState<AppState>(INITIAL_STATE);
  const { status: backendStatus, healthData } = useHealthCheck(30_000);

  // Helper to update a single field
  const update = useCallback(
    <K extends keyof AppState>(key: K, value: AppState[K]) =>
      setState((prev) => ({ ...prev, [key]: value })),
    []
  );

  const handleFileSelect = useCallback(
    (file: File, previewUrl: string) => {
      // Revoke previous object URL to avoid memory leaks
      if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
      setState((prev) => ({
        ...prev,
        file,
        previewUrl,
        result: null,
        error: null,
      }));
    },
    [state.previewUrl]
  );

  const handleEpsilonChange = useCallback(
    (epsilon: number) => update('epsilon', epsilon),
    [update]
  );

  const handleDismissError = useCallback(
    () => update('error', null),
    [update]
  );

  const handleSubmit = useCallback(async () => {
    if (!state.file) return;

    setState((prev) => ({ ...prev, loading: true, error: null, result: null }));

    const res = await cloakImage(state.file, state.epsilon);

    if (res.ok) {
      setState((prev) => ({
        ...prev,
        loading: false,
        result: res.data as CloakResponse,
      }));
    } else {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: res.error,
      }));
    }
  }, [state.file, state.epsilon]);

  const isBackendUnavailable =
    backendStatus === 'offline' || backendStatus === 'degraded';
  const canSubmit =
    state.file !== null && !state.loading && !isBackendUnavailable;

  return (
    <main id="home-page" className="min-h-screen bg-surface-DEFAULT">
      {/* ── Hero section ────────────────────────────────────────────────────── */}
      <section
        id="hero-section"
        className="relative overflow-hidden pt-24 pb-16 px-4 sm:px-6 lg:px-8"
        style={{
          background:
            'radial-gradient(ellipse 80% 50% at 50% -10%, rgba(43,141,255,0.12) 0%, transparent 70%)',
        }}
      >
        {/* Grid overlay */}
        <div
          className="absolute inset-0 pointer-events-none opacity-30"
          style={{
            backgroundImage:
              'linear-gradient(rgba(43,141,255,0.07) 1px, transparent 1px), linear-gradient(90deg, rgba(43,141,255,0.07) 1px, transparent 1px)',
            backgroundSize: '40px 40px',
          }}
        />

        <div className="relative max-w-3xl mx-auto text-center">
          {/* Tag */}
          <div
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-cyber-700/40 bg-cyber-950/40 text-xs font-mono text-cyber-400 mb-6 animate-fade-in"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-cyber-400 animate-pulse" />
            FGSM Adversarial Cloaking · MTCNN Face Detection
          </div>

          {/* Headline */}
          <h1
            id="hero-title"
            className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-white leading-tight tracking-tight mb-4 animate-slide-up"
          >
            Protect Images Against{' '}
            <span
              className="text-transparent bg-clip-text"
              style={{
                backgroundImage: 'linear-gradient(135deg, #2b8dff, #54adff)',
              }}
            >
              Deepfakes
            </span>
          </h1>

          <p
            className="text-lg text-gray-400 max-w-2xl mx-auto leading-relaxed mb-8 animate-slide-up"
            style={{ animationDelay: '0.1s' }}
          >
            Upload a facial image to apply an imperceptible FGSM adversarial
            perturbation that disrupts AI face-recognition systems — while
            remaining completely natural to the human eye.
          </p>

          {/* Backend status badge */}
          <div className="flex justify-center animate-fade-in">
            <BackendStatus status={backendStatus} healthData={healthData} />
          </div>
        </div>
      </section>

      {/* ── Offline warning banner ───────────────────────────────────────────── */}
      {isBackendUnavailable && (
        <div
          id="offline-warning-banner"
          className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 -mt-4 mb-6"
        >
          <div className="flex items-start gap-3 p-4 rounded-xl border border-amber-500/30 bg-amber-950/30 animate-fade-in">
            <svg viewBox="0 0 24 24" fill="none" className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5">
              <path
                d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <div>
              <p className="text-sm font-semibold text-amber-300">
                {backendStatus === 'degraded' ? 'Models are still loading' : 'Backend unavailable'}
              </p>
              <p className="text-sm text-amber-400/70 mt-0.5">
                {backendStatus === 'degraded'
                  ? 'The server is running but the ML models are initializing. Please wait a moment before processing.'
                  : 'Start the FastAPI backend with: uvicorn src.backend.main:app --reload'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── Main workflow card ───────────────────────────────────────────────── */}
      <section
        id="upload-section"
        className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 pb-8"
      >
        <div
          className="rounded-2xl border border-surface-border shadow-card overflow-hidden"
          style={{
            background: 'linear-gradient(135deg, #161b22 0%, #1c2128 100%)',
          }}
        >
          {/* Card header */}
          <div className="px-6 py-5 border-b border-surface-border">
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5 text-cyber-500">
                <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Upload &amp; Configure
            </h2>
            <p className="text-sm text-gray-500 mt-0.5">
              Choose an image and set the perturbation strength
            </p>
          </div>

          <div className="p-6 space-y-6">
            {/* Uploader */}
            <ImageUploader
              onFileSelect={handleFileSelect}
              disabled={state.loading}
            />

            {/* Separator */}
            <div className="h-px bg-surface-border" />

            {/* Epsilon slider */}
            <EpsilonSlider
              value={state.epsilon}
              onChange={handleEpsilonChange}
              disabled={state.loading}
            />

            {/* Error display */}
            {state.error && (
              <ErrorMessage
                message={state.error}
                onDismiss={handleDismissError}
              />
            )}

            {/* Loading state */}
            {state.loading && <LoadingSpinner />}

            {/* Submit button */}
            {!state.loading && (
              <button
                id="protect-image-button"
                onClick={() => void handleSubmit()}
                disabled={!canSubmit}
                className="
                  w-full flex items-center justify-center gap-3
                  px-6 py-3.5 rounded-xl text-base font-bold
                  transition-all duration-300 select-none
                  disabled:opacity-40 disabled:cursor-not-allowed
                  enabled:hover:scale-[1.01] enabled:active:scale-[0.99]
                "
                style={
                  canSubmit
                    ? {
                        background: 'linear-gradient(135deg, #1a6ef5, #2b8dff)',
                        boxShadow: '0 0 20px rgba(43,141,255,0.3)',
                        color: '#fff',
                      }
                    : {
                        background: 'rgba(43,141,255,0.1)',
                        color: '#6b7280',
                      }
                }
                aria-label="Apply adversarial cloaking to selected image"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5">
                  <path d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                Protect Image
              </button>
            )}

            {/* No-file hint */}
            {!state.file && !state.loading && (
              <p className="text-center text-xs text-gray-600">
                Select an image above to enable processing
              </p>
            )}
          </div>
        </div>
      </section>

      {/* ── Results section ──────────────────────────────────────────────────── */}
      {state.result && state.previewUrl && (
        <section
          id="results-section"
          className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 pb-16 space-y-8 animate-slide-up"
        >
          {/* Image comparison */}
          <ImageComparison
            originalUrl={state.previewUrl}
            cloakedBase64={state.result.cloaked_image_base64}
            originalFilename={state.file?.name ?? 'image'}
          />

          {/* Metrics */}
          <MetricsCard
            metrics={state.result.metrics}
            processingTimeMs={state.result.processing_time_ms}
          />

          {/* Process another image */}
          <div className="text-center pt-2">
            <button
              id="reset-button"
              onClick={() => {
                if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
                setState(INITIAL_STATE);
              }}
              className="text-sm text-gray-500 hover:text-cyber-400 transition-colors duration-200 underline underline-offset-4"
            >
              Process another image
            </button>
          </div>
        </section>
      )}

      {/* ── Footer ──────────────────────────────────────────────────────────── */}
      <footer
        id="footer"
        className="border-t border-surface-border py-8 px-4 text-center"
      >
        <p className="text-xs text-gray-600 font-mono">
          AntiDeepfake v1.0 · FGSM Adversarial Cloaking · MTCNN Face Detection
        </p>
        <p className="text-xs text-gray-700 mt-1">
          For research and privacy protection purposes only.
        </p>
      </footer>
    </main>
  );
};

export default Home;
