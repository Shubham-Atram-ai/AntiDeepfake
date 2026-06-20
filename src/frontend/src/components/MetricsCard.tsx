/**
 * MetricsCard.tsx — Displays SSIM, PSNR, and processing time from CloakResponse.
 *
 * psnr may be `null` (identical images → infinite PSNR) — rendered as "∞ dB".
 */

import React from 'react';
import type { MetricsCardProps } from '../types/api';

interface MetricItemProps {
  id: string;
  label: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
  quality?: 'good' | 'warn' | 'neutral';
}

const MetricItem: React.FC<MetricItemProps> = ({
  id,
  label,
  value,
  subtitle,
  icon,
  quality = 'neutral',
}) => {
  const qualityColors = {
    good:    'text-emerald-400 bg-emerald-950/40 border-emerald-500/20',
    warn:    'text-amber-400 bg-amber-950/40 border-amber-500/20',
    neutral: 'text-cyber-400 bg-cyber-950/40 border-cyber-500/20',
  };

  return (
    <div
      id={id}
      className={`flex flex-col gap-2 p-4 rounded-xl border transition-all duration-300 hover:shadow-card-hover ${qualityColors[quality]}`}
    >
      <div className="flex items-center gap-2">
        <div className="flex-shrink-0 opacity-80">{icon}</div>
        <span className="text-xs font-semibold uppercase tracking-wider opacity-70">
          {label}
        </span>
      </div>
      <div>
        <p className="text-2xl font-bold font-mono leading-none">{value}</p>
        <p className="text-xs opacity-60 mt-1 leading-relaxed">{subtitle}</p>
      </div>
    </div>
  );
};

// ── Icons ─────────────────────────────────────────────────────────────────────

const SsimIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5">
    <path d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2z" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const PsnrIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5">
    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const TimeIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5">
    <circle cx="12" cy="12" r="10" />
    <path d="M12 6v6l4 2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

// ── Main component ────────────────────────────────────────────────────────────

const MetricsCard: React.FC<MetricsCardProps> = ({ metrics, processingTimeMs }) => {
  const ssimDisplay = metrics.ssim.toFixed(4);
  const psnrDisplay =
    metrics.psnr === null ? '∞' : `${metrics.psnr.toFixed(2)} dB`;
  const timeDisplay =
    processingTimeMs >= 1000
      ? `${(processingTimeMs / 1000).toFixed(2)} s`
      : `${processingTimeMs.toFixed(0)} ms`;

  // Quality thresholds based on research norms
  const ssimQuality: MetricItemProps['quality'] =
    metrics.ssim >= 0.95 ? 'good' : metrics.ssim >= 0.85 ? 'warn' : 'warn';
  const psnrQuality: MetricItemProps['quality'] =
    metrics.psnr === null || metrics.psnr >= 40 ? 'good' : metrics.psnr >= 30 ? 'warn' : 'warn';

  return (
    <div
      id="metrics-card"
      className="w-full animate-slide-up"
    >
      <div className="flex items-center gap-3 mb-4">
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-surface-border to-transparent" />
        <h2 className="text-sm font-bold uppercase tracking-widest text-gray-400 flex items-center gap-2">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-cyber-500">
            <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Quality Metrics
        </h2>
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-surface-border to-transparent" />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <MetricItem
          id="metric-ssim"
          label="SSIM"
          value={ssimDisplay}
          subtitle="Structural Similarity Index. Closer to 1.0 = more imperceptible."
          icon={<SsimIcon />}
          quality={ssimQuality}
        />
        <MetricItem
          id="metric-psnr"
          label="PSNR"
          value={psnrDisplay}
          subtitle="Peak Signal-to-Noise Ratio. >40 dB = virtually identical to original."
          icon={<PsnrIcon />}
          quality={psnrQuality}
        />
        <MetricItem
          id="metric-time"
          label="Processing Time"
          value={timeDisplay}
          subtitle="Total server-side pipeline time: detect → perturb → encode."
          icon={<TimeIcon />}
          quality="neutral"
        />
      </div>
    </div>
  );
};

export default MetricsCard;
