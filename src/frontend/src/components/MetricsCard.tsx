/**
 * MetricsCard.tsx — Displays SSIM, PSNR, Cosine Similarity, Detection Probability,
 * and processing time from CloakResponse.
 *
 * Features:
 *  - Animated radial gauge for AI Detection Probability
 *  - Colour-coded Protection Status banner (PROTECTED / PARTIAL / VULNERABLE)
 *  - Cosine similarity with embedding divergence context
 *  - SSIM / PSNR / Processing Time metric tiles
 */

import React from 'react';
import type { MetricsCardProps } from '../types/api';

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Returns label + colour theme based on detection probability */
function getProtectionLevel(prob: number): {
  label: string;
  sublabel: string;
  color: string;
  trackColor: string;
  badgeBg: string;
  badgeText: string;
  glow: string;
} {
  if (prob <= 25) {
    return {
      label: 'PROTECTED',
      sublabel: 'AI is very unlikely to identify this face',
      color: '#10b981',      // emerald-500
      trackColor: '#064e3b', // emerald-950
      badgeBg: 'bg-emerald-950/60 border-emerald-500/40',
      badgeText: 'text-emerald-400',
      glow: '0 0 20px rgba(16,185,129,0.35)',
    };
  }
  if (prob <= 60) {
    return {
      label: 'PARTIAL',
      sublabel: 'Some risk — consider increasing ε',
      color: '#f59e0b',      // amber-500
      trackColor: '#451a03', // amber-950
      badgeBg: 'bg-amber-950/60 border-amber-500/40',
      badgeText: 'text-amber-400',
      glow: '0 0 20px rgba(245,158,11,0.35)',
    };
  }
  return {
    label: 'VULNERABLE',
    sublabel: 'AI can likely still identify this face',
    color: '#ef4444',       // red-500
    trackColor: '#450a0a',  // red-950
    badgeBg: 'bg-red-950/60 border-red-500/40',
    badgeText: 'text-red-400',
    glow: '0 0 20px rgba(239,68,68,0.35)',
  };
}

// ── Radial Detection Gauge ────────────────────────────────────────────────────

interface GaugeProps {
  probability: number; // 0–100
}

const DetectionGauge: React.FC<GaugeProps> = ({ probability }) => {
  const protection = getProtectionLevel(probability);
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  // Gauge arc covers 270° (¾ of circle), starting from bottom-left
  const arcLen = circumference * 0.75;
  const filled = (probability / 100) * arcLen;
  const dashOffset = arcLen - filled;

  return (
    <div
      id="detection-gauge"
      className="flex flex-col items-center gap-3"
    >
      {/* SVG gauge */}
      <div className="relative" style={{ width: 140, height: 140 }}>
        <svg
          width="140"
          height="140"
          viewBox="0 0 140 140"
          style={{ transform: 'rotate(135deg)' }}
        >
          {/* Track */}
          <circle
            cx="70" cy="70" r={radius}
            fill="none"
            stroke={protection.trackColor}
            strokeWidth="10"
            strokeDasharray={`${arcLen} ${circumference}`}
            strokeLinecap="round"
            strokeDashoffset={0}
          />
          {/* Fill */}
          <circle
            cx="70" cy="70" r={radius}
            fill="none"
            stroke={protection.color}
            strokeWidth="10"
            strokeDasharray={`${arcLen} ${circumference}`}
            strokeLinecap="round"
            strokeDashoffset={dashOffset}
            style={{
              transition: 'stroke-dashoffset 1s ease-out',
              filter: `drop-shadow(${protection.glow})`,
            }}
          />
        </svg>
        {/* Centre label */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="text-3xl font-black font-mono leading-none"
            style={{ color: protection.color }}
          >
            {probability.toFixed(0)}%
          </span>
          <span className="text-[10px] text-gray-400 font-semibold tracking-widest uppercase mt-0.5">
            detection
          </span>
        </div>
      </div>

      {/* Badge */}
      <div
        className={`flex items-center gap-2 px-4 py-1.5 rounded-full border text-xs font-bold uppercase tracking-widest ${protection.badgeBg} ${protection.badgeText}`}
        style={{ boxShadow: protection.glow }}
      >
        <span
          className="w-2 h-2 rounded-full animate-pulse"
          style={{ backgroundColor: protection.color }}
        />
        {protection.label}
      </div>
      <p className="text-[11px] text-gray-500 text-center leading-tight max-w-[160px]">
        {protection.sublabel}
      </p>
    </div>
  );
};

// ── Small Metric Tile ─────────────────────────────────────────────────────────

interface MetricItemProps {
  id: string;
  label: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
  quality?: 'good' | 'warn' | 'bad' | 'neutral';
}

const MetricItem: React.FC<MetricItemProps> = ({
  id, label, value, subtitle, icon, quality = 'neutral',
}) => {
  const colors = {
    good:    'text-emerald-400 bg-emerald-950/40 border-emerald-500/20',
    warn:    'text-amber-400  bg-amber-950/40  border-amber-500/20',
    bad:     'text-red-400    bg-red-950/40    border-red-500/20',
    neutral: 'text-sky-400    bg-sky-950/40    border-sky-500/20',
  };
  return (
    <div
      id={id}
      className={`flex flex-col gap-2 p-4 rounded-xl border transition-all duration-300 hover:brightness-110 ${colors[quality]}`}
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

// ── Cosine Bar ────────────────────────────────────────────────────────────────

interface CosineBarProps {
  value: number; // -1 to 1
}

const CosineBar: React.FC<CosineBarProps> = ({ value }) => {
  // Map [-1, 1] → [0%, 100%] for bar width
  const pct = ((value + 1) / 2) * 100;
  // Closer to 0 (or negative) = better protection = green; closer to 1 = red
  const barColor =
    value <= 0.3 ? '#10b981' : value <= 0.6 ? '#f59e0b' : '#ef4444';

  return (
    <div id="cosine-bar-container" className="flex flex-col gap-2 p-4 rounded-xl border border-surface-border bg-surface-card/40">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <CosineIcon />
          <span className="text-xs font-semibold uppercase tracking-wider text-gray-400">
            Cosine Similarity
          </span>
        </div>
        <span
          className="text-lg font-black font-mono"
          style={{ color: barColor }}
        >
          {value.toFixed(4)}
        </span>
      </div>
      {/* Bar track */}
      <div className="w-full h-2 rounded-full bg-gray-800 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-1000"
          style={{
            width: `${pct}%`,
            backgroundColor: barColor,
            boxShadow: `0 0 8px ${barColor}80`,
          }}
        />
      </div>
      {/* Scale labels */}
      <div className="flex justify-between text-[10px] text-gray-600 font-mono">
        <span>-1.0 (max protection)</span>
        <span>0.0</span>
        <span>+1.0 (no protection)</span>
      </div>
      <p className="text-xs text-gray-500 leading-relaxed">
        FaceNet embedding divergence between original and cloaked face.
        Values ≤ 0.5 indicate the AI identity has been disrupted.
      </p>
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

const CosineIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5">
    <path d="M12 2a10 10 0 100 20 10 10 0 000-20z" strokeLinecap="round"/>
    <path d="M8 12h8M12 8l4 4-4 4" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

// ── Main Component ────────────────────────────────────────────────────────────

const MetricsCard: React.FC<MetricsCardProps> = ({ metrics, processingTimeMs }) => {
  const ssimDisplay = metrics.ssim.toFixed(4);
  const psnrDisplay =
    metrics.psnr === null ? '∞' : `${metrics.psnr.toFixed(2)} dB`;
  const timeDisplay =
    processingTimeMs >= 1000
      ? `${(processingTimeMs / 1000).toFixed(2)} s`
      : `${processingTimeMs.toFixed(0)} ms`;

  const ssimQuality: MetricItemProps['quality'] =
    metrics.ssim >= 0.95 ? 'good' : metrics.ssim >= 0.85 ? 'warn' : 'bad';
  const psnrQuality: MetricItemProps['quality'] =
    metrics.psnr === null || metrics.psnr >= 40 ? 'good' : metrics.psnr >= 30 ? 'warn' : 'bad';

  return (
    <div id="metrics-card" className="w-full animate-slide-up space-y-5">

      {/* ── Section header ─────────────────────────────────────────────────── */}
      <div className="flex items-center gap-3">
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-surface-border to-transparent" />
        <h2 className="text-sm font-bold uppercase tracking-widest text-gray-400 flex items-center gap-2">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-cyber-500">
            <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Protection Analysis
        </h2>
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-surface-border to-transparent" />
      </div>

      {/* ── Top row: Detection gauge + Cosine bar ──────────────────────────── */}
      <div className="flex flex-col sm:flex-row gap-4 items-start">
        {/* Gauge */}
        <div className="flex-shrink-0 w-full sm:w-auto flex justify-center sm:justify-start p-4 rounded-2xl border border-surface-border bg-surface-card/30 backdrop-blur-sm">
          <DetectionGauge probability={metrics.detection_probability} />
        </div>

        {/* Cosine bar — fills remaining space */}
        <div className="flex-1 w-full">
          <CosineBar value={metrics.cosine_similarity} />

          {/* Explainer box */}
          <div className="mt-3 px-4 py-3 rounded-xl bg-surface-card/30 border border-surface-border text-xs text-gray-500 leading-relaxed">
            <span className="text-gray-300 font-semibold">How to read this:</span>{' '}
            The <span className="text-sky-400 font-mono">cosine similarity</span> measures how
            close the AI's face-embedding of the cloaked image is to the original. A{' '}
            <span className="text-emerald-400">low value (≤ 0.3)</span> means the AI sees a
            completely different identity — your face is protected. A{' '}
            <span className="text-red-400">high value (≥ 0.7)</span> means the AI can still
            match you. The <span className="text-sky-400 font-mono">detection %</span> directly
            reflects this divergence.
          </div>
        </div>
      </div>

      {/* ── Bottom row: SSIM / PSNR / Time tiles ───────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <MetricItem
          id="metric-ssim"
          label="SSIM"
          value={ssimDisplay}
          subtitle="Structural Similarity Index. Closer to 1.0 = more imperceptible to humans."
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
