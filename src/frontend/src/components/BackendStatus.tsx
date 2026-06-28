/**
 * BackendStatus.tsx — Banner that shows the backend connection state.
 *
 * Possible states:
 *   checking — initial health check in-progress
 *   online   — all models loaded, API healthy
 *   degraded — server responding but models not fully loaded
 *   offline  — server unreachable
 */

import React from 'react';
import type { BackendStatusProps } from '../types/api';

const BackendStatus: React.FC<BackendStatusProps> = ({ status, healthData }) => {
  // Don't render the banner while the initial check is running
  if (status === 'checking') return null;

  const configs = {
    online: {
      bg: 'bg-emerald-950/60 border-emerald-500/30',
      dot: 'bg-emerald-400',
      dotAnimation: 'animate-pulse',
      textColor: 'text-emerald-400',
      label: 'Backend Online',
      description: healthData
        ? `API v${healthData.version} · RetinaFace ${healthData.face_detector_loaded ? '✓' : '✗'} · PGD Engine ${healthData.pgd_engine_loaded ? '✓' : '✗'}`
        : 'All systems operational',
    },
    degraded: {
      bg: 'bg-amber-950/60 border-amber-500/30',
      dot: 'bg-amber-400',
      dotAnimation: 'animate-pulse-slow',
      textColor: 'text-amber-400',
      label: 'Models Loading',
      description: 'The backend is running but ML models are still initializing. Processing will be available shortly.',
    },
    offline: {
      bg: 'bg-red-950/60 border-red-500/30',
      dot: 'bg-red-400',
      dotAnimation: 'animate-pulse',
      textColor: 'text-red-400',
      label: 'Backend Offline',
      description: 'Cannot connect to the AntiDeepfake API. Please start the backend server.',
    },
  };

  const cfg = configs[status];

  return (
    <div
      id={`backend-status-${status}`}
      className={`flex items-center gap-3 px-4 py-2.5 rounded-xl border ${cfg.bg} backdrop-blur-sm animate-fade-in`}
    >
      {/* Pulsing status dot */}
      <span className="relative flex h-2.5 w-2.5 flex-shrink-0">
        <span
          className={`animate-ping absolute inline-flex h-full w-full rounded-full ${cfg.dot} opacity-60`}
        />
        <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${cfg.dot}`} />
      </span>

      {/* Text */}
      <div className="min-w-0">
        <span className={`text-sm font-semibold ${cfg.textColor}`}>
          {cfg.label}
        </span>
        <span className="text-gray-400 text-xs ml-2 truncate">
          {cfg.description}
        </span>
      </div>
    </div>
  );
};

export default BackendStatus;
