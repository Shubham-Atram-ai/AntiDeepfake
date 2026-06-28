/**
 * LoadingSpinner.tsx — Animated loading indicator shown during API calls.
 */

import React from 'react';
import type { LoadingSpinnerProps } from '../types/api';

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = 'Generating protected image…',
}) => {
  return (
    <div
      id="loading-spinner"
      className="flex flex-col items-center justify-center gap-5 py-12 animate-fade-in"
      role="status"
      aria-live="polite"
      aria-label={message}
    >
      {/* Layered spinning rings */}
      <div className="relative w-16 h-16">
        {/* Outer ring */}
        <div className="absolute inset-0 rounded-full border-2 border-cyber-500/20" />
        <div
          className="absolute inset-0 rounded-full border-2 border-t-cyber-500 border-r-transparent border-b-transparent border-l-transparent animate-spin"
          style={{ animationDuration: '1s' }}
        />
        {/* Middle ring */}
        <div className="absolute inset-2 rounded-full border-2 border-cyber-400/20" />
        <div
          className="absolute inset-2 rounded-full border-2 border-b-cyber-400 border-t-transparent border-r-transparent border-l-transparent animate-spin"
          style={{ animationDuration: '0.7s', animationDirection: 'reverse' }}
        />
        {/* Center dot */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-2 h-2 rounded-full bg-cyber-500 animate-pulse" />
        </div>
      </div>

      {/* Message */}
      <div className="text-center">
        <p className="text-cyber-400 font-semibold text-sm">{message}</p>
        <p className="text-gray-500 text-xs mt-1 font-mono">
          PGD adversarial perturbation in progress…
        </p>
      </div>

      {/* Progress bar shimmer */}
      <div className="w-48 h-1 bg-surface-border rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-transparent via-cyber-500 to-transparent rounded-full"
          style={{
            animation: 'shimmer 1.5s ease-in-out infinite',
            width: '60%',
          }}
        />
      </div>
    </div>
  );
};

export default LoadingSpinner;
