/**
 * EpsilonSlider.tsx — FGSM perturbation strength control.
 *
 * Range:   0.01 → 0.10
 * Step:    0.01
 * Default: 0.02
 *
 * Shows a visual intensity bar and descriptive label for the selected level.
 */

import React from 'react';
import type { EpsilonSliderProps } from '../types/api';

const EPSILON_MIN = 0.01;
const EPSILON_MAX = 0.10;
const EPSILON_STEP = 0.01;

interface LevelInfo {
  label: string;
  description: string;
  color: string;
}

function getLevelInfo(value: number): LevelInfo {
  if (value <= 0.02) {
    return {
      label: 'Subtle',
      description: 'Imperceptible noise — recommended for everyday use',
      color: 'text-emerald-400',
    };
  }
  if (value <= 0.05) {
    return {
      label: 'Moderate',
      description: 'Stronger protection, virtually invisible to the human eye',
      color: 'text-amber-400',
    };
  }
  return {
    label: 'Strong',
    description: 'Maximum disruption — may introduce barely-visible texture',
    color: 'text-red-400',
  };
}

/** Maps epsilon value to a 0–100 percentage for the track fill */
function toPercent(value: number): number {
  return ((value - EPSILON_MIN) / (EPSILON_MAX - EPSILON_MIN)) * 100;
}

const EpsilonSlider: React.FC<EpsilonSliderProps> = ({ value, onChange, disabled }) => {
  const level = getLevelInfo(value);
  const percent = toPercent(value);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(parseFloat(e.target.value));
  };

  return (
    <div id="epsilon-slider-container" className="w-full space-y-3">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div>
          <label
            htmlFor="epsilon-slider"
            className="text-sm font-semibold text-gray-200"
          >
            Perturbation Strength (ε)
          </label>
          <p className="text-xs text-gray-500 mt-0.5">
            Controls the FGSM L-infinity perturbation budget
          </p>
        </div>
        <div className="text-right">
          <span
            id="epsilon-value-display"
            className="text-2xl font-bold font-mono text-cyber-400"
          >
            {value.toFixed(2)}
          </span>
          <p className={`text-xs font-semibold mt-0.5 ${level.color}`}>
            {level.label}
          </p>
        </div>
      </div>

      {/* Slider track + input */}
      <div className="relative pt-1">
        {/* Custom track background */}
        <div className="relative w-full h-2 rounded-full bg-surface-border overflow-hidden">
          <div
            className="absolute left-0 top-0 bottom-0 rounded-full transition-all duration-150"
            style={{
              width: `${percent}%`,
              background: 'linear-gradient(90deg, #1a6ef5, #2b8dff)',
            }}
          />
        </div>

        {/* Range input (overlays the custom track) */}
        <input
          id="epsilon-slider"
          type="range"
          min={EPSILON_MIN}
          max={EPSILON_MAX}
          step={EPSILON_STEP}
          value={value}
          onChange={handleChange}
          disabled={disabled}
          aria-label={`Epsilon value: ${value.toFixed(2)}`}
          aria-valuemin={EPSILON_MIN}
          aria-valuemax={EPSILON_MAX}
          aria-valuenow={value}
          className="absolute inset-0 w-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
          style={{ height: '8px', top: 0 }}
        />
      </div>

      {/* Min / Max labels */}
      <div className="flex justify-between text-xs font-mono text-gray-600">
        <span>ε = 0.01</span>
        <span>ε = 0.10</span>
      </div>

      {/* Level description */}
      <div
        className={`flex items-start gap-2 p-3 rounded-xl border transition-colors duration-300 ${
          value <= 0.02
            ? 'border-emerald-500/20 bg-emerald-950/20'
            : value <= 0.05
            ? 'border-amber-500/20 bg-amber-950/20'
            : 'border-red-500/20 bg-red-950/20'
        }`}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          className={`w-4 h-4 flex-shrink-0 mt-0.5 ${level.color}`}
        >
          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5" />
          <path
            d="M12 8v4M12 16h.01"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          />
        </svg>
        <p className="text-xs text-gray-400 leading-relaxed">{level.description}</p>
      </div>
    </div>
  );
};

export default EpsilonSlider;
