/**
 * ImageComparison.tsx — Side-by-side (desktop) / stacked (mobile) display
 * of the original and adversarially cloaked images.
 *
 * Also provides the client-side download button for the cloaked image.
 */

import React, { useState } from 'react';
import { downloadBase64Image } from '../services/api';
import type { ImageComparisonProps } from '../types/api';

interface ImagePanelProps {
  id: string;
  title: string;
  badge: string;
  badgeColor: string;
  src: string;
  alt: string;
  children?: React.ReactNode;
}

const ImagePanel: React.FC<ImagePanelProps> = ({
  id,
  title,
  badge,
  badgeColor,
  src,
  alt,
  children,
}) => {
  const [imageLoaded, setImageLoaded] = useState(false);

  return (
    <div
      id={id}
      className="flex flex-col rounded-2xl border border-surface-border bg-surface-card overflow-hidden shadow-card transition-all duration-300 hover:shadow-card-hover"
    >
      {/* Panel header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-surface-border bg-surface-hover">
        <h3 className="text-sm font-semibold text-gray-200">{title}</h3>
        <span
          className={`px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wide ${badgeColor}`}
        >
          {badge}
        </span>
      </div>

      {/* Image */}
      <div className="relative flex items-center justify-center bg-surface-DEFAULT min-h-[220px] overflow-hidden">
        {/* Skeleton while loading */}
        {!imageLoaded && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface-card animate-pulse">
            <svg viewBox="0 0 24 24" fill="none" className="w-10 h-10 text-gray-700">
              <path
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 16M14 8h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
        )}
        <img
          src={src}
          alt={alt}
          onLoad={() => setImageLoaded(true)}
          className={`w-full h-auto max-h-96 object-contain transition-opacity duration-500 ${
            imageLoaded ? 'opacity-100' : 'opacity-0'
          }`}
        />
      </div>

      {/* Footer actions */}
      {children && (
        <div className="px-4 py-3 border-t border-surface-border">{children}</div>
      )}
    </div>
  );
};

// ── Main component ────────────────────────────────────────────────────────────

const ImageComparison: React.FC<ImageComparisonProps> = ({
  originalUrl,
  cloakedBase64,
  originalFilename,
}) => {
  const [downloading, setDownloading] = useState(false);

  const cloakedDataUri = `data:image/jpeg;base64,${cloakedBase64}`;

  const handleDownload = () => {
    setDownloading(true);
    try {
      const baseName = originalFilename.replace(/\.[^.]+$/, '') || 'image';
      downloadBase64Image(cloakedBase64, `${baseName}_cloaked.jpg`);
    } finally {
      setTimeout(() => setDownloading(false), 1_500);
    }
  };

  return (
    <div
      id="image-comparison"
      className="w-full space-y-4 animate-slide-up"
    >
      {/* Section heading */}
      <div className="flex items-center gap-3">
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-surface-border to-transparent" />
        <h2 className="text-sm font-bold uppercase tracking-widest text-gray-400 flex items-center gap-2">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-cyber-500">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
          </svg>
          Image Comparison
        </h2>
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-surface-border to-transparent" />
      </div>

      {/* Side-by-side grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Original image */}
        <ImagePanel
          id="original-image-panel"
          title="Original Image"
          badge="Original"
          badgeColor="text-gray-300 bg-gray-700/50"
          src={originalUrl}
          alt="Original uploaded image"
        />

        {/* Cloaked image */}
        <ImagePanel
          id="cloaked-image-panel"
          title="Protected Image"
          badge="Cloaked"
          badgeColor="text-cyber-300 bg-cyber-900/60"
          src={cloakedDataUri}
          alt="Adversarially cloaked image"
        >
          <button
            id="download-button"
            onClick={handleDownload}
            disabled={downloading}
            className="
              w-full flex items-center justify-center gap-2
              px-4 py-2.5 rounded-xl text-sm font-semibold
              bg-cyber-600 hover:bg-cyber-500 active:bg-cyber-700
              text-white transition-all duration-200
              disabled:opacity-60 disabled:cursor-not-allowed
              shadow-cyber hover:shadow-cyber-lg
            "
            aria-label="Download cloaked image as JPEG"
          >
            {downloading ? (
              <>
                <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span>Preparing download…</span>
              </>
            ) : (
              <>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span>Download Protected Image</span>
              </>
            )}
          </button>
        </ImagePanel>
      </div>

      {/* Comparison note */}
      <p className="text-center text-xs text-gray-600 font-mono">
        The cloaked image is visually identical to humans but confuses AI face-recognition systems.
      </p>
    </div>
  );
};

export default ImageComparison;
