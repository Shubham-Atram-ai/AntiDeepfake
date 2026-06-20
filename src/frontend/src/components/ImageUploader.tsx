/**
 * ImageUploader.tsx — Drag-and-drop / click-to-upload image selector.
 *
 * Accepts: JPEG, PNG, BMP, WebP
 * Validates: file extension + MIME type before calling onFileSelect
 * Shows: file name, size, and thumbnail after selection
 */

import React, { useCallback, useRef, useState } from 'react';
import type { ImageUploaderProps } from '../types/api';

/** Accepted MIME types matching the backend's validate_content_type() */
const ACCEPTED_MIME_TYPES = new Set([
  'image/jpeg',
  'image/jpg',
  'image/png',
  'image/bmp',
  'image/webp',
]);

/** Accepted file extensions (fallback when MIME is empty/octet-stream) */
const ACCEPTED_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.webp']);

function validateFile(file: File): string | null {
  const ext = '.' + (file.name.split('.').pop() ?? '').toLowerCase();
  const mimeOk = ACCEPTED_MIME_TYPES.has(file.type);
  const extOk = ACCEPTED_EXTENSIONS.has(ext);

  if (!mimeOk && !extOk) {
    return `Unsupported file type "${file.type || ext}". Please upload a JPEG, PNG, BMP, or WebP image.`;
  }
  if (file.size === 0) {
    return 'The selected file is empty. Please choose a valid image.';
  }
  if (file.size > 20 * 1024 * 1024) {
    return 'File size exceeds 20 MB. Please upload a smaller image.';
  }
  return null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({ onFileSelect, disabled }) => {
  const [dragging, setDragging] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      setValidationError(null);
      const error = validateFile(file);
      if (error) {
        setValidationError(error);
        return;
      }
      setSelectedFile(file);
      const previewUrl = URL.createObjectURL(file);
      onFileSelect(file, previewUrl);
    },
    [onFileSelect]
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    // Reset input so the same file can be re-selected
    e.target.value = '';
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!disabled) setDragging(true);
  };

  const handleDragLeave = () => setDragging(false);

  const handleClick = () => {
    if (!disabled) inputRef.current?.click();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') handleClick();
  };

  const borderClass = dragging
    ? 'border-cyber-500 bg-cyber-950/30 shadow-cyber'
    : selectedFile
    ? 'border-cyber-700/60 bg-surface-card'
    : 'border-surface-border bg-surface-card hover:border-cyber-600/50 hover:bg-surface-hover';

  return (
    <div id="image-uploader" className="w-full">
      {/* Hidden file input */}
      <input
        ref={inputRef}
        id="file-input"
        type="file"
        accept=".jpg,.jpeg,.png,.bmp,.webp,image/jpeg,image/png,image/bmp,image/webp"
        onChange={handleInputChange}
        className="hidden"
        aria-label="Upload image"
        disabled={disabled}
      />

      {/* Drop zone */}
      <div
        id="drop-zone"
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Drag and drop an image here, or click to browse"
        aria-disabled={disabled}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          relative flex flex-col items-center justify-center gap-4
          border-2 border-dashed rounded-2xl p-8 cursor-pointer
          transition-all duration-300 select-none
          ${borderClass}
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        {selectedFile ? (
          /* Selected state */
          <div className="flex flex-col items-center gap-3 w-full">
            <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-cyber-900/40 border border-cyber-700/30">
              <svg viewBox="0 0 24 24" fill="none" className="w-7 h-7 text-cyber-400">
                <path
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 16M14 8h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            <div className="text-center">
              <p className="text-sm font-semibold text-white truncate max-w-xs">
                {selectedFile.name}
              </p>
              <p className="text-xs text-gray-500 mt-0.5">
                {formatBytes(selectedFile.size)} · {selectedFile.type || 'image'}
              </p>
            </div>
            <span className="text-xs text-cyber-500 font-medium">
              Click to change image
            </span>
          </div>
        ) : (
          /* Empty state */
          <>
            <div
              className={`flex items-center justify-center w-16 h-16 rounded-2xl border border-dashed transition-colors duration-300 ${
                dragging
                  ? 'border-cyber-500 bg-cyber-900/30'
                  : 'border-surface-border bg-surface-hover'
              }`}
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                className={`w-8 h-8 transition-colors duration-300 ${
                  dragging ? 'text-cyber-400' : 'text-gray-500'
                }`}
              >
                <path
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>

            <div className="text-center">
              <p className="text-base font-semibold text-gray-200">
                {dragging ? 'Release to upload' : 'Drop your image here'}
              </p>
              <p className="text-sm text-gray-500 mt-1">
                or{' '}
                <span className="text-cyber-400 font-medium">click to browse</span>
              </p>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-2">
              {['JPEG', 'PNG', 'BMP', 'WebP'].map((fmt) => (
                <span
                  key={fmt}
                  className="px-2.5 py-0.5 rounded-full text-xs font-mono font-medium text-gray-400 bg-surface-hover border border-surface-border"
                >
                  {fmt}
                </span>
              ))}
              <span className="text-xs text-gray-600">· up to 20 MB</span>
            </div>
          </>
        )}
      </div>

      {/* Validation error */}
      {validationError && (
        <div
          id="upload-validation-error"
          role="alert"
          className="mt-3 flex items-start gap-2 text-sm text-red-400 animate-fade-in"
        >
          <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4 flex-shrink-0 mt-0.5">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5" />
            <path d="M12 8v5M12 16h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          <span>{validationError}</span>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
