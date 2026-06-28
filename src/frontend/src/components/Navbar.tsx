/**
 * Navbar.tsx — Top navigation bar with branding and GitHub link.
 */

import React from 'react';

const Navbar: React.FC = () => {
  return (
    <nav
      id="navbar"
      className="fixed top-0 left-0 right-0 z-50 border-b border-surface-border"
      style={{
        background: 'rgba(13, 17, 23, 0.85)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
      }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo + Brand */}
          <div className="flex items-center gap-3">
            {/* Shield icon */}
            <div className="relative flex items-center justify-center w-9 h-9">
              <svg
                viewBox="0 0 36 36"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="w-9 h-9"
              >
                <path
                  d="M18 3L4 9v10c0 8.28 5.94 16.02 14 18 8.06-1.98 14-9.72 14-18V9L18 3z"
                  fill="url(#shieldGrad)"
                  stroke="rgba(43,141,255,0.6)"
                  strokeWidth="1"
                />
                <path
                  d="M13 18l3.5 3.5L23 14"
                  stroke="#fff"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <defs>
                  <linearGradient id="shieldGrad" x1="4" y1="3" x2="32" y2="39" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#1a6ef5" />
                    <stop offset="1" stopColor="#2b8dff" stopOpacity="0.6" />
                  </linearGradient>
                </defs>
              </svg>
            </div>

            <div>
              <span className="text-white font-bold text-lg tracking-tight">
                Anti<span className="text-cyber-500">Deepfake</span>
              </span>
              <p className="text-xs text-gray-500 font-mono hidden sm:block">
                v1.0 — PGD Adversarial Cloaking
              </p>
            </div>
          </div>

          {/* Right side links */}
          <div className="flex items-center gap-4">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              id="navbar-github-link"
              className="flex items-center gap-2 text-sm text-gray-400 hover:text-cyber-400 transition-colors duration-200"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
              </svg>
              <span className="hidden sm:inline">GitHub</span>
            </a>

            <div className="h-4 w-px bg-surface-border" />

            <a
              href="/docs"
              target="_blank"
              rel="noopener noreferrer"
              id="navbar-api-docs-link"
              className="text-sm text-gray-400 hover:text-cyber-400 transition-colors duration-200 hidden sm:block"
            >
              API Docs
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
