/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          50:  '#eef7ff',
          100: '#d9ecff',
          200: '#bbdeff',
          300: '#8bcaff',
          400: '#54adff',
          500: '#2b8dff',
          600: '#1a6ef5',
          700: '#1459e1',
          800: '#1648b6',
          900: '#18408f',
          950: '#132856',
        },
        dark: {
          50:  '#f0f4ff',
          100: '#dde5f7',
          200: '#bfcef0',
          300: '#94abe4',
          400: '#6284d5',
          500: '#4163c5',
          600: '#324faa',
          700: '#293f8a',
          800: '#253772',
          900: '#242f60',
          950: '#181e3d',
        },
        surface: {
          DEFAULT: '#0d1117',
          card:    '#161b22',
          border:  '#21262d',
          hover:   '#1c2128',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      animation: {
        'pulse-slow':  'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in':     'fadeIn 0.4s ease-out',
        'slide-up':    'slideUp 0.5s ease-out',
        'glow':        'glow 2s ease-in-out infinite alternate',
        'spin-slow':   'spin 2s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%':   { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        glow: {
          '0%':   { boxShadow: '0 0 5px rgba(43,141,255,0.3)' },
          '100%': { boxShadow: '0 0 20px rgba(43,141,255,0.7), 0 0 40px rgba(43,141,255,0.3)' },
        },
      },
      backgroundImage: {
        'grid-pattern': `linear-gradient(rgba(43,141,255,0.05) 1px, transparent 1px),
                         linear-gradient(90deg, rgba(43,141,255,0.05) 1px, transparent 1px)`,
        'hero-gradient': 'linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%)',
        'card-gradient': 'linear-gradient(135deg, #161b22 0%, #1c2128 100%)',
      },
      backgroundSize: {
        'grid': '40px 40px',
      },
      boxShadow: {
        'cyber':      '0 0 20px rgba(43,141,255,0.15)',
        'cyber-lg':   '0 0 40px rgba(43,141,255,0.25)',
        'card':       '0 4px 24px rgba(0,0,0,0.4)',
        'card-hover': '0 8px 32px rgba(0,0,0,0.6), 0 0 20px rgba(43,141,255,0.1)',
      },
    },
  },
  plugins: [],
}
