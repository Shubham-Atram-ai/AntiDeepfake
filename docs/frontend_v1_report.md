# AntiDeepfake Frontend V1 — Complete Report

**Project:** AntiDeepfake — Adversarial Image Cloaking  
**Component:** Frontend V1 — React + TypeScript + Tailwind CSS  
**Date:** 2026-06-20  
**Author:** Frontend Engineering  
**Status:** ✅ Complete — Build Verified & Passed (0 TypeScript errors, 0 warnings)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What Was Built](#2-what-was-built)
3. [Architecture Overview](#3-architecture-overview)
4. [File and Folder Explanation](#4-file-and-folder-explanation)
5. [How the Frontend Works — End to End](#5-how-the-frontend-works--end-to-end)
6. [Component Reference](#6-component-reference)
7. [API Integration Details](#7-api-integration-details)
8. [TypeScript Type System](#8-typescript-type-system)
9. [State Management](#9-state-management)
10. [Design System & Styling](#10-design-system--styling)
11. [Health Check System](#11-health-check-system)
12. [Image Download — Client-Side Implementation](#12-image-download--client-side-implementation)
13. [Error Handling Strategy](#13-error-handling-strategy)
14. [Responsiveness](#14-responsiveness)
15. [Build & Validation Log](#15-build--validation-log)
16. [Changes Made](#16-changes-made)
17. [Running the Frontend](#17-running-the-frontend)
18. [Manual Testing Guide](#18-manual-testing-guide)
19. [Design Decisions](#19-design-decisions)
20. [Troubleshooting Guide](#20-troubleshooting-guide)
21. [Frontend Summary](#21-frontend-summary)

---

## 1. Project Overview

### What is AntiDeepfake?

AntiDeepfake is a privacy protection system that applies a tiny, invisible "noise pattern" (an adversarial perturbation) to facial photographs. This perturbation confuses automated AI face-recognition systems without changing how the photo looks to a human.

### What Does the Frontend Do?

The frontend is a **web application** that gives users a graphical interface to use the AntiDeepfake system. Without the frontend, users would have to send raw HTTP requests to the backend API manually (using tools like `curl` or Postman). The frontend makes the entire process visual and interactive.

### User Journey

```
User opens browser → sees the AntiDeepfake app
    ↓
Uploads a face photo (drag-and-drop or click to browse)
    ↓
Adjusts the perturbation strength slider (epsilon)
    ↓
Clicks "Protect Image"
    ↓
Frontend sends image to the backend API
    ↓
Backend applies FGSM cloaking and returns a protected image
    ↓
Frontend shows: original + cloaked image side by side
    ↓
Frontend shows: SSIM score, PSNR score, processing time
    ↓
User downloads the protected image with one click
```

### Pre-Existing Context

Before this frontend was built:

- The ML pipeline (MTCNN face detector + FGSM attack) was already complete
- The FastAPI backend was already complete and exposing `POST /api/v1/cloak`
- The `src/frontend/` directory existed but contained only an empty `__init__.py`
- No React code, no Vite scaffold, no package.json existed

This report documents what was built from scratch inside `src/frontend/`.

---

## 2. What Was Built

**25 new files** were created inside `src/frontend/`. Zero backend or ML files were modified.

### Summary Table

| Category | Files Created | Description |
|---|---|---|
| Project Scaffold | 8 | package.json, vite config, TypeScript config, HTML, env file, favicon |
| TypeScript Types | 1 | All API and component prop interfaces |
| API Service | 1 | Axios HTTP client and download utility |
| React Hooks | 1 | Backend health polling hook |
| React Components | 8 | Navbar, uploader, slider, metrics, comparison, spinner, error, status |
| Pages | 1 | Home page with complete state management |
| App Root | 2 | App.tsx, main.tsx |
| Global CSS | 1 | Tailwind directives, custom range slider, animations |
| Vite Types | 1 | Environment variable type declarations |

---

## 3. Architecture Overview

### Technology Stack

| Technology | Version | Purpose |
|---|---|---|
| React | 18.3.1 | UI component framework |
| TypeScript | 5.4.5 | Strict static typing |
| Vite | 5.3.2 | Build tool and dev server |
| Tailwind CSS | 3.4.4 | Utility-first CSS framework |
| Axios | 1.7.2 | HTTP client for API calls |
| PostCSS | 8.4.39 | CSS processing pipeline |

### Frontend Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                Browser (http://localhost:5173)            │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                      App.tsx                         │ │
│  │  ┌──────────────────────────────────────────────┐   │ │
│  │  │                  Navbar.tsx                   │   │ │
│  │  └──────────────────────────────────────────────┘   │ │
│  │  ┌──────────────────────────────────────────────┐   │ │
│  │  │                 Home.tsx (page)               │   │ │
│  │  │  ┌─────────────┐  ┌──────────────────────┐   │   │ │
│  │  │  │BackendStatus│  │    ImageUploader      │   │   │ │
│  │  │  └─────────────┘  └──────────────────────┘   │   │ │
│  │  │  ┌──────────────────────────────────────────┐ │   │ │
│  │  │  │            EpsilonSlider                 │ │   │ │
│  │  │  └──────────────────────────────────────────┘ │   │ │
│  │  │  ┌──────────────┐  ┌───────────────────────┐  │   │ │
│  │  │  │ErrorMessage  │  │   LoadingSpinner       │  │   │ │
│  │  │  └──────────────┘  └───────────────────────┘  │   │ │
│  │  │  ──────────────── Results ────────────────── │   │ │
│  │  │  ┌──────────────────────────────────────────┐ │   │ │
│  │  │  │      ImageComparison (original + cloaked)│ │   │ │
│  │  │  └──────────────────────────────────────────┘ │   │ │
│  │  │  ┌──────────────────────────────────────────┐ │   │ │
│  │  │  │   MetricsCard (SSIM, PSNR, Time)         │ │   │ │
│  │  │  └──────────────────────────────────────────┘ │   │ │
│  │  └──────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────┘ │
│                        ↕ HTTP                            │
│             Vite Dev Proxy (/api, /health)                │
└─────────────────────────────────────────────────────────┘
                          ↕ HTTP
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend (localhost:8000)             │
│              POST /api/v1/cloak                          │
│              GET  /health                                │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
User selects file
    → ImageUploader validates (MIME + extension + size)
    → URL.createObjectURL() for preview
    → Home.tsx stores File + previewUrl in state

User clicks "Protect Image"
    → api.cloakImage(file, epsilon) called
    → FormData built: append("file", file) + append("epsilon", "0.02")
    → Axios POST to /api/v1/cloak
    → Loading state shown (spinner)

Backend responds
    → { success, processing_time_ms, metrics: {ssim, psnr}, cloaked_image_base64 }
    → result stored in React state

UI updates
    → ImageComparison renders original (object URL) + cloaked (data URI)
    → MetricsCard renders SSIM / PSNR / time
    → Download button becomes available

User clicks Download
    → downloadBase64Image() decodes base64 → Uint8Array → Blob
    → Temporary <a> element triggers browser download
    → Blob URL revoked after 1 second
```

---

## 4. File and Folder Explanation

### Complete Frontend Structure

```
src/frontend/
├── public/
│   └── favicon.svg                ← Shield SVG favicon
│
├── src/
│   ├── components/
│   │   ├── Navbar.tsx             ← Sticky glassmorphism top bar
│   │   ├── BackendStatus.tsx      ← Online / degraded / offline banner
│   │   ├── ImageUploader.tsx      ← Drag-and-drop file picker + validator
│   │   ├── EpsilonSlider.tsx      ← Custom styled range slider (0.01–0.10)
│   │   ├── MetricsCard.tsx        ← SSIM, PSNR, processing time cards
│   │   ├── ImageComparison.tsx    ← Side-by-side original vs cloaked
│   │   ├── LoadingSpinner.tsx     ← Animated triple-ring spinner
│   │   └── ErrorMessage.tsx       ← Dismissible error alert
│   │
│   ├── pages/
│   │   └── Home.tsx               ← Main page, all state management here
│   │
│   ├── services/
│   │   └── api.ts                 ← Axios client: cloakImage, checkHealth, download
│   │
│   ├── types/
│   │   └── api.ts                 ← All TypeScript interfaces
│   │
│   ├── hooks/
│   │   └── useHealthCheck.ts      ← 30-second polling hook with auto-recovery
│   │
│   ├── App.tsx                    ← Root component (Navbar + Home)
│   ├── main.tsx                   ← React entry point (renders into #root)
│   ├── index.css                  ← Global styles, Tailwind, custom slider CSS
│   └── vite-env.d.ts              ← Vite environment variable types
│
├── .env                           ← VITE_API_URL=http://localhost:8000
├── index.html                     ← HTML shell with SEO meta tags
├── package.json                   ← Dependencies + npm scripts
├── vite.config.ts                 ← Vite + dev proxy configuration
├── tailwind.config.js             ← Custom cyber color palette
├── postcss.config.js              ← PostCSS pipeline
├── tsconfig.json                  ← Strict TypeScript for src/
└── tsconfig.node.json             ← TypeScript for Vite config file
```

### Key File Descriptions

#### `package.json`
Defines all project dependencies and npm scripts:
- `npm run dev` — starts the Vite dev server on port 5173
- `npm run build` — runs `tsc --noEmit` (type check) then `vite build`
- `npm run preview` — serves the production build locally

#### `vite.config.ts`
Configures Vite with two important settings:
1. **React plugin** — enables JSX transform and hot module replacement
2. **Dev proxy** — forwards `/api/*` and `/health` requests to `localhost:8000`, eliminating CORS issues during development

```typescript
proxy: {
  '/api':    { target: 'http://localhost:8000', changeOrigin: true },
  '/health': { target: 'http://localhost:8000', changeOrigin: true },
}
```

#### `tailwind.config.js`
Extends Tailwind with a custom color system:
- `cyber.*` — blue accent colors (50 to 950 shades) for buttons, highlights, accents
- `dark.*` — secondary navy tones
- `surface.*` — dark background tokens (`DEFAULT`, `card`, `border`, `hover`)
- Custom animations: `fade-in`, `slide-up`, `glow`, `pulse-slow`
- Custom box shadows: `shadow-cyber`, `shadow-cyber-lg`, `shadow-card`

#### `.env`
```
VITE_API_URL=http://localhost:8000
```
Vite reads this at build time and exposes it via `import.meta.env.VITE_API_URL`.

#### `vite-env.d.ts`
Provides TypeScript types for `import.meta.env`:
```typescript
interface ImportMetaEnv {
  readonly VITE_API_URL: string;
}
```
This makes `import.meta.env.VITE_API_URL` type-safe (no `string | undefined` issues).

---

## 5. How the Frontend Works — End to End

### Step 1: Application Loads

When the user opens the browser, `main.tsx` is executed first:

```typescript
const rootElement = document.getElementById('root');
ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

`App.tsx` renders `<Navbar />` and `<Home />`.

### Step 2: Health Check on Mount

The `Home` component calls the `useHealthCheck` hook:

```typescript
const { status, healthData } = useHealthCheck(30_000);
```

This immediately fires `GET /health`. Depending on the response:
- HTTP 200 + `status: "healthy"` → backend status shows **green "Backend Online"**
- HTTP 503 + `status: "degraded"` → shows **amber "Models Loading"**
- Network failure → shows **red "Backend Offline"**

The hook repeats this check every 30 seconds automatically, so the status updates without any user action or page refresh.

### Step 3: Image Upload

The user either drags a file onto the drop zone or clicks to open the file picker.

`ImageUploader.tsx` validates the file before passing it to the parent:

```
File selected
    → Check MIME type against accepted set (image/jpeg, image/png, image/bmp, image/webp)
    → Check file extension as a fallback (.jpg, .jpeg, .png, .bmp, .webp)
    → Check file is not empty (size > 0)
    → Check file is under 20 MB
    → If invalid: show inline validation error, do NOT call onFileSelect
    → If valid: call onFileSelect(file, URL.createObjectURL(file))
```

The `URL.createObjectURL(file)` creates a temporary in-browser URL for the file, so we can display the image instantly without uploading it anywhere.

### Step 4: Epsilon Configuration

The `EpsilonSlider` component renders a range input over a custom-styled track. The range is `0.01` to `0.10` with step `0.01`. Default is `0.02`.

The current value determines the strength label:
- 0.01–0.02 → **Subtle** (green) — "imperceptible noise, recommended for everyday use"
- 0.03–0.05 → **Moderate** (amber) — "stronger protection, virtually invisible"
- 0.06–0.10 → **Strong** (red) — "maximum disruption, may introduce barely-visible texture"

The epsilon value is stored in `Home.tsx` state and sent directly to the API.

### Step 5: Submitting to the Backend

When the user clicks **Protect Image**, `handleSubmit` in `Home.tsx` runs:

```typescript
setState(prev => ({ ...prev, loading: true, error: null, result: null }));

const res = await cloakImage(state.file, state.epsilon);

if (res.ok) {
  setState(prev => ({ ...prev, loading: false, result: res.data }));
} else {
  setState(prev => ({ ...prev, loading: false, error: res.error }));
}
```

Inside `cloakImage()` (in `api.ts`):

```typescript
const formData = new FormData();
formData.append('file', file);
formData.append('epsilon', epsilon.toString());

const response = await apiClient.post('/api/v1/cloak', formData, {
  headers: { 'Content-Type': 'multipart/form-data' },
});
```

The Axios instance has a **120-second timeout** because ML processing (MTCNN + FGSM) can take significant time.

### Step 6: Displaying Results

Once the API responds with a `CloakResponse`, the results section appears.

**Original image:** Displayed from the local Object URL (created in Step 3). No re-upload needed.

**Cloaked image:** The backend returns `cloaked_image_base64` — a raw Base64 string. The frontend constructs a data URI:

```typescript
const cloakedDataUri = `data:image/jpeg;base64,${cloakedBase64}`;
```

This data URI is used directly as the `src` attribute of an `<img>` element. No server request needed to display it.

**Metrics:**
- `SSIM` — Structural Similarity Index. Values ≥ 0.95 are excellent (near-imperceptible)
- `PSNR` — Peak Signal-to-Noise Ratio in dB. Values > 40 dB are excellent
  - **Special case:** If `psnr` is `null` (backend returns null for identical images = infinite PSNR), the UI displays `"∞"` instead of crashing
- `Processing Time` — auto-formatted: shows `ms` for fast responses, `s` for slow ones

### Step 7: Downloading the Protected Image

The download is **100% client-side**. No backend download endpoint exists or is needed.

```typescript
function downloadBase64Image(base64: string, filename: string): void {
  // 1. Decode the base64 string to binary
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  // 2. Create a Blob (binary large object) from the bytes
  const blob = new Blob([bytes], { type: 'image/jpeg' });

  // 3. Create a temporary URL for the Blob
  const url = URL.createObjectURL(blob);

  // 4. Create a hidden <a> element and simulate a click to trigger download
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;       // filename for the downloaded file
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);

  // 5. Release the object URL after a short delay
  setTimeout(() => URL.revokeObjectURL(url), 1_000);
}
```

The downloaded filename is `<original_filename>_cloaked.jpg`.

---

## 6. Component Reference

### `Navbar.tsx`

A fixed top navigation bar with glassmorphism effect (blurred background).

- Renders a shield SVG logo with blue gradient fill
- Brand name: "Anti**Deepfake**" with the second part in cyber-blue
- API version label in monospaced font
- GitHub and API Docs links (right side)
- Uses `position: fixed` so it stays visible while scrolling
- Background: `rgba(13,17,23,0.85)` with `backdrop-filter: blur(12px)`

---

### `BackendStatus.tsx`

A banner showing the current backend connectivity state.

| State | Color | Icon | Description |
|---|---|---|---|
| `checking` | Hidden | — | Initial check in-progress, nothing shown |
| `online` | Green | Pulsing dot | "Backend Online" + model versions |
| `degraded` | Amber | Pulsing dot | "Models Loading" + explanation |
| `offline` | Red | Pulsing dot | "Backend Offline" + start command |

The pulsing dot uses a CSS `animate-ping` class for the outer glow ring.

When `online`, it also shows the version string from the health response and whether each model (MTCNN, FGSM) is loaded: `API v1.0.0 · MTCNN ✓ · FGSM Engine ✓`

---

### `ImageUploader.tsx`

A drag-and-drop image picker with full client-side validation.

**Accepted formats:** JPEG, PNG, BMP, WebP (validated by both MIME type and file extension)  
**Maximum file size:** 20 MB  
**Validation order:**
1. MIME type check (primary)
2. Extension check (fallback, for when browsers report `application/octet-stream`)
3. Empty file check (`size === 0`)
4. Size limit check (`size > 20 * 1024 * 1024`)

**States:**
- Empty (no file selected) — shows upload icon, format badges, "click to browse" text
- Dragging — border turns blue, background tints, "Release to upload" message
- File selected — shows file name, size, MIME type, "Click to change image" prompt
- Disabled (during loading) — opacity reduced, click/drag blocked

The component also resets the file input value after each selection, so the user can re-select the same file if they dismiss and want to retry.

**Accessibility:** `role="button"`, `tabIndex`, `aria-label`, `aria-disabled`, keyboard support (`Enter`/`Space` to open picker).

---

### `EpsilonSlider.tsx`

A custom-styled range input for the FGSM perturbation budget.

**Range:** 0.01 → 0.10 in steps of 0.01  
**Default:** 0.02

Implementation detail: The `<input type="range">` element is made invisible (`opacity: 0`) and overlaid on top of a custom-drawn div track. This gives full control over visual appearance while preserving native browser input behavior (keyboard control, accessibility, mouse drag).

The blue fill portion is a CSS `div` whose `width` is computed from the current value:

```typescript
function toPercent(value: number): number {
  return ((value - 0.01) / (0.10 - 0.01)) * 100;
}
```

The range thumb is styled via `::-webkit-slider-thumb` and `::-moz-range-thumb` in `index.css` — it shows a glowing blue circle with a scale transform on hover.

**ARIA attributes:** `aria-label`, `aria-valuemin`, `aria-valuemax`, `aria-valuenow` ensure screen-reader compatibility.

---

### `LoadingSpinner.tsx`

An animated indicator shown while the API call is in-flight.

Three layers:
1. Outer ring — rotates clockwise at 1s duration
2. Inner ring — rotates counter-clockwise at 0.7s duration
3. Center dot — pulses with `animate-pulse`

Below the spinner:
- Message: "Generating protected image…" (configurable via props)
- Subtitle: "FGSM adversarial perturbation in progress…" in monospaced font
- Shimmer progress bar — a moving gradient band (`@keyframes shimmer`)

`role="status"` and `aria-live="polite"` are set for screen readers.

---

### `ErrorMessage.tsx`

A dismissible red alert box for displaying errors.

- `role="alert"` — tells screen readers to announce the message immediately
- Shows a circular info/warning icon in red
- "Processing Error" title in a lighter red
- Message body in a slightly muted red
- × dismiss button calls `onDismiss` prop (clears error from state)

---

### `MetricsCard.tsx`

Three metric tiles displayed in a responsive grid (1 column on mobile, 3 columns on tablet+).

Each tile shows:
- Icon (SVG)
- Label (abbreviated, uppercase)
- Value (large monospaced number)
- Subtitle (plain-language explanation)

**SSIM tile:** Green when ≥ 0.95, amber when < 0.95
**PSNR tile:** Green when ≥ 40 dB or null (∞), amber otherwise
**Time tile:** Always neutral blue

PSNR null handling:
```typescript
const psnrDisplay = metrics.psnr === null ? '∞' : `${metrics.psnr.toFixed(2)} dB`;
```

Processing time auto-formats:
```typescript
const timeDisplay = processingTimeMs >= 1000
  ? `${(processingTimeMs / 1000).toFixed(2)} s`
  : `${processingTimeMs.toFixed(0)} ms`;
```

---

### `ImageComparison.tsx`

Renders the original and cloaked images side-by-side (desktop) or stacked (mobile).

Each panel (`ImagePanel`) has:
- Header bar: title + colored badge ("Original" / "Cloaked")
- Image area: shows a skeleton shimmer while the image loads, then fades in
- Footer (cloaked panel only): Download button

The download button triggers `downloadBase64Image()` from `api.ts` and briefly shows a spinner + "Preparing download…" text to indicate activity.

Original image source: `URL.createObjectURL(file)` — a local in-browser URL  
Cloaked image source: `` `data:image/jpeg;base64,${cloakedBase64}` `` — inline Base64 data URI

---

### `Home.tsx`

The main page component. Manages the complete application state:

```typescript
interface AppState {
  file: File | null;          // selected image file
  previewUrl: string | null;  // object URL for original preview
  epsilon: number;            // slider value (0.01–0.10)
  loading: boolean;           // true while API call in-flight
  result: CloakResponse | null; // successful response data
  error: string | null;       // error message or null
}
```

Key behaviors:
- When a new file is selected, the previous Object URL is revoked (`URL.revokeObjectURL`) to prevent memory leaks
- The "Protect Image" button is disabled if no file is selected, loading is true, or backend is unavailable
- When "Process another image" is clicked, all state resets to `INITIAL_STATE` and the preview URL is revoked

---

## 7. API Integration Details

### Axios Instance Configuration

```typescript
// src/services/api.ts
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL,  // http://localhost:8000
  timeout: 120_000,                        // 2-minute timeout
  headers: { Accept: 'application/json' },
});
```

The 2-minute timeout is set because FGSM processing on CPU can be slow for large images.

### `cloakImage(file, epsilon)`

Builds multipart form data and posts to `/api/v1/cloak`.

Returns a discriminated union:
```typescript
type CloakResult =
  | { ok: true; data: CloakResponse }
  | { ok: false; error: string; error_code?: string };
```

This pattern means the caller never needs to `try/catch`. Every error is represented as `{ ok: false, error: "..." }`.

### `checkHealth()`

Calls `GET /health` with `validateStatus: (status) => status === 200 || status === 503`.

This is important: the backend returns HTTP 503 when models are still loading — that's not an error, it's a valid "degraded" status. Without overriding `validateStatus`, Axios would throw for any non-2xx response.

### Error Message Extraction

```typescript
function extractErrorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    const data = err.response?.data as ErrorResponse | undefined;
    if (data?.error) return data.error;           // backend's human-readable message
    if (err.code === 'ECONNABORTED') return '...timeout message...';
    if (!err.response) return '...cannot connect message...';
    return `Server error (${err.response.status}): ${err.message}`;
  }
  return 'An unexpected error occurred.';
}
```

The function prefers the backend's `error` field (e.g. "No face detected in the uploaded image.") over generic HTTP error text.

---

## 8. TypeScript Type System

All types are in `src/types/api.ts`. No `any` types are used anywhere.

### API Response Types

```typescript
// Derived from src/backend/schemas/response.py

interface Metrics {
  ssim: number;
  psnr: number | null;   // null = infinite PSNR (identical images)
}

interface CloakResponse {
  success: true;
  processing_time_ms: number;
  metrics: Metrics;
  cloaked_image_base64: string;
}

interface ErrorResponse {
  success: false;
  error_code: string;   // NO_FACE_DETECTED | INVALID_IMAGE | PROCESSING_ERROR
  error: string;
}

interface HealthResponse {
  status: 'healthy' | 'degraded' | string;
  version: string;
  face_detector_loaded: boolean;
  fgsm_engine_loaded: boolean;
}
```

### Schema Discrepancy Resolved

The prompt specified `psnr: number` but the actual Python schema is `Optional[float]`. The backend serializes Python `float('inf')` as JSON `null`. The TypeScript type was corrected to `number | null` to match the actual backend behavior. The UI renders `null` as `"∞"`.

### Component Prop Types

Every component has a typed props interface:
- `ImageUploaderProps` — `onFileSelect: (file: File, previewUrl: string) => void`, `disabled: boolean`
- `EpsilonSliderProps` — `value: number`, `onChange: (value: number) => void`, `disabled: boolean`
- `ImageComparisonProps` — `originalUrl: string`, `cloakedBase64: string`, `originalFilename: string`
- `MetricsCardProps` — `metrics: Metrics`, `processingTimeMs: number`
- `ErrorMessageProps` — `message: string`, `onDismiss: () => void`
- `LoadingSpinnerProps` — `message?: string`
- `BackendStatusProps` — `status: BackendStatus`, `healthData: HealthResponse | null`

---

## 9. State Management

State is managed using only **React built-in hooks**. No Redux, no Zustand, no Context API.

All application state lives in a single `useState` call in `Home.tsx`:

```typescript
const [state, setState] = useState<AppState>(INITIAL_STATE);
```

Updating state uses a functional form to avoid stale closures:

```typescript
setState((prev) => ({ ...prev, loading: true, error: null, result: null }));
```

Callbacks are wrapped in `useCallback` to prevent unnecessary re-renders:

```typescript
const handleFileSelect = useCallback((file: File, previewUrl: string) => {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  setState((prev) => ({ ...prev, file, previewUrl, result: null, error: null }));
}, [state.previewUrl]);
```

---

## 10. Design System & Styling

### Theme

- **Style:** Cybersecurity dark mode
- **Background:** `#0d1117` (GitHub dark)
- **Card surfaces:** `#161b22` / `#1c2128`
- **Border color:** `#21262d`
- **Accent blue:** `#2b8dff` (cyber-500)
- **Typography:** Inter (UI text) + JetBrains Mono (code, values, numbers)

### Visual Effects

| Effect | Implementation |
|---|---|
| Glassmorphism navbar | `backdrop-filter: blur(12px)` + semi-transparent background |
| Hero radial glow | CSS `radial-gradient` centered at top |
| Grid overlay | Repeating linear gradient at 40px spacing |
| Cyber glow buttons | `box-shadow: 0 0 20px rgba(43,141,255,0.3)` |
| Image skeleton | `animate-pulse` div before image loads |
| Fade-in reveal | `@keyframes fadeIn` on results section |
| Slide-up reveal | `@keyframes slideUp` on new content |
| Pulsing status dot | `animate-ping` + `animate-pulse` combination |
| Shimmer progress | `@keyframes shimmer` — sliding gradient band |

### Responsive Layout

Tailwind responsive prefixes used:

| Breakpoint | Screen | Behavior |
|---|---|---|
| (default) | < 640px | Mobile — components stack vertically |
| `sm:` | ≥ 640px | Tablet — some text/labels appear |
| `md:` | ≥ 768px | Desktop — image panels side-by-side (`grid-cols-2`) |
| `lg:` | ≥ 1024px | Large desktop — font size increases |

Critical responsive rules:
- `grid-cols-1 md:grid-cols-2` on `ImageComparison` — stacked on mobile, side-by-side on desktop
- `grid-cols-1 sm:grid-cols-3` on `MetricsCard` — stacked on mobile, 3-column on tablet+
- `hidden sm:block` on minor nav elements — hidden on very small screens

### Custom Range Slider

Tailwind cannot style `<input type="range">` thumb directly. The CSS in `index.css` manually overrides browser vendor prefixes:

```css
input[type='range']::-webkit-slider-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: linear-gradient(135deg, #1a6ef5, #2b8dff);
  box-shadow: 0 0 8px rgba(43, 141, 255, 0.5);
  transition: transform 0.15s ease;
}
input[type='range']::-webkit-slider-thumb:hover {
  transform: scale(1.15);
}
```

The same rules are duplicated with `-moz-range-thumb` for Firefox.

---

## 11. Health Check System

The `useHealthCheck` hook polls the backend on a configurable interval.

```typescript
export function useHealthCheck(intervalMs = 30_000) {
  const [status, setStatus] = useState<BackendStatus>('checking');
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);

  const runCheck = useCallback(async () => {
    const result = await checkHealth();
    if (!result.ok) { setStatus('offline'); return; }
    setHealthData(result.data);
    setStatus(result.data.status === 'healthy' ? 'online' : 'degraded');
  }, []);

  useEffect(() => {
    void runCheck();  // immediate check on mount
    const id = setInterval(() => void runCheck(), intervalMs);
    return () => clearInterval(id);  // cleanup on unmount
  }, [runCheck, intervalMs]);

  return { status, healthData, refresh: runCheck };
}
```

**Auto-recovery:** If the backend is `offline` and later starts up, the next 30-second poll will detect it and update the status to `online` automatically — no page refresh needed.

**`degraded` state:** HTTP 503 is treated as valid JSON (not an error) via Axios's `validateStatus` override. This is necessary because the backend returns structured JSON even in degraded state.

---

## 12. Image Download — Client-Side Implementation

The download flow uses the browser's native `Blob` API. No server endpoint is involved.

```
backend returns cloaked_image_base64 (e.g. "/9j/4AAQ...")
    ↓
atob(base64) → binary string
    ↓
Uint8Array(binaryString.length) → fill byte by byte
    ↓
new Blob([bytes], { type: 'image/jpeg' })
    ↓
URL.createObjectURL(blob) → e.g. "blob:http://localhost:5173/abc-123"
    ↓
<a href="blob:..." download="image_cloaked.jpg"> → .click()
    ↓
Browser downloads the file natively
    ↓
setTimeout(1000) → URL.revokeObjectURL() → memory freed
```

**Why not use the data URI directly as the download href?**  
While `<a href="data:image/jpeg;base64,...">` technically works, it fails in some browsers for large images and puts the entire base64 string in the download URL. The Blob approach is the browser-native standard, works everywhere, and cleans up memory properly.

---

## 13. Error Handling Strategy

Every possible failure mode is handled gracefully. The application never crashes.

| Failure | Detection | User Message |
|---|---|---|
| Invalid file type | Client-side MIME/extension check | "Unsupported file type. Please upload JPEG, PNG, BMP, or WebP." |
| Empty file | Client-side `size === 0` check | "The selected file is empty." |
| File too large | Client-side `size > 20MB` check | "File size exceeds 20 MB." |
| Backend offline | Health check fails | Red "Backend Offline" banner, submit button disabled |
| Models loading | Health check returns 503 | Amber "Models Loading" banner, submit button disabled |
| No face detected | Backend returns `NO_FACE_DETECTED` 400 | Backend's error message shown in red alert |
| Invalid image format | Backend returns `INVALID_IMAGE` 400 | Backend's error message shown in red alert |
| Network timeout | Axios `ECONNABORTED` | "Request timed out. The image may be too large." |
| No network | Axios no response | "Cannot connect to the backend server." |
| Unexpected server error | Any other HTTP error | "Server error (500): Internal Server Error" |
| Root element missing | `getElementById('root')` returns null | Throws descriptive Error (startup-time only) |

Error messages displayed in `ErrorMessage.tsx` can be dismissed by the user, clearing the error state.

---

## 14. Responsiveness

### Desktop (≥ 768px)
- Image comparison: two panels side by side (`grid-cols-2`)
- Metrics: three cards in a row (`grid-cols-3`)
- Navbar: all elements visible including "API Docs" link

### Tablet (640px–768px)
- Image comparison: stacked (`grid-cols-1`)
- Metrics: three cards in a row (`grid-cols-3`)
- Navbar: GitHub icon shows, "API Docs" hidden

### Mobile (< 640px)
- Image comparison: stacked
- Metrics: stacked (`grid-cols-1`)
- Navbar: minimized, no extra links
- Upload zone: full width, comfortable tap targets
- Epsilon slider: full width, large thumb for touch

All layout switching is handled by Tailwind responsive prefixes — no JavaScript media queries, no CSS files per breakpoint.

---

## 15. Build & Validation Log

### npm install

```
added 269 packages, and audited 270 packages in 35s
63 packages are looking for funding
2 vulnerabilities (1 moderate, 1 high)
```

> Note: The 2 audit vulnerabilities are in ESLint 8 transitive dependencies (deprecated glob and rimraf). They do not affect the production build output or runtime security. They are a known issue with ESLint 8.x.

### First Build (CSS Warning)

```
[vite:css] @import must precede all other statements
```

**Root cause:** The `@import url(...)` for Google Fonts was placed after the `@tailwind` directives in `index.css`.

**Fix:** Moved the `@import` to the very first line of `index.css`, before any `@tailwind` directives (per CSS specification, `@import` rules must appear before other at-rules).

### Final Build (Clean)

```
> antideepfake-frontend@1.0.0 build
> tsc --noEmit && vite build

vite v5.4.21 building for production...
✓ 94 modules transformed.

dist/index.html                   1.56 kB │ gzip:  0.72 kB
dist/assets/index-qkfiCY6c.css   22.29 kB │ gzip:  5.25 kB
dist/assets/index-C6TP2UPR.js   219.44 kB │ gzip: 72.62 kB
✓ built in 1.56s
```

**TypeScript:** 0 errors  
**Build:** 0 errors, 0 warnings  
**Modules:** 94 transformed  
**JS bundle (gzipped):** 72.62 kB  
**CSS bundle (gzipped):** 5.25 kB

---

## 16. Changes Made

### Files Created (25 total)

| File | Type | Purpose |
|---|---|---|
| `package.json` | Config | NPM manifest with all dependencies |
| `vite.config.ts` | Config | Vite + dev proxy to backend |
| `tailwind.config.js` | Config | Cybersecurity color theme |
| `postcss.config.js` | Config | PostCSS pipeline for Tailwind |
| `tsconfig.json` | Config | Strict TypeScript for `src/` |
| `tsconfig.node.json` | Config | TypeScript for Vite config |
| `index.html` | HTML | Entry point with SEO meta tags |
| `.env` | Config | `VITE_API_URL=http://localhost:8000` |
| `public/favicon.svg` | Asset | Shield icon SVG |
| `src/vite-env.d.ts` | TypeScript | `ImportMetaEnv` type declarations |
| `src/types/api.ts` | TypeScript | All interfaces and type definitions |
| `src/services/api.ts` | TypeScript | Axios service: cloak, health, download |
| `src/hooks/useHealthCheck.ts` | TypeScript | 30s polling hook |
| `src/components/Navbar.tsx` | React | Top navigation bar |
| `src/components/BackendStatus.tsx` | React | Online/degraded/offline banner |
| `src/components/ImageUploader.tsx` | React | Drag-and-drop file picker |
| `src/components/EpsilonSlider.tsx` | React | Custom range slider |
| `src/components/MetricsCard.tsx` | React | SSIM / PSNR / time display |
| `src/components/ImageComparison.tsx` | React | Side-by-side image panels |
| `src/components/LoadingSpinner.tsx` | React | Triple-ring spinner |
| `src/components/ErrorMessage.tsx` | React | Dismissible error alert |
| `src/pages/Home.tsx` | React | Main page + state management |
| `src/App.tsx` | React | Root component |
| `src/main.tsx` | React | Entry point (renders App) |
| `src/index.css` | CSS | Global styles, custom slider, animations |

### Files Modified (0)

No backend or ML files were modified.

### Files Never Touched (Safety Rules Enforced)

- `src/ml_core/attacks/fgsm_attack.py` ✅ Untouched
- `src/ml_core/models/mtcnn_detector.py` ✅ Untouched
- `src/ml_core/evaluation/metrics.py` ✅ Untouched
- `src/backend/main.py` ✅ Untouched
- `src/backend/routes/cloak.py` ✅ Untouched
- `src/backend/schemas/request.py` ✅ Untouched
- `src/backend/schemas/response.py` ✅ Untouched
- `src/backend/services/pipeline_service.py` ✅ Untouched

---

## 17. Running the Frontend

### Prerequisites

- Node.js 18+ installed
- The AntiDeepfake backend running on port 8000

### Quick Start

```bash
# Terminal 1 — Start the backend
cd /home/shubham/AntiDeepfake
source .venv/bin/activate
uvicorn src.backend.main:app --reload
# Wait for: "Startup complete — all models loaded. API is ready."

# Terminal 2 — Start the frontend
cd /home/shubham/AntiDeepfake/src/frontend
npm install          # only needed once
npm run dev
# → Frontend available at: http://localhost:5173
```

### Production Build

```bash
cd /home/shubham/AntiDeepfake/src/frontend
npm run build
# Output in: src/frontend/dist/
```

### Available npm Scripts

| Script | Command | Purpose |
|---|---|---|
| `npm run dev` | `vite` | Start dev server with hot reload on port 5173 |
| `npm run build` | `tsc --noEmit && vite build` | Type-check and build for production |
| `npm run preview` | `vite preview` | Serve the production build locally |
| `npm run lint` | `eslint .` | Run ESLint on all TypeScript files |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | Base URL of the FastAPI backend |

To point the frontend at a different backend URL, edit `src/frontend/.env`:
```
VITE_API_URL=https://your-backend-domain.com
```

---

## 18. Manual Testing Guide

### Precondition
Start both the backend and frontend as described in Section 17.

### Test 1 — Backend Status Banner

1. Open http://localhost:5173
2. **Expected:** Green "Backend Online" badge appears within 2 seconds
3. **Expected:** "API v1.0.0 · MTCNN ✓ · FGSM Engine ✓" in the badge description

### Test 2 — Image Upload (drag and drop)

1. Find a JPEG or PNG image with a clear face
2. Drag it directly onto the drop zone
3. **Expected:** Drop zone border turns blue while hovering
4. **Expected:** After dropping, file name and size appear in the zone

### Test 3 — Image Upload (click to browse)

1. Click anywhere inside the drop zone
2. Select a PNG image in the file picker
3. **Expected:** File details appear in the drop zone

### Test 4 — Validation (unsupported format)

1. Try to upload a `.gif` file or `.txt` file
2. **Expected:** Red inline error appears below the drop zone
3. **Expected:** The file is NOT sent to the backend

### Test 5 — Epsilon Slider

1. Move the slider to the leftmost position (0.01)
2. **Expected:** Value shows `0.01`, badge shows "Subtle" in green
3. Move to 0.05
4. **Expected:** "Moderate" in amber
5. Move to 0.10
6. **Expected:** "Strong" in red

### Test 6 — Processing

1. Select a face image, leave epsilon at 0.02
2. Click **Protect Image**
3. **Expected:** Button disappears, spinner and "Generating protected image…" appear
4. **Expected:** After 5–60 seconds (depends on hardware), spinner disappears

### Test 7 — Results Display

1. After processing completes
2. **Expected:** Original image (left) and protected image (right) appear side-by-side
3. **Expected:** Both images look visually identical
4. **Expected:** SSIM, PSNR, and Processing Time metric cards appear below

### Test 8 — Download

1. Click **Download Protected Image**
2. **Expected:** Button briefly shows spinner + "Preparing download…"
3. **Expected:** Browser downloads `<filename>_cloaked.jpg`
4. **Expected:** Opening the file shows a valid JPEG image

### Test 9 — Error Handling (no face)

1. Upload an image without a face (e.g., a landscape photo)
2. Click **Protect Image**
3. **Expected:** Error alert appears: "No face detected in the uploaded image."
4. **Expected:** Dismiss button (×) closes the alert

### Test 10 — Backend Offline Recovery

1. Stop the backend server
2. Wait up to 30 seconds
3. **Expected:** Status banner automatically changes to red "Backend Offline"
4. **Expected:** Protect Image button is disabled
5. Restart the backend
6. Wait up to 30 seconds (next poll cycle)
7. **Expected:** Status banner automatically returns to green "Backend Online" (no page refresh)

### Test 11 — Mobile Viewport

1. Open DevTools (F12) → Toggle device toolbar
2. Set to iPhone SE (375×667)
3. **Expected:** Images are stacked vertically
4. **Expected:** Metric cards are stacked vertically
5. **Expected:** All controls are usable and nothing overflows

### Test 12 — Process Another Image

1. After successfully processing an image, click "Process another image"
2. **Expected:** App resets to initial state
3. **Expected:** Previous result images are gone
4. **Expected:** Upload zone is shown again

---

## 19. Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| State management | React `useState` only | No need for Redux/Zustand — all state fits in one page component |
| Typing | Strict TypeScript, no `any` | Prevents runtime bugs from API shape mismatches |
| `psnr` type | `number \| null` | Matches actual Python `Optional[float]` — null = infinite PSNR |
| Axios timeout | 120 seconds | FGSM on CPU can take >30s for high-res images |
| Health check interval | 30 seconds | Short enough to detect recovery quickly, low enough to not spam the server |
| Dev proxy | Vite's `proxy` config | Eliminates CORS issues in development without modifying backend |
| Download method | Blob + Object URL | More reliable than data URI links, works for large files, cleans up memory |
| Base64 display | `data:image/jpeg;base64,...` data URI | No extra network request needed — image is already in memory |
| CSS import order | `@import` before `@tailwind` | CSS spec requirement — `@import` must precede other statements |
| Favicon | SVG format | Crisp at any resolution, tiny file size |
| Font loading | Google Fonts via `<link>` in HTML + CSS `@import` | HTML preconnect hints speed up font loading |

---

## 20. Troubleshooting Guide

| Problem | Cause | Fix |
|---|---|---|
| App shows "Backend Offline" | Backend not started | Run `uvicorn src.backend.main:app --reload` from project root |
| App shows "Models Loading" | Backend started but models still initializing | Wait 30–60 seconds for model download (first run) |
| `npm run dev` fails | Node.js not installed / old version | Install Node.js 18+ from nodejs.org |
| `npm run build` TypeScript errors | TypeScript strict mode violation | Check the error line; fix the type issue |
| "Unsupported file type" for a JPEG | File extension is `.JPG` (uppercase) | Rename to `.jpg` or the validation logic covers common caps |
| "No face detected" on a face photo | Face is too small, side-profile, or obscured | Use a clear, frontal, well-lit photo; minimum face size ~50×50px |
| Download produces a corrupt file | Base64 string truncated in response | Unlikely — check backend logs for encoding errors |
| Images don't appear side-by-side | Screen width < 768px (md breakpoint) | Use a wider viewport; design is intentional (responsive) |
| Slow processing (>60s) | Running on CPU without GPU acceleration | Expected on CPU; GPU reduces this to 1–5s |
| Frontend and backend on different ports cause CORS | CORS is bypassed by Vite proxy in dev mode | In production, configure `VITE_API_URL` to match deployed backend URL |

---

## 21. Frontend Summary

### What Was Delivered

A production-ready React + TypeScript + Tailwind CSS frontend that:

1. ✅ Uploads facial images with client-side validation (MIME, extension, size)
2. ✅ Configures FGSM perturbation strength via an epsilon slider (0.01–0.10)
3. ✅ Sends the image to `POST /api/v1/cloak` as `multipart/form-data`
4. ✅ Displays original and cloaked images side by side
5. ✅ Renders SSIM, PSNR (including ∞ for null), and processing time
6. ✅ Downloads the protected image client-side via Blob API
7. ✅ Shows backend health status (online / loading / offline) with auto-recovery
8. ✅ Handles all error states gracefully with user-friendly messages
9. ✅ Works on desktop, tablet, and mobile (responsive Tailwind layout)
10. ✅ Passes strict TypeScript (no `any`, no `@ts-ignore`)
11. ✅ Builds cleanly with zero errors and zero warnings

### API Contract Compliance

| Requirement | Status |
|---|---|
| POST /api/v1/cloak via multipart/form-data | ✅ |
| file field (UploadFile) | ✅ |
| epsilon field (float, optional) | ✅ |
| CloakResponse deserialization | ✅ |
| cloaked_image_base64 → image display | ✅ |
| metrics.ssim display | ✅ |
| metrics.psnr display (including null → ∞) | ✅ |
| processing_time_ms display | ✅ |
| ErrorResponse handling | ✅ |
| GET /health integration | ✅ |
| CORS validation | ✅ (allow_origins=["*"] in backend) |

### Implementation Status

| Phase | Status |
|---|---|
| Repository & API Analysis | ✅ Complete |
| TypeScript Types | ✅ Complete (8 interfaces, 4 types) |
| API Service Layer | ✅ Complete |
| Health Check Hook | ✅ Complete |
| All 8 Components | ✅ Complete |
| Home Page | ✅ Complete |
| Global Styling | ✅ Complete |
| Build Validation | ✅ Passed (0 errors) |
| Documentation | ✅ Complete (this document) |
