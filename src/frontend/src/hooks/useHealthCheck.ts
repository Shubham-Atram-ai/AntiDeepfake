/**
 * useHealthCheck.ts — Custom React hook for polling the backend health endpoint.
 *
 * Polls GET /health on mount and then every `intervalMs` milliseconds.
 * Automatically recovers if the backend becomes available without a page refresh.
 *
 * Returns:
 *   - status:     BackendStatus — 'checking' | 'online' | 'degraded' | 'offline'
 *   - healthData: HealthResponse | null — last successful health payload
 *   - refresh:    () => void — manually trigger a health check
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { checkHealth } from '../services/api';
import type { BackendStatus, HealthResponse } from '../types/api';

const DEFAULT_INTERVAL_MS = 30_000; // 30 seconds

interface UseHealthCheckReturn {
  status: BackendStatus;
  healthData: HealthResponse | null;
  refresh: () => void;
}

export function useHealthCheck(
  intervalMs: number = DEFAULT_INTERVAL_MS
): UseHealthCheckReturn {
  const [status, setStatus] = useState<BackendStatus>('checking');
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const runCheck = useCallback(async () => {
    const result = await checkHealth();

    if (!result.ok) {
      setStatus('offline');
      setHealthData(null);
      return;
    }

    setHealthData(result.data);

    if (result.data.status === 'healthy') {
      setStatus('online');
    } else {
      // degraded = server is up but models not fully loaded
      setStatus('degraded');
    }
  }, []);

  // Run immediately on mount, then on the polling interval
  useEffect(() => {
    void runCheck();

    intervalRef.current = setInterval(() => {
      void runCheck();
    }, intervalMs);

    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
      }
    };
  }, [runCheck, intervalMs]);

  return { status, healthData, refresh: runCheck };
}
