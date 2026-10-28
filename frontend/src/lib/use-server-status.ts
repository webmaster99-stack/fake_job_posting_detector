"use client";

// Render's free tier sleeps after 15 idle minutes and takes ~50 s to wake up. Ping /health on
// page load so the API is warm by the time the user submits a posting.

import { useCallback, useEffect, useState } from "react";
import { health } from "./api";

export type ServerStatus = "checking" | "waking" | "ready" | "unreachable";

const WAKING_AFTER_MS = 2_500; // a warm API answers well within this
const GIVE_UP_AFTER_MS = 120_000;

export function useServerStatus() {
  const [status, setStatus] = useState<ServerStatus>("checking");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const waking = setTimeout(() => !cancelled && setStatus("waking"), WAKING_AFTER_MS);
    health(GIVE_UP_AFTER_MS)
      .then(() => !cancelled && setStatus("ready"))
      .catch(() => !cancelled && setStatus("unreachable"))
      .finally(() => clearTimeout(waking));
    return () => {
      cancelled = true;
      clearTimeout(waking);
    };
  }, [attempt]);

  const retry = useCallback(() => {
    setStatus("checking");
    setAttempt((n) => n + 1);
  }, []);

  return { status, retry };
}
