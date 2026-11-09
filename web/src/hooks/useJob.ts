"use client";

import { useEffect, useState } from "react";
import { api, isFinished, type Job } from "@/lib/api";

const POLL_MS = 500;

/** Poll a processing job until it completes, fails or is cancelled. Key the caller by jobId to reset state. */
export function useJob(jobId: string | null) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const poll = async () => {
      try {
        const next = await api.job(jobId);
        if (!active) return;
        setJob(next);
        setError(null);
        if (isFinished(next)) return;
      } catch (e) {
        if (!active) return;
        setError((e as Error).message);
      }
      timer = setTimeout(poll, POLL_MS);
    };
    poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [jobId]);

  return { job, setJob, error };
}
