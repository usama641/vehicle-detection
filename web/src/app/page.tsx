"use client";

import { useEffect, useState } from "react";
import { AboutTab } from "@/components/AboutTab";
import { LiveTab } from "@/components/LiveTab";
import { SettingsPanel, defaultSettings, sanitizeSettings } from "@/components/SettingsPanel";
import { Alert, Button } from "@/components/ui";
import { VideoTab } from "@/components/VideoTab";
import { useStoredState } from "@/hooks/useStoredState";
import { API_URL, api, type AppConfig, type DetectionSettings } from "@/lib/api";

const TABS = [
  { id: "video", label: "📁 Video file" },
  { id: "live", label: "🎥 Live camera" },
  { id: "about", label: "ℹ️ About" },
] as const;
type TabId = (typeof TABS)[number]["id"];

const CHIP_TONES = {
  on: "border-success/30 bg-success-soft text-success",
  off: "border-danger/30 bg-danger-soft text-danger",
  wait: "border-warning/30 bg-warning-soft text-warning",
  neutral: "border-border bg-white text-muted",
} as const;

function StatusChip({ tone, children }: { tone: keyof typeof CHIP_TONES; children: React.ReactNode }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 font-mono text-xs ${CHIP_TONES[tone]}`}>
      {tone !== "neutral" && (
        <span className={`h-1.5 w-1.5 rounded-full bg-current ${tone === "on" || tone === "wait" ? "animate-pulse" : ""}`} />
      )}
      {children}
    </span>
  );
}

export default function Home() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [storedSettings, setStoredSettings] = useStoredState<DetectionSettings | null>("vd.settings", null);
  const [tab, setTab] = useStoredState<TabId>("vd.tab", "video");
  const [videoRunning, setVideoRunning] = useState(false);
  const [liveRunning, setLiveRunning] = useState(false);

  const [configAttempt, setConfigAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    api
      .config()
      .then((c) => active && (setConfig(c), setConfigError(null)))
      .catch((e: Error) => active && setConfigError(e.message));
    return () => {
      active = false;
    };
  }, [configAttempt]);

  const loadConfig = () => {
    setConfigError(null);
    setConfigAttempt((n) => n + 1);
  };

  const settings = config ? sanitizeSettings(storedSettings ?? defaultSettings(config), config) : null;
  const running = videoRunning || liveRunning;

  return (
    <div className="mx-auto w-full max-w-350 flex-1 px-4 py-6 sm:px-6 sm:py-8">
      <header className="glass hud mb-7 overflow-hidden rounded-2xl px-5 py-5 sm:px-7 sm:py-6">
        <div
          className="pointer-events-none absolute -right-24 -top-24 h-64 w-64 rounded-full bg-gradient-accent opacity-[0.12] blur-3xl"
          aria-hidden
        />
        <div className="relative flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-4">
            <div className="grid h-12 w-12 shrink-0 place-items-center rounded-xl bg-gradient-accent text-2xl shadow-[0_8px_24px_-8px_rgba(124,58,237,0.6)]">
              <span aria-hidden>🚗</span>
            </div>
            <div className="min-w-0">
              <div className="label-tech">YOLO · OpenCV · ByteTrack</div>
              <h1 className="font-display text-xl font-bold tracking-wide sm:text-2xl">
                <span className="text-gradient">AI VEHICLE</span> DETECTION
              </h1>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <StatusChip tone={configError ? "off" : config ? "on" : "wait"}>
              {configError ? "API offline" : config ? "API online" : "Connecting"}
            </StatusChip>
            {config && <StatusChip tone="neutral">{config.auto_device}</StatusChip>}
            {settings && <StatusChip tone="neutral">{settings.model_name || "custom model"}</StatusChip>}
            {running && <StatusChip tone="on">Processing</StatusChip>}
          </div>
        </div>
        <p className="relative mt-4 max-w-3xl text-sm text-muted">
          Detect <strong className="text-foreground">cars, motorcycles, buses and trucks</strong> in videos and live streams.
          Upload a clip, tune the model, and download the annotated video and reports.
        </p>
      </header>

      {configError && (
        <div className="space-y-3">
          <Alert tone="error">
            {configError}
            {"\n"}Start it from the api/ folder:{" "}
            <code className="font-mono">uvicorn main:app --host 127.0.0.1 --port 8000</code>
            {"\n"}(Frontend is configured for {API_URL}; change NEXT_PUBLIC_API_URL in web/.env.local if needed.)
          </Alert>
          <Button onClick={loadConfig}>↻ Retry</Button>
        </div>
      )}

      {!config && !configError && (
        <div className="glass rounded-xl p-5">
          <div className="label-tech mb-3">Connecting to the API…</div>
          <div className="h-1.5 overflow-hidden rounded-full bg-[#e8eef7]">
            <div className="shimmer h-full w-full" />
          </div>
        </div>
      )}

      {config && settings && (
        <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
          <aside className="glass hud w-full shrink-0 rounded-2xl p-5 lg:sticky lg:top-6 lg:max-h-[calc(100vh-3rem)] lg:w-80 lg:overflow-y-auto">
            <div className="label-tech mb-5 flex items-center justify-between">
              <span>Control panel</span>
              <span className={running ? "text-warning" : "text-success"}>{running ? "● locked" : "● ready"}</span>
            </div>
            <SettingsPanel config={config} settings={settings} onChange={setStoredSettings} disabled={running} />
            {running && <p className="mt-4 text-xs text-muted">Settings are locked while a job is running.</p>}
          </aside>

          <main className="min-w-0 flex-1">
            <nav
              className="mb-6 inline-flex max-w-full gap-1 overflow-x-auto rounded-xl border border-border bg-white/80 p-1 backdrop-blur"
              role="tablist"
            >
              {TABS.map((t) => (
                <button
                  key={t.id}
                  role="tab"
                  aria-selected={tab === t.id}
                  onClick={() => setTab(t.id)}
                  className={`whitespace-nowrap rounded-lg px-4 py-2 text-sm font-semibold transition ${
                    tab === t.id
                      ? "bg-gradient-accent text-white shadow-[0_6px_18px_-6px_rgba(124,58,237,0.55)]"
                      : "text-muted hover:bg-panel-muted hover:text-foreground"
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </nav>

            {/* Tabs stay mounted so a running job keeps its progress view when you switch tabs. */}
            <div hidden={tab !== "video"}>
              <VideoTab config={config} settings={settings} onRunningChange={setVideoRunning} />
            </div>
            <div hidden={tab !== "live"}>
              <LiveTab config={config} settings={settings} onRunningChange={setLiveRunning} />
            </div>
            <div hidden={tab !== "about"}>
              <AboutTab trackingDisclaimer={config.tracking.disclaimer} />
            </div>
          </main>
        </div>
      )}
    </div>
  );
}
