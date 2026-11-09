"use client";

import type { AppConfig, DetectionSettings, VehicleClass } from "@/lib/api";
import { CLASS_COLOR, CLASS_EMOJI, VEHICLE_CLASSES, capitalize } from "@/lib/format";
import { Field, Segmented, Toggle, inputClass } from "./ui";

export const CUSTOM_MODEL = "__custom__";

export function defaultSettings(config: AppConfig): DetectionSettings {
  return {
    model_name: config.default_model,
    device: "auto",
    imgsz: config.defaults.imgsz,
    confidence: config.defaults.confidence,
    iou: config.defaults.iou,
    all_classes: false,
    classes: [...VEHICLE_CLASSES],
    enable_tracking: false,
    save_records: true,
    max_frames: 0,
  };
}

/** Drop stored values that the current backend no longer supports. */
export function sanitizeSettings(settings: DetectionSettings, config: AppConfig): DetectionSettings {
  const fallback = defaultSettings(config);
  return {
    ...fallback,
    ...settings,
    device: config.devices.includes(settings.device) ? settings.device : "auto",
    imgsz: config.image_sizes.includes(settings.imgsz) ? settings.imgsz : fallback.imgsz,
    enable_tracking: config.tracking.available && settings.enable_tracking,
  };
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-4">
      <h2 className="flex items-center gap-2 font-display text-[0.72rem] font-semibold uppercase tracking-[0.16em]">
        <span className="h-3 w-1 rounded-full bg-gradient-accent" aria-hidden />
        {title}
        <span className="h-px flex-1 bg-linear-to-r from-border to-transparent" aria-hidden />
      </h2>
      {children}
    </div>
  );
}

function Slider({
  value,
  min,
  max,
  step,
  onChange,
  disabled,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <input
        type="range"
        className="w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="w-12 rounded-md border border-border bg-white py-0.5 text-center font-mono text-xs text-accent tabular-nums">{value.toFixed(2)}</span>
    </div>
  );
}

export function SettingsPanel({
  config,
  settings,
  onChange,
  disabled,
}: {
  config: AppConfig;
  settings: DetectionSettings;
  onChange: (s: DetectionSettings) => void;
  disabled?: boolean;
}) {
  const set = <K extends keyof DetectionSettings>(key: K, value: DetectionSettings[K]) => onChange({ ...settings, [key]: value });
  const isCustom = !config.models.includes(settings.model_name);
  const toggleClass = (name: VehicleClass, on: boolean) =>
    set("classes", on ? VEHICLE_CLASSES.filter((c) => c === name || settings.classes.includes(c)) : settings.classes.filter((c) => c !== name));

  return (
    <fieldset disabled={disabled} className="space-y-7">
      <Section title="Model">
        <Field
          label="YOLO model"
          hint="'n' (nano) is fastest; 's'/'m' are more accurate but slower. Official weights download automatically into api/models/ on first use."
        >
          <select
            className={inputClass}
            value={isCustom ? CUSTOM_MODEL : settings.model_name}
            onChange={(e) => set("model_name", e.target.value === CUSTOM_MODEL ? "" : e.target.value)}
          >
            {config.models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
            <option value={CUSTOM_MODEL}>Custom weights (.pt path)…</option>
          </select>
          {isCustom && (
            <input
              className={inputClass}
              placeholder="C:\models\my_model.pt"
              value={settings.model_name}
              onChange={(e) => set("model_name", e.target.value)}
            />
          )}
        </Field>

        <Field
          label="Device"
          hint={
            config.cuda_available
              ? `'auto' uses ${config.auto_device}.`
              : "No CUDA GPU detected by PyTorch - running on CPU."
          }
        >
          <select className={inputClass} value={settings.device} onChange={(e) => set("device", e.target.value as DetectionSettings["device"])}>
            {config.devices.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Inference image size" hint="Larger sizes detect small/distant vehicles better but are slower.">
          <select className={inputClass} value={settings.imgsz} onChange={(e) => set("imgsz", Number(e.target.value))}>
            {config.image_sizes.map((s) => (
              <option key={s} value={s}>
                {s} px
              </option>
            ))}
          </select>
        </Field>
      </Section>

      <Section title="Detection">
        <Field label="Confidence threshold">
          <Slider value={settings.confidence} min={0.05} max={0.95} step={0.05} onChange={(v) => set("confidence", v)} />
        </Field>
        <Field label="NMS IoU threshold" hint="Lower values merge overlapping boxes more aggressively.">
          <Slider value={settings.iou} min={0.1} max={0.9} step={0.05} onChange={(v) => set("iou", v)} />
        </Field>
        <Field label="Detection mode">
          <Segmented
            value={settings.all_classes ? "all" : "vehicles"}
            options={[
              { value: "vehicles", label: "Vehicles only" },
              { value: "all", label: "All COCO classes (80)" },
            ]}
            onChange={(v) => set("all_classes", v === "all")}
            disabled={disabled}
          />
        </Field>
        <Field label="Vehicle classes">
          <div className={`grid grid-cols-2 gap-2 ${settings.all_classes ? "opacity-50" : ""}`}>
            {VEHICLE_CLASSES.map((name) => (
              <label
                key={name}
                className="flex cursor-pointer items-center gap-2 rounded-lg border border-border bg-white px-2.5 py-1.5 text-sm transition hover:glow has-checked:bg-accent-soft/50"
                style={{ borderLeft: `3px solid ${CLASS_COLOR[name]}` }}
              >
                <input
                  type="checkbox"
                  checked={settings.classes.includes(name)}
                  disabled={settings.all_classes}
                  onChange={(e) => toggleClass(name, e.target.checked)}
                />
                {CLASS_EMOJI[name]} {capitalize(name)}
              </label>
            ))}
          </div>
        </Field>
      </Section>

      <Section title="Tracking & reports">
        <Toggle
          label="Enable ByteTrack tracking"
          checked={settings.enable_tracking}
          disabled={!config.tracking.available}
          onChange={(v) => set("enable_tracking", v)}
          hint={
            config.tracking.available
              ? "Assigns IDs to vehicles across frames to estimate unique vehicles."
              : `⚠️ ${config.tracking.message}`
          }
        />
        <Toggle
          label="Save frame-level detection records (CSV)"
          checked={settings.save_records}
          onChange={(v) => set("save_records", v)}
        />
        <Field label="Max frames to process" hint="0 = whole video. Useful for a quick test on long videos.">
          <input
            type="number"
            min={0}
            step={100}
            className={inputClass}
            value={settings.max_frames}
            onChange={(e) => set("max_frames", Math.max(0, Math.floor(Number(e.target.value) || 0)))}
          />
        </Field>
      </Section>
    </fieldset>
  );
}
