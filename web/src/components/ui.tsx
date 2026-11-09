"use client";

import type { ButtonHTMLAttributes, ReactNode } from "react";
import type { Row } from "@/lib/api";

export function Card({ title, children, className = "" }: { title?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`glass hud rounded-xl p-4 sm:p-5 ${className}`}>
      {title && (
        <h3 className="mb-4 flex items-center gap-2 font-display text-[0.8rem] font-semibold uppercase tracking-[0.14em] text-foreground">
          <span className="h-3.5 w-1 rounded-full bg-gradient-accent" aria-hidden />
          {title}
        </h3>
      )}
      {children}
    </section>
  );
}

export function Metric({ label, value, accent }: { label: ReactNode; value: ReactNode; accent?: string }) {
  return (
    <div className="relative overflow-hidden rounded-lg border border-border bg-white px-3.5 py-3 transition hover:glow">
      <span
        className="absolute inset-x-0 top-0 h-0.5"
        style={{ background: accent ?? "linear-gradient(90deg, var(--accent), var(--accent-2))" }}
        aria-hidden
      />
      <div className="label-tech truncate">{label}</div>
      <div className="mt-1 truncate font-mono text-2xl font-semibold tabular-nums text-foreground">{value}</div>
    </div>
  );
}

type Tone = "info" | "success" | "warning" | "error";
const TONES: Record<Tone, string> = {
  info: "bg-accent-soft border-accent/30 [--bar:var(--accent)]",
  success: "bg-success-soft border-success/30 [--bar:var(--success)]",
  warning: "bg-warning-soft border-warning/30 [--bar:var(--warning)]",
  error: "bg-danger-soft border-danger/30 [--bar:var(--danger)]",
};
const ICONS: Record<Tone, string> = { info: "ℹ️", success: "✅", warning: "⚠️", error: "❌" };

export function Alert({ tone = "info", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <div
      className={`relative flex gap-2 overflow-hidden rounded-lg border py-2.5 pl-4 pr-3 text-sm ${TONES[tone]}`}
      role={tone === "error" ? "alert" : "status"}
    >
      <span className="absolute inset-y-0 left-0 w-1 bg-(--bar)" aria-hidden />
      <span aria-hidden>{ICONS[tone]}</span>
      <div className="min-w-0 whitespace-pre-line break-words">{children}</div>
    </div>
  );
}

type Variant = "primary" | "secondary" | "danger";
const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-gradient-accent text-white shadow-[0_8px_24px_-8px_rgba(124,58,237,0.55)] hover:shadow-[0_10px_30px_-6px_rgba(8,145,178,0.6)] hover:-translate-y-px",
  secondary: "border border-border bg-white text-foreground hover:border-accent/60 hover:text-accent hover:glow",
  danger: "bg-danger text-white shadow-[0_8px_24px_-10px_rgba(220,38,38,0.6)] hover:brightness-110",
};

export function Button({
  variant = "secondary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-1.5 rounded-lg px-4 py-2.5 text-sm font-semibold tracking-wide transition disabled:cursor-not-allowed disabled:opacity-45 disabled:shadow-none disabled:hover:translate-y-0 ${VARIANTS[variant]} ${className}`}
      {...props}
    />
  );
}

export function LinkButton({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      href={href}
      className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border bg-white px-4 py-2.5 text-sm font-semibold transition hover:border-accent/60 hover:text-accent hover:glow"
    >
      {children}
    </a>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  disabled,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="inline-flex flex-wrap gap-1 rounded-xl border border-border bg-panel-muted p-1" role="radiogroup">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          disabled={disabled}
          onClick={() => onChange(o.value)}
          className={`rounded-lg px-3.5 py-1.5 text-sm font-medium transition disabled:opacity-50 ${
            value === o.value
              ? "bg-white text-accent shadow-[0_0_0_1px_rgba(8,145,178,0.35),0_4px_14px_-6px_rgba(8,145,178,0.5)]"
              : "text-muted hover:text-foreground"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  disabled,
  hint,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  disabled?: boolean;
  hint?: ReactNode;
}) {
  return (
    <label className={`flex items-start gap-3 ${disabled ? "opacity-55" : "cursor-pointer"}`}>
      <span className="relative mt-0.5 inline-flex shrink-0">
        <input
          type="checkbox"
          className="peer sr-only"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className="h-5 w-9 rounded-full bg-[#d7dfeb] transition peer-checked:bg-gradient-accent peer-checked:shadow-[0_0_12px_rgba(8,145,178,0.55)] peer-focus-visible:ring-2 peer-focus-visible:ring-accent/40" />
        <span className="absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow transition peer-checked:translate-x-4" />
      </span>
      <span className="text-sm">
        {label}
        {hint && <span className="mt-0.5 block text-xs text-muted">{hint}</span>}
      </span>
    </label>
  );
}

export function ProgressBar({ fraction, indeterminate }: { fraction: number | null; indeterminate?: boolean }) {
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-[#e8eef7]">
      {indeterminate || fraction === null ? (
        <div className="shimmer h-full w-full rounded-full opacity-70" />
      ) : (
        <div
          className="h-full rounded-full bg-gradient-accent shadow-[0_0_12px_rgba(8,145,178,0.6)] transition-[width] duration-300"
          style={{ width: `${Math.round(fraction * 100)}%` }}
        />
      )}
    </div>
  );
}

export function DataTable({ rows, maxHeight = 420 }: { rows: Row[]; maxHeight?: number }) {
  if (!rows.length) return <p className="text-sm text-muted">No rows.</p>;
  const columns = Object.keys(rows[0]);
  return (
    <div className="overflow-auto rounded-lg border border-border bg-white" style={{ maxHeight }}>
      <table className="w-full border-collapse text-sm">
        <thead className="sticky top-0 bg-panel-muted/95 backdrop-blur">
          <tr>
            {columns.map((c) => (
              <th key={c} className="label-tech whitespace-nowrap border-b border-border px-3 py-2.5 text-left font-medium">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="transition hover:bg-accent-soft/60">
              {columns.map((c) => (
                <td key={c} className="whitespace-nowrap border-b border-border/70 px-3 py-2 font-mono text-[0.82rem] tabular-nums">
                  {row[c] === null || row[c] === undefined ? "" : String(row[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Expander({ title, children, onOpen }: { title: ReactNode; children: ReactNode; onOpen?: () => void }) {
  return (
    <details
      className="group glass rounded-xl"
      onToggle={(e) => (e.currentTarget as HTMLDetailsElement).open && onOpen?.()}
    >
      <summary className="flex cursor-pointer select-none list-none items-center px-4 py-3 text-sm font-semibold transition hover:text-accent">
        <span className="mr-2 inline-block text-accent transition group-open:rotate-90">▸</span>
        {title}
      </summary>
      <div className="border-t border-border px-4 py-3">{children}</div>
    </details>
  );
}

export function Field({ label, hint, children }: { label: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="text-sm font-medium">{label}</div>
      {children}
      {hint && <div className="text-xs leading-relaxed text-muted">{hint}</div>}
    </div>
  );
}

export const inputClass =
  "w-full rounded-lg border border-border bg-white px-3 py-2 text-sm outline-none transition focus:border-accent focus:shadow-[0_0_0_3px_rgba(8,145,178,0.15)] disabled:opacity-50";
