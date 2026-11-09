import type { VehicleClass } from "./api";

export const VEHICLE_CLASSES: VehicleClass[] = ["car", "motorcycle", "bus", "truck"];

export const CLASS_EMOJI: Record<VehicleClass, string> = {
  car: "🚗",
  motorcycle: "🏍️",
  bus: "🚌",
  truck: "🚚",
};

// Same colours the backend draws boxes with (utils.CLASS_COLORS_BGR, converted to RGB).
export const CLASS_COLOR: Record<VehicleClass, string> = {
  car: "#3cb44b",
  motorcycle: "#ffa500",
  bus: "#008cff",
  truck: "#dc143c",
};

/** Same format as utils.format_seconds: "1h 02m 03s" / "2m 05s" / "4.2s". */
export function formatSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const pad = (n: number) => n.toString().padStart(2, "0");
  return hours ? `${hours}h ${pad(minutes)}m ${pad(secs)}s` : `${minutes}m ${pad(secs)}s`;
}

export const formatInt = (n: number) => n.toLocaleString("en-US");

export const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
