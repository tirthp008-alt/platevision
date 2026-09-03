import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPercentage(val: number): string {
  return `${Math.round(val * 100)}%`;
}

export function getConfidenceColor(conf: number): {
  badgeBg: string;
  badgeText: string;
  borderColor: string;
} {
  if (conf >= 0.85) {
    return {
      badgeBg: "bg-emerald-500/10 border-emerald-500/30 text-emerald-400",
      badgeText: "text-emerald-400",
      borderColor: "border-emerald-500",
    };
  } else if (conf >= 0.65) {
    return {
      badgeBg: "bg-amber-500/10 border-amber-500/30 text-amber-400",
      badgeText: "text-amber-400",
      borderColor: "border-amber-500",
    };
  }
  return {
    badgeBg: "bg-rose-500/10 border-rose-500/30 text-rose-400",
    badgeText: "text-rose-400",
    borderColor: "border-rose-500",
  };
}

export function getFormatStatusDetails(status: "valid" | "possible" | "uncertain") {
  switch (status) {
    case "valid":
      return {
        label: "Standard Format Verified",
        colorClass: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
        iconName: "ShieldCheck",
      };
    case "possible":
      return {
        label: "Likely Plate Format",
        colorClass: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40",
        iconName: "AlertCircle",
      };
    case "uncertain":
    default:
      return {
        label: "Uncertain Format",
        colorClass: "bg-amber-500/20 text-amber-300 border-amber-500/40",
        iconName: "HelpCircle",
      };
  }
}

export function triggerVibration(durationMs: number = 70) {
  if (typeof window !== "undefined" && "vibrate" in navigator) {
    try {
      navigator.vibrate(durationMs);
    } catch (e) {
      // Ignore if blocked
    }
  }
}

export async function validateAndReadFile(
  file: File,
  maxSizeMb: number = 10
): Promise<{ valid: boolean; error?: string; dataUrl?: string }> {
  if (!file) {
    return { valid: false, error: "No file selected." };
  }

  const validMimes = ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"];
  const fileExt = file.name.split(".").pop()?.toLowerCase();
  const validExts = ["jpg", "jpeg", "png", "webp", "heic", "heif"];

  if (!validMimes.includes(file.type) && !validExts.includes(fileExt || "")) {
    return {
      valid: false,
      error: `Unsupported file type (${file.type || fileExt}). Please select a JPG, PNG, WEBP, or HEIC image.`,
    };
  }

  const sizeMb = file.size / (1024 * 1024);
  if (sizeMb > maxSizeMb) {
    return {
      valid: false,
      error: `File size (${sizeMb.toFixed(1)} MB) exceeds maximum allowed limit of ${maxSizeMb} MB.`,
    };
  }

  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      resolve({ valid: true, dataUrl: e.target?.result as string });
    };
    reader.onerror = () => {
      resolve({ valid: false, error: "Failed to read image file." });
    };
    reader.readAsDataURL(file);
  });
}
