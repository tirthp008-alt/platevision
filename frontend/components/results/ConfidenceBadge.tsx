"use client";

import { formatPercentage, getConfidenceColor } from "@/lib/utils";

interface ConfidenceBadgeProps {
  label: string;
  confidence: number;
  showPercent?: boolean;
}

export function ConfidenceBadge({
  label,
  confidence,
  showPercent = true,
}: ConfidenceBadgeProps) {
  const { badgeBg, badgeText } = getConfidenceColor(confidence);

  return (
    <div
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs font-mono font-medium ${badgeBg}`}
    >
      <span className="text-slate-400">{label}:</span>
      <span className={`font-bold ${badgeText}`}>
        {showPercent ? formatPercentage(confidence) : confidence.toFixed(2)}
      </span>
    </div>
  );
}
