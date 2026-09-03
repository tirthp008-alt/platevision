"use client";

import { useEffect, useState } from "react";
import { fetchHealth } from "@/lib/api";
import { HealthStatus } from "@/lib/types";

export function StatusIndicator() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(false);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const data = await fetchHealth();
        if (mounted) {
          setHealth(data);
          setIsOnline(true);
        }
      } catch (e) {
        if (mounted) {
          setIsOnline(false);
        }
      }
    };
    check();
    const interval = setInterval(check, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full border border-slate-800 bg-navy-950/80 text-xs font-mono">
      <span
        className={`w-2 h-2 rounded-full ${
          isOnline ? "bg-emerald-400 animate-pulse" : "bg-rose-500"
        }`}
      />
      <span className="text-slate-400">
        {isOnline
          ? `Engine: ${health?.backend_type === "OnnxPlateDetector" ? "ONNX" : "Active"}`
          : "Connecting..."}
      </span>
    </div>
  );
}
