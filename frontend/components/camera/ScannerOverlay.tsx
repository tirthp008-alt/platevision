"use client";

import { DetectionItem } from "@/lib/types";
import { ScannerStatus } from "@/hooks/useLiveDetection";

interface ScannerOverlayProps {
  status: ScannerStatus;
  statusText: string;
  detections: DetectionItem[];
  isScanning: boolean;
}

export function ScannerOverlay({
  status,
  statusText,
  detections,
  isScanning,
}: ScannerOverlayProps) {
  const getStatusColor = () => {
    switch (status) {
      case "plate_detected":
        return "border-emerald-400 text-emerald-400 bg-emerald-500/20";
      case "low_confidence":
        return "border-amber-400 text-amber-400 bg-amber-500/20";
      case "error":
        return "border-rose-400 text-rose-400 bg-rose-500/20";
      case "scanning":
        return "border-cyan-400 text-cyan-400 bg-cyan-500/20";
      default:
        return "border-slate-400 text-slate-300 bg-slate-800/40";
    }
  };

  return (
    <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-4 overflow-hidden select-none">
      {/* Top Status Badge */}
      <div className="flex justify-center w-full">
        <div
          className={`flex items-center gap-2 px-4 py-1.5 rounded-full border backdrop-blur-md text-xs font-mono font-semibold tracking-wide transition-all shadow-lg ${getStatusColor()}`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              isScanning ? "bg-cyan-400 animate-ping" : "bg-slate-400"
            }`}
          />
          <span>{statusText}</span>
        </div>
      </div>

      {/* Center Target Box / Reticle */}
      <div className="relative mx-auto w-full max-w-xs sm:max-w-sm aspect-[3.2/1] my-auto">
        {/* Reticle Corner Brackets */}
        <div className="absolute -top-1 -left-1 w-6 h-6 border-t-3 border-l-3 border-cyan-400 rounded-tl" />
        <div className="absolute -top-1 -right-1 w-6 h-6 border-t-3 border-r-3 border-cyan-400 rounded-tr" />
        <div className="absolute -bottom-1 -left-1 w-6 h-6 border-b-3 border-l-3 border-cyan-400 rounded-bl" />
        <div className="absolute -bottom-1 -right-1 w-6 h-6 border-b-3 border-r-3 border-cyan-400 rounded-br" />

        {/* Central Grid Guidelines */}
        <div className="w-full h-full border border-dashed border-cyan-500/30 rounded-lg flex items-center justify-center bg-cyan-950/10 backdrop-brightness-110">
          <span className="text-[10px] uppercase font-mono text-cyan-400/60 tracking-wider">
            Target Plate Area
          </span>
        </div>

        {/* Animated Laser Scanning Line */}
        {isScanning && (
          <div className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_12px_#00f0ff] animate-scan" />
        )}
      </div>

      {/* Bounding Boxes for detected plates */}
      {detections.map((det) => {
        const nb = det.normalized_box;
        return (
          <div
            key={det.id}
            className="absolute border-2 border-emerald-400 bg-emerald-500/10 rounded transition-all duration-150 pointer-events-none shadow-[0_0_10px_rgba(16,185,129,0.5)]"
            style={{
              left: `${nb.x * 100}%`,
              top: `${nb.y * 100}%`,
              width: `${nb.width * 100}%`,
              height: `${nb.height * 100}%`,
            }}
          >
            <div className="absolute -top-6 left-0 bg-emerald-600 text-white font-mono text-[10px] font-bold px-1.5 py-0.5 rounded shadow">
              {det.formatted_text || det.normalized_text} ({Math.round(det.detection_confidence * 100)}%)
            </div>
          </div>
        );
      })}

      {/* Bottom helper text */}
      <div className="text-center">
        <p className="text-[11px] font-mono text-slate-400 bg-navy-950/60 backdrop-blur-md px-3 py-1 rounded-full inline-block border border-slate-800">
          Position license plate clearly inside the guide frame
        </p>
      </div>
    </div>
  );
}
