"use client";

import { useRef, useEffect, useState } from "react";
import { DetectionItem } from "@/lib/types";

interface DetectionCanvasProps {
  imageUrl: string;
  detections: DetectionItem[];
  selectedIndex: number | null;
  onSelectDetection: (index: number) => void;
}

export function DetectionCanvas({
  imageUrl,
  detections,
  selectedIndex,
  onSelectDetection,
}: DetectionCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  return (
    <div
      ref={containerRef}
      className="relative w-full rounded-2xl bg-black overflow-hidden border border-slate-800 shadow-2xl flex items-center justify-center select-none"
    >
      {/* Underlying Image */}
      <img
        src={imageUrl}
        alt="Analyzed Vehicle"
        onLoad={() => setImageLoaded(true)}
        className="w-full h-auto max-h-[60vh] object-contain block"
      />

      {/* Interactive Bounding Box Overlay Layers */}
      {imageLoaded && (
        <div className="absolute inset-0 pointer-events-auto">
          {detections.map((det, idx) => {
            const isSelected = selectedIndex === idx;
            const nb = det.normalized_box;

            return (
              <div
                key={det.id || idx}
                onClick={() => onSelectDetection(idx)}
                className={`absolute transition-all duration-150 cursor-pointer rounded ${
                  isSelected
                    ? "border-3 border-cyan-400 bg-cyan-400/20 shadow-[0_0_15px_#00f0ff]"
                    : "border-2 border-emerald-400 bg-emerald-500/15 hover:border-cyan-300 hover:bg-cyan-500/25"
                }`}
                style={{
                  left: `${nb.x * 100}%`,
                  top: `${nb.y * 100}%`,
                  width: `${nb.width * 100}%`,
                  height: `${nb.height * 100}%`,
                }}
              >
                {/* Confidence & Plate Label Badge */}
                <div
                  className={`absolute -top-7 left-0 px-2 py-0.5 rounded text-[10px] font-mono font-bold tracking-wider whitespace-nowrap shadow-md transition-colors ${
                    isSelected
                      ? "bg-cyan-400 text-navy-950"
                      : "bg-emerald-600 text-white"
                  }`}
                >
                  #{idx + 1} {det.formatted_text || det.normalized_text} (
                  {Math.round(det.detection_confidence * 100)}%)
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
