"use client";

import { useState } from "react";
import { RefreshCw, Upload, Sparkles, Clock, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DetectionItem } from "@/lib/types";
import { DetectionCanvas } from "./DetectionCanvas";
import { PlateResultCard } from "./PlateResultCard";

interface DetectionResultsProps {
  detections: DetectionItem[];
  originalImageUrl?: string;
  processingTimeMs?: number;
  onRescan?: () => void;
  onUploadAnother?: () => void;
}

export function DetectionResults({
  detections,
  originalImageUrl,
  processingTimeMs,
  onRescan,
  onUploadAnother,
}: DetectionResultsProps) {
  const [selectedIndex, setSelectedIndex] = useState<number>(0);

  if (!detections || detections.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-navy-900/90 p-8 text-center space-y-4 max-w-lg mx-auto shadow-xl">
        <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-400 w-16 h-16 mx-auto flex items-center justify-center">
          <AlertTriangle className="w-8 h-8" />
        </div>
        <div className="space-y-1">
          <h3 className="text-lg font-bold text-white">No License Plates Detected</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Ensure the vehicle plate is clearly visible, well-lit, and unobstructed.
          </p>
        </div>
        <div className="flex gap-3 justify-center pt-2">
          {onRescan && (
            <Button variant="cyan" onClick={onRescan}>
              <RefreshCw className="w-4 h-4 mr-1.5" /> Rescan Camera
            </Button>
          )}
          {onUploadAnother && (
            <Button variant="outline" onClick={onUploadAnother}>
              <Upload className="w-4 h-4 mr-1.5" /> Upload Another
            </Button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 w-full max-w-5xl mx-auto">
      {/* Header Info Banner */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-2xl bg-navy-900/80 border border-slate-800/80 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">
              {detections.length === 1
                ? "1 Number Plate Detected"
                : `${detections.length} Number Plates Detected`}
            </h3>
            <p className="text-xs text-slate-400">
              High-resolution OCR bounding boxes extracted
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {processingTimeMs !== undefined && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-navy-950 border border-slate-800 text-xs font-mono text-slate-300">
              <Clock className="w-3.5 h-3.5 text-cyan-400" />
              <span>{processingTimeMs} ms</span>
            </div>
          )}

          {onRescan && (
            <Button variant="outline" size="sm" onClick={onRescan}>
              <RefreshCw className="w-3.5 h-3.5 mr-1" /> Rescan
            </Button>
          )}
          {onUploadAnother && (
            <Button variant="outline" size="sm" onClick={onUploadAnother}>
              <Upload className="w-3.5 h-3.5 mr-1" /> New Image
            </Button>
          )}
        </div>
      </div>

      {/* Main Analysis Visuals */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left / Top: Interactive Image Canvas with Bounding Boxes */}
        {originalImageUrl && (
          <div className="lg:col-span-7 space-y-2">
            <span className="text-xs font-mono text-slate-400 uppercase tracking-wider block px-1">
              Vehicle Image & Bounding Box Coordinates
            </span>
            <DetectionCanvas
              imageUrl={originalImageUrl}
              detections={detections}
              selectedIndex={selectedIndex}
              onSelectDetection={(idx) => setSelectedIndex(idx)}
            />
          </div>
        )}

        {/* Right / Bottom: List of Plate Result Cards */}
        <div
          className={`space-y-4 ${
            originalImageUrl ? "lg:col-span-5" : "w-full max-w-3xl mx-auto"
          }`}
        >
          <span className="text-xs font-mono text-slate-400 uppercase tracking-wider block px-1">
            Detected Plate Details
          </span>
          {detections.map((det, idx) => (
            <PlateResultCard
              key={det.id || idx}
              detection={det}
              index={idx}
              isSelected={selectedIndex === idx}
              onSelect={() => setSelectedIndex(idx)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
