"use client";

import { useState } from "react";
import {
  RefreshCw,
  Upload,
  Sparkles,
  Clock,
  AlertTriangle,
  ShieldCheck,
  UserCheck,
  CheckCircle2,
  HelpCircle,
  FileQuestion,
} from "lucide-react";
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
  detections: initialDetections,
  originalImageUrl,
  processingTimeMs,
  onRescan,
  onUploadAnother,
}: DetectionResultsProps) {
  const [detections, setDetections] = useState<DetectionItem[]>(initialDetections);
  const [selectedIndex, setSelectedIndex] = useState<number>(0);

  // Update detection text and promote to verified status
  const handleUpdateDetection = (index: number, newText: string, newFormatted: string) => {
    setDetections((prev) => {
      const updated = [...prev];
      updated[index] = {
        ...updated[index],
        raw_text: newText,
        normalized_text: newText.replace(/\s+/g, ""),
        formatted_text: newFormatted || newText,
        format_status: "valid", // Promoted by human supervisor
      };
      return updated;
    });
  };

  if (!detections || detections.length === 0) {
    return (
      <div className="rounded-2xl border border-gov-border bg-gov-deepnavy/90 p-8 text-center space-y-4 max-w-lg mx-auto shadow-xl">
        <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-400 w-16 h-16 mx-auto flex items-center justify-center">
          <AlertTriangle className="w-8 h-8" />
        </div>
        <div className="space-y-1">
          <h3 className="text-lg font-bold text-white font-mono">No License Plates Detected</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Ensure the vehicle plate is clearly visible, well-lit, and unobstructed.
          </p>
        </div>
        <div className="flex gap-3 justify-center pt-2">
          {onRescan && (
            <Button variant="default" onClick={onRescan} className="bg-gov-saffron text-gov-navy font-bold text-xs">
              <RefreshCw className="w-4 h-4 mr-1.5" /> Rescan Camera
            </Button>
          )}
          {onUploadAnother && (
            <Button variant="outline" onClick={onUploadAnother} className="border-gov-border text-slate-300 text-xs">
              <Upload className="w-4 h-4 mr-1.5" /> Upload Another
            </Button>
          )}
        </div>
      </div>
    );
  }

  // Separate verified vs uncertain (human supervision) detections
  const verifiedDetections = detections
    .map((det, originalIndex) => ({ det, originalIndex }))
    .filter(({ det }) => det.format_status === "valid");

  const supervisionDetections = detections
    .map((det, originalIndex) => ({ det, originalIndex }))
    .filter(({ det }) => det.format_status !== "valid");

  return (
    <div className="space-y-6 w-full max-w-6xl mx-auto font-sans pb-16">
      {/* Header Info Banner */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-2xl bg-gov-deepnavy/90 border border-gov-border backdrop-blur-md shadow-xl">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-gov-saffron/15 text-gov-saffron border border-gov-saffron/30">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-bold text-white tracking-tight font-mono">
                {detections.length === 1
                  ? "1 License Plate Detected"
                  : `${detections.length} License Plates Detected`}
              </h3>
              {supervisionDetections.length > 0 && (
                <span className="px-2 py-0.5 rounded-full bg-amber-500/20 border border-amber-500/40 text-amber-300 text-[10px] font-mono font-bold animate-pulse">
                  {supervisionDetections.length} Requires Human Review
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400">
              High-resolution OCR crops & automated RTO format classification
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {processingTimeMs !== undefined && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-gov-panel border border-gov-border text-xs font-mono text-slate-300">
              <Clock className="w-3.5 h-3.5 text-gov-saffron" />
              <span>{processingTimeMs} ms</span>
            </div>
          )}

          {onRescan && (
            <Button variant="outline" size="sm" onClick={onRescan} className="border-gov-border text-xs text-slate-300">
              <RefreshCw className="w-3.5 h-3.5 mr-1 text-gov-saffron" /> Rescan
            </Button>
          )}
          {onUploadAnother && (
            <Button variant="outline" size="sm" onClick={onUploadAnother} className="border-gov-border text-xs text-slate-300">
              <Upload className="w-3.5 h-3.5 mr-1 text-gov-saffron" /> New Image
            </Button>
          )}
        </div>
      </div>

      {/* Main Grid: Interactive Canvas on Left, Structured Results on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Interactive Image Canvas with Bounding Boxes */}
        {originalImageUrl && (
          <div className="lg:col-span-7 space-y-2">
            <span className="text-xs font-mono text-slate-400 uppercase tracking-wider block px-1">
              Vehicle Image & Spatial Coordinate Overlay
            </span>
            <DetectionCanvas
              imageUrl={originalImageUrl}
              detections={detections}
              selectedIndex={selectedIndex}
              onSelectDetection={(idx) => setSelectedIndex(idx)}
            />
          </div>
        )}

        {/* Right: Plate Results & Supervision Queue */}
        <div
          className={`space-y-6 ${
            originalImageUrl ? "lg:col-span-5" : "w-full max-w-3xl mx-auto"
          }`}
        >
          {/* Section 1: Human Supervision Queue (If any plates have non-standard/ambiguous format) */}
          {supervisionDetections.length > 0 && (
            <div className="space-y-3 p-4 rounded-2xl bg-amber-950/20 border-2 border-amber-500/50 shadow-xl">
              <div className="flex items-center gap-2 text-amber-300">
                <UserCheck className="w-5 h-5 text-amber-400 shrink-0" />
                <div>
                  <h4 className="text-xs font-bold font-mono uppercase tracking-wider text-amber-300">
                    Human Supervision & Review Queue ({supervisionDetections.length})
                  </h4>
                  <p className="text-[11px] text-amber-200/70">
                    Detected format is uncertain or ambiguous (e.g. O vs D). Please verify and confirm with the crop.
                  </p>
                </div>
              </div>

              <div className="space-y-3 pt-1">
                {supervisionDetections.map(({ det, originalIndex }) => (
                  <PlateResultCard
                    key={det.id || originalIndex}
                    detection={det}
                    index={originalIndex}
                    isSelected={selectedIndex === originalIndex}
                    isSupervisionQueue={true}
                    onSelect={() => setSelectedIndex(originalIndex)}
                    onSaveCorrection={(newTxt, newFmt) =>
                      handleUpdateDetection(originalIndex, newTxt, newFmt)
                    }
                  />
                ))}
              </div>
            </div>
          )}

          {/* Section 2: Standard Format Verified Plates */}
          <div className="space-y-3">
            <div className="flex items-center justify-between px-1">
              <span className="text-xs font-mono text-slate-300 uppercase tracking-wider flex items-center gap-1.5 font-bold">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                Verified Registration Plates ({verifiedDetections.length})
              </span>
            </div>

            {verifiedDetections.length === 0 && supervisionDetections.length > 0 && (
              <p className="text-xs text-slate-500 italic p-3 rounded-xl bg-gov-panel border border-gov-border">
                All detected plates currently pending human supervision review above.
              </p>
            )}

            {verifiedDetections.map(({ det, originalIndex }) => (
              <PlateResultCard
                key={det.id || originalIndex}
                detection={det}
                index={originalIndex}
                isSelected={selectedIndex === originalIndex}
                isSupervisionQueue={false}
                onSelect={() => setSelectedIndex(originalIndex)}
                onSaveCorrection={(newTxt, newFmt) =>
                  handleUpdateDetection(originalIndex, newTxt, newFmt)
                }
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
