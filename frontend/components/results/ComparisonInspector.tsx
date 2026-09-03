"use client";

import { useState } from "react";
import { Sliders, Eye, Layers, Sparkles } from "lucide-react";
import { DetectionItem } from "@/lib/types";

interface ComparisonInspectorProps {
  detection: DetectionItem;
}

export function ComparisonInspector({ detection }: ComparisonInspectorProps) {
  const [activeTab, setActiveTab] = useState<"crop" | "preprocess" | "details">("crop");

  return (
    <div className="rounded-xl border border-slate-800 bg-navy-950/70 p-3.5 space-y-3">
      {/* Sub Tabs */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
        <span className="text-xs font-mono font-semibold text-cyan-400 flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5" /> Pipeline Inspector
        </span>
        <div className="flex items-center gap-1 bg-navy-900 p-0.5 rounded-lg border border-slate-800">
          <button
            onClick={() => setActiveTab("crop")}
            className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
              activeTab === "crop" ? "bg-cyan-500 text-navy-950 font-bold" : "text-slate-400 hover:text-white"
            }`}
          >
            Raw Crop
          </button>
          <button
            onClick={() => setActiveTab("preprocess")}
            className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
              activeTab === "preprocess" ? "bg-cyan-500 text-navy-950 font-bold" : "text-slate-400 hover:text-white"
            }`}
          >
            OCR Filtered
          </button>
          <button
            onClick={() => setActiveTab("details")}
            className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
              activeTab === "details" ? "bg-cyan-500 text-navy-950 font-bold" : "text-slate-400 hover:text-white"
            }`}
          >
            Coordinates
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <div className="min-h-[100px] flex items-center justify-center">
        {activeTab === "crop" && (
          <div className="flex flex-col items-center gap-1.5 w-full">
            <div className="p-1 rounded-lg border border-slate-700 bg-black max-w-xs w-full overflow-hidden flex items-center justify-center">
              <img
                src={detection.crop_base64 || detection.crop_url}
                alt="Raw Cropped Plate"
                className="max-h-24 object-contain rounded"
              />
            </div>
            <span className="text-[10px] text-slate-400 font-mono">Original High-Resolution Crop</span>
          </div>
        )}

        {activeTab === "preprocess" && (
          <div className="flex flex-col items-center gap-1.5 w-full">
            <div className="p-1 rounded-lg border border-cyan-500/40 bg-black max-w-xs w-full overflow-hidden flex items-center justify-center">
              {detection.preprocessed_base64 ? (
                <img
                  src={detection.preprocessed_base64}
                  alt="Preprocessed OCR Image"
                  className="max-h-24 object-contain rounded"
                />
              ) : (
                <div className="text-xs text-slate-400 font-mono py-4">Adaptive Thresholding Applied</div>
              )}
            </div>
            <span className="text-[10px] text-cyan-300 font-mono flex items-center gap-1">
              <Sparkles className="w-3 h-3" /> Enhanced Grayscale + Adaptive Thresholding
            </span>
          </div>
        )}

        {activeTab === "details" && (
          <div className="w-full grid grid-cols-2 gap-2 text-xs font-mono text-slate-300">
            <div className="p-2 rounded bg-navy-900/90 border border-slate-800">
              <span className="text-slate-500 text-[10px] block">Pixel Bounding Box</span>
              <span>
                X: {detection.bounding_box.x}, Y: {detection.bounding_box.y}
              </span>
              <span className="block text-[11px] text-slate-400">
                {detection.bounding_box.width} x {detection.bounding_box.height} px
              </span>
            </div>
            <div className="p-2 rounded bg-navy-900/90 border border-slate-800">
              <span className="text-slate-500 text-[10px] block">Normalized Relative Box</span>
              <span>
                X: {detection.normalized_box.x}, Y: {detection.normalized_box.y}
              </span>
              <span className="block text-[11px] text-slate-400">
                W: {detection.normalized_box.width}, H: {detection.normalized_box.height}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
