"use client";

import { useState } from "react";
import {
  Copy,
  Check,
  Download,
  Edit2,
  Save,
  RotateCcw,
  SlidersHorizontal,
  ShieldCheck,
  AlertCircle,
  HelpCircle,
} from "lucide-react";
import { DetectionItem } from "@/lib/types";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { ComparisonInspector } from "./ComparisonInspector";
import { downloadCrop } from "@/lib/api";
import { getFormatStatusDetails } from "@/lib/utils";

interface PlateResultCardProps {
  detection: DetectionItem;
  index: number;
  isSelected?: boolean;
  onSelect?: () => void;
}

export function PlateResultCard({
  detection,
  index,
  isSelected,
  onSelect,
}: PlateResultCardProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editedText, setEditedText] = useState(
    detection.formatted_text || detection.normalized_text
  );
  const [copied, setCopied] = useState(false);
  const [showInspector, setShowInspector] = useState(false);

  const statusDetails = getFormatStatusDetails(detection.format_status);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(editedText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.warn("Failed to copy:", e);
    }
  };

  const handleDownload = async () => {
    const filename = `plate_${editedText.replace(/\s+/g, "_")}_${Date.now()}.jpg`;
    await downloadCrop(detection.crop_base64 || detection.crop_url, filename);
  };

  const handleReset = () => {
    setEditedText(detection.formatted_text || detection.normalized_text);
    setIsEditing(false);
  };

  return (
    <div
      onClick={onSelect}
      className={`rounded-2xl border bg-navy-900/90 p-4 sm:p-5 shadow-xl backdrop-blur-md transition-all duration-200 cursor-pointer space-y-4 ${
        isSelected
          ? "border-cyan-400 ring-2 ring-cyan-400/40 shadow-cyan-500/10"
          : "border-slate-800/90 hover:border-slate-700"
      }`}
    >
      {/* Top Header: Plate # & Validation Status Badge */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="flex items-center justify-center w-6 h-6 rounded-full bg-cyan-500/20 text-cyan-400 text-xs font-mono font-bold border border-cyan-500/40">
            #{index + 1}
          </span>
          <span className="text-xs font-semibold text-slate-300 font-mono">
            Plate #{index + 1}
          </span>
        </div>

        {/* Validation Status Badge */}
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${statusDetails.colorClass}`}
        >
          {detection.format_status === "valid" ? (
            <ShieldCheck className="w-3.5 h-3.5" />
          ) : detection.format_status === "possible" ? (
            <AlertCircle className="w-3.5 h-3.5" />
          ) : (
            <HelpCircle className="w-3.5 h-3.5" />
          )}
          <span>{statusDetails.label}</span>
        </div>
      </div>

      {/* Center Grid: Plate Crop Preview & Recognized Plate Number */}
      <div className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-center">
        {/* Cropped Image */}
        <div className="sm:col-span-5 bg-black rounded-xl p-1.5 border border-slate-800 flex items-center justify-center overflow-hidden max-h-28">
          <img
            src={detection.crop_base64 || detection.crop_url}
            alt={`Cropped Plate #${index + 1}`}
            className="max-h-24 w-auto object-contain rounded"
          />
        </div>

        {/* OCR Result & Edit Box */}
        <div className="sm:col-span-7 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
              Registration Number
            </span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setIsEditing(!isEditing);
              }}
              className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-mono"
            >
              {isEditing ? <Save className="w-3.5 h-3.5" /> : <Edit2 className="w-3.5 h-3.5" />}
              <span>{isEditing ? "Done" : "Edit"}</span>
            </button>
          </div>

          {isEditing ? (
            <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
              <input
                type="text"
                value={editedText}
                onChange={(e) => setEditedText(e.target.value.toUpperCase())}
                className="w-full bg-navy-950 border border-cyan-500 rounded-xl px-3 py-2 text-base font-mono font-bold text-white tracking-widest uppercase focus:outline-none focus:ring-2 focus:ring-cyan-400"
              />
              <button
                onClick={handleReset}
                title="Reset to raw OCR"
                className="p-2 rounded-lg bg-slate-800 text-slate-300 hover:text-white"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <div className="p-2.5 rounded-xl bg-navy-950 border border-slate-800 flex items-center justify-between">
              <span className="text-xl sm:text-2xl font-black font-mono tracking-widest text-white">
                {editedText}
              </span>
            </div>
          )}

          {/* Separate Confidence Badges */}
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <ConfidenceBadge
              label="Detection"
              confidence={detection.detection_confidence}
            />
            <ConfidenceBadge
              label="OCR"
              confidence={detection.ocr_confidence}
            />
          </div>
        </div>
      </div>

      {/* Action Buttons Row */}
      <div
        className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-800/80"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={() => setShowInspector(!showInspector)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium transition-colors ${
            showInspector
              ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
              : "bg-navy-950 text-slate-400 hover:text-slate-200 border border-slate-800"
          }`}
        >
          <SlidersHorizontal className="w-3.5 h-3.5" />
          <span>{showInspector ? "Hide Inspector" : "Inspect Filters"}</span>
        </button>

        <div className="flex items-center gap-2">
          {/* Copy Button */}
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-navy-950 text-slate-300 hover:text-white border border-slate-800 transition-colors"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy Text</span>
              </>
            )}
          </button>

          {/* Download Crop Button */}
          <button
            onClick={handleDownload}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-electric-500/20 text-blue-300 hover:text-white hover:bg-electric-500/30 border border-blue-500/40 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download Crop</span>
          </button>
        </div>
      </div>

      {/* Comparison Inspector Drawer */}
      {showInspector && <ComparisonInspector detection={detection} />}
    </div>
  );
}
