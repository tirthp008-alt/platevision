"use client";

import { useState, useEffect } from "react";
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
  Sparkles,
  CheckCheck,
  ArrowRight,
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
  isSupervisionQueue?: boolean;
  onSelect?: () => void;
  onSaveCorrection?: (newRaw: string, newFormatted: string) => void;
}

export function PlateResultCard({
  detection,
  index,
  isSelected,
  isSupervisionQueue = false,
  onSelect,
  onSaveCorrection,
}: PlateResultCardProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editedText, setEditedText] = useState(
    detection.formatted_text || detection.normalized_text
  );
  const [copied, setCopied] = useState(false);
  const [showInspector, setShowInspector] = useState(false);

  useEffect(() => {
    setEditedText(detection.formatted_text || detection.normalized_text);
  }, [detection]);

  const statusDetails = getFormatStatusDetails(detection.format_status);

  // Generate intelligent suggestions (e.g. O -> D in series or 0 -> O/D)
  const currentStr = editedText.replace(/\s+/g, "").toUpperCase();
  const suggestions: string[] = [];

  if (currentStr.includes("O")) {
    const dSuggestion = currentStr.replace(/O/g, "D");
    if (dSuggestion !== currentStr) suggestions.push(dSuggestion);
  }
  if (currentStr.includes("0") && currentStr.length >= 8) {
    // 0 in series position 4
    if (currentStr.length === 10 && currentStr[4] === "0") {
      suggestions.push(currentStr.slice(0, 4) + "D" + currentStr.slice(5));
    }
  }

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

  const handleConfirmCorrection = () => {
    if (onSaveCorrection) {
      onSaveCorrection(editedText.replace(/\s+/g, ""), editedText);
    }
    setIsEditing(false);
  };

  const handleApplySuggestion = (sug: string) => {
    setEditedText(sug);
    if (onSaveCorrection) {
      onSaveCorrection(sug.replace(/\s+/g, ""), sug);
    }
  };

  return (
    <div
      onClick={onSelect}
      className={`rounded-2xl border p-4 sm:p-5 shadow-xl backdrop-blur-md transition-all duration-200 cursor-pointer space-y-4 ${
        isSupervisionQueue
          ? isSelected
            ? "bg-gov-panel border-amber-400 ring-2 ring-amber-400/50 shadow-amber-500/15"
            : "bg-gov-panel/90 border-amber-500/40 hover:border-amber-400"
          : isSelected
          ? "bg-gov-panel border-gov-saffron ring-2 ring-gov-saffron/40 shadow-gov-saffron/10"
          : "bg-gov-deepnavy/90 border-gov-border hover:border-slate-700"
      }`}
    >
      {/* Top Header: Plate # & Validation Status Badge */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span
            className={`flex items-center justify-center w-6 h-6 rounded-full text-xs font-mono font-bold border ${
              isSupervisionQueue
                ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                : "bg-gov-saffron/20 text-gov-saffron border-gov-saffron/40"
            }`}
          >
            #{index + 1}
          </span>
          <span className="text-xs font-semibold text-slate-200 font-mono">
            {isSupervisionQueue ? `Candidate #${index + 1} (Review Required)` : `Plate #${index + 1}`}
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
        <div className="sm:col-span-5 bg-black rounded-xl p-1.5 border border-gov-border flex items-center justify-center overflow-hidden max-h-28 shadow-inner">
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
                if (isEditing) handleConfirmCorrection();
                else setIsEditing(true);
              }}
              className={`text-xs flex items-center gap-1 font-mono font-bold ${
                isSupervisionQueue ? "text-amber-400 hover:text-amber-300" : "text-gov-saffron hover:text-orange-400"
              }`}
            >
              {isEditing ? <Save className="w-3.5 h-3.5" /> : <Edit2 className="w-3.5 h-3.5" />}
              <span>{isEditing ? "Save & Verify" : "Edit / Fix"}</span>
            </button>
          </div>

          {isEditing ? (
            <div className="space-y-2" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={editedText}
                  onChange={(e) => setEditedText(e.target.value.toUpperCase())}
                  placeholder="e.g. GJ 27 DS 4837"
                  className="w-full bg-gov-navy border border-gov-saffron rounded-xl px-3 py-2 text-base font-mono font-bold text-white tracking-widest uppercase focus:outline-none focus:ring-2 focus:ring-gov-saffron"
                />
                <button
                  type="button"
                  onClick={handleConfirmCorrection}
                  className="px-3 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-mono text-xs font-bold shrink-0 flex items-center gap-1"
                >
                  <CheckCheck className="w-3.5 h-3.5" /> Confirm
                </button>
              </div>
            </div>
          ) : (
            <div className="p-2.5 rounded-xl bg-gov-navy border border-gov-border flex items-center justify-between">
              <span className="text-xl sm:text-2xl font-black font-mono tracking-widest text-white">
                {editedText || "UNREADABLE"}
              </span>
            </div>
          )}

          {/* Quick Suggestions for O vs D or format fix */}
          {suggestions.length > 0 && isSupervisionQueue && !isEditing && (
            <div className="pt-1" onClick={(e) => e.stopPropagation()}>
              <div className="flex flex-wrap items-center gap-1.5 text-[10px] font-mono">
                <span className="text-amber-400 flex items-center gap-1 font-bold">
                  <Sparkles className="w-3 h-3" /> Suggestion:
                </span>
                {suggestions.map((sug, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => handleApplySuggestion(sug)}
                    className="px-2 py-0.5 rounded bg-amber-500/20 border border-amber-500/40 text-amber-200 hover:bg-amber-500/30 hover:border-amber-400 transition-colors font-bold flex items-center gap-1"
                  >
                    <span>{sug}</span>
                    <ArrowRight className="w-2.5 h-2.5 text-amber-400" />
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Confidence Badges */}
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
        className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-gov-border"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={() => setShowInspector(!showInspector)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium transition-colors ${
            showInspector
              ? "bg-gov-saffron/20 text-gov-saffron border border-gov-saffron/40"
              : "bg-gov-navy text-slate-400 hover:text-slate-200 border border-gov-border"
          }`}
        >
          <SlidersHorizontal className="w-3.5 h-3.5" />
          <span>{showInspector ? "Hide Inspector" : "Inspect Filters"}</span>
        </button>

        <div className="flex items-center gap-2">
          {/* Copy Button */}
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-gov-navy text-slate-300 hover:text-white border border-gov-border transition-colors"
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
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-gov-saffron/15 text-gov-saffron hover:text-white hover:bg-gov-saffron/25 border border-gov-saffron/30 transition-colors"
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
