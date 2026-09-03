"use client";

import { useState } from "react";
import {
  History,
  Trash2,
  Download,
  Copy,
  Check,
  Camera,
  Upload,
  ShieldCheck,
  AlertCircle,
  Clock,
  HardDrive,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useLocalHistory } from "@/hooks/useLocalHistory";
import { downloadCrop } from "@/lib/api";
import { getFormatStatusDetails } from "@/lib/utils";

export function HistoryView() {
  const {
    history,
    storeCropsLocally,
    updateCropStorageOpt,
    deleteRecord,
    clearAllHistory,
    exportHistoryJSON,
  } = useLocalHistory();

  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleCopy = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (e) {}
  };

  const handleDownloadCrop = async (recordId: string, base64OrUrl: string, text: string) => {
    const filename = `history_plate_${text.replace(/\s+/g, "_")}.jpg`;
    await downloadCrop(base64OrUrl, filename);
  };

  return (
    <div className="space-y-6 w-full max-w-4xl mx-auto pb-24">
      {/* Top Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-2xl bg-navy-900/90 border border-slate-800 shadow-xl backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <History className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white tracking-tight">Local Scan History</h2>
            <p className="text-xs text-slate-400">
              {history.length === 0
                ? "No scans recorded yet"
                : `${history.length} scans stored in browser localStorage`}
            </p>
          </div>
        </div>

        {history.length > 0 && (
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={exportHistoryJSON}>
              <Download className="w-3.5 h-3.5 mr-1" /> Export JSON
            </Button>
            <Button variant="destructive" size="sm" onClick={clearAllHistory}>
              <Trash2 className="w-3.5 h-3.5 mr-1" /> Clear All
            </Button>
          </div>
        )}
      </div>

      {/* Settings Row: Storage Preference */}
      <div className="p-4 rounded-2xl bg-navy-950/60 border border-slate-800 flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <HardDrive className="w-4 h-4 text-cyan-400 shrink-0" />
          <span className="text-xs text-slate-300">
            Save thumbnail plate crops inside local browser storage
          </span>
        </div>
        <input
          type="checkbox"
          checked={storeCropsLocally}
          onChange={(e) => updateCropStorageOpt(e.target.checked)}
          className="w-4 h-4 rounded border-slate-700 text-cyan-500 focus:ring-cyan-400 bg-navy-900 cursor-pointer"
        />
      </div>

      {/* Empty State */}
      {history.length === 0 && (
        <div className="p-12 text-center rounded-2xl border border-slate-800 bg-navy-900/60 space-y-3">
          <History className="w-12 h-12 text-slate-600 mx-auto" />
          <h3 className="text-base font-bold text-white">No Scan History</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            Vehicle plates scanned through live camera or image uploads will appear here securely on your device.
          </p>
        </div>
      )}

      {/* History List */}
      {history.length > 0 && (
        <div className="space-y-3">
          {history.map((record) => {
            const det = record.detection;
            const statusDetails = getFormatStatusDetails(det.format_status);
            const dateStr = new Date(record.timestamp).toLocaleString(undefined, {
              dateStyle: "medium",
              timeStyle: "short",
            });

            return (
              <div
                key={record.id}
                className="p-4 rounded-2xl border border-slate-800/80 bg-navy-900/80 hover:border-slate-700 transition-all flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
              >
                {/* Left: Thumbnail & Details */}
                <div className="flex items-center gap-3.5">
                  {record.crop_thumbnail ? (
                    <div className="w-20 h-10 bg-black rounded-lg border border-slate-700 overflow-hidden flex items-center justify-center shrink-0">
                      <img
                        src={record.crop_thumbnail}
                        alt="Crop"
                        className="max-h-full max-w-full object-contain"
                      />
                    </div>
                  ) : (
                    <div className="w-10 h-10 rounded-xl bg-navy-950 border border-slate-800 flex items-center justify-center text-cyan-400 shrink-0">
                      {record.source === "camera" ? (
                        <Camera className="w-5 h-5" />
                      ) : (
                        <Upload className="w-5 h-5" />
                      )}
                    </div>
                  )}

                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-lg font-black font-mono tracking-wider text-white">
                        {det.formatted_text || det.normalized_text}
                      </span>
                      <span
                        className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${statusDetails.colorClass}`}
                      >
                        {statusDetails.label}
                      </span>
                    </div>

                    <div className="flex items-center gap-3 text-xs text-slate-400 mt-0.5 font-mono">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3 text-slate-500" />
                        {dateStr}
                      </span>
                      <span>•</span>
                      <span className="capitalize">{record.source}</span>
                      <span>•</span>
                      <span>Conf: {Math.round(det.detection_confidence * 100)}%</span>
                    </div>
                  </div>
                </div>

                {/* Right: Actions */}
                <div className="flex items-center gap-2 self-end sm:self-auto">
                  {/* Copy */}
                  <button
                    onClick={() =>
                      handleCopy(record.id, det.formatted_text || det.normalized_text)
                    }
                    title="Copy plate number"
                    className="p-2 rounded-xl bg-navy-950 border border-slate-800 text-slate-300 hover:text-white transition-colors"
                  >
                    {copiedId === record.id ? (
                      <Check className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </button>

                  {/* Download Crop if available */}
                  {record.crop_thumbnail && (
                    <button
                      onClick={() =>
                        handleDownloadCrop(
                          record.id,
                          record.crop_thumbnail!,
                          det.formatted_text || det.normalized_text
                        )
                      }
                      title="Download crop image"
                      className="p-2 rounded-xl bg-navy-950 border border-slate-800 text-slate-300 hover:text-white transition-colors"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                  )}

                  {/* Delete */}
                  <button
                    onClick={() => deleteRecord(record.id)}
                    title="Delete record"
                    className="p-2 rounded-xl bg-navy-950 border border-slate-800 text-slate-400 hover:text-rose-400 hover:border-rose-500 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
