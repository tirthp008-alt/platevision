"use client";

import { useEffect, useState } from "react";
import {
  Plus,
  Trash2,
  Play,
  Square,
  RefreshCw,
  Cctv,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  MapPin,
  Radio,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  CameraDeviceInfo,
  CameraStatus,
  GridCamera,
  SourceType,
} from "@/lib/grid-types";

const STATUS_STYLES: Record<CameraStatus, { text: string; dot: string; label: string }> = {
  ONLINE: { text: "text-emerald-400", dot: "bg-emerald-400", label: "ONLINE" },
  PROCESSING: { text: "text-cyan-400", dot: "bg-cyan-400", label: "PROCESSING" },
  CONNECTING: { text: "text-amber-400", dot: "bg-amber-400", label: "CONNECTING" },
  OFFLINE: { text: "text-slate-400", dot: "bg-slate-500", label: "OFFLINE" },
  ERROR: { text: "text-rose-400", dot: "bg-rose-500", label: "ERROR" },
};

export interface AddCameraDraft {
  name: string;
  sourceType: SourceType;
  sourceUri: string;
  deviceId: string;
  latitude: number | null;
  longitude: number | null;
}

interface CameraManagerPanelProps {
  cameras: GridCamera[];
  devices: CameraDeviceInfo[];
  devicesNote?: string;
  selectedCameraId: string | null;
  onSelectCamera: (id: string) => void;
  onStart: (id: string) => void;
  onStop: (id: string) => void;
  onDelete: (id: string) => void;
  onRefreshDevices: () => void;
  onRefresh: () => void;
  // Add-camera flow (map picking is controlled by the parent)
  addDraft: AddCameraDraft;
  onAddDraftChange: (patch: Partial<AddCameraDraft>) => void;
  onBeginPick: () => void;
  picking: boolean;
  onSaveCamera: () => void;
  onCancelAdd: () => void;
  saving: boolean;
  canWrite: boolean;
  busy: boolean;
  error?: string | null;
  onProbe?: () => void;
  probing?: boolean;
  probeResult?: { status: string; message: string } | null;
}

export function CameraManagerPanel(props: CameraManagerPanelProps) {
  const {
    cameras,
    devices,
    devicesNote,
    selectedCameraId,
    onSelectCamera,
    onStart,
    onStop,
    onDelete,
    onRefreshDevices,
    onRefresh,
    addDraft,
    onAddDraftChange,
    onBeginPick,
    picking,
    onSaveCamera,
    onCancelAdd,
    saving,
    canWrite,
    busy,
    error,
    onProbe,
    probing,
    probeResult,
  } = props;

  const [showAdd, setShowAdd] = useState(false);

  useEffect(() => {
    if (picking) setShowAdd(true);
  }, [picking]);

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cctv className="h-4 w-4 text-gov-saffron" />
          <h2 className="text-sm font-bold tracking-tight text-white">Camera Network</h2>
          <span className="rounded bg-gov-panel px-1.5 py-0.5 text-[10px] font-mono text-slate-300 border border-gov-border">
            {cameras.length}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon" onClick={onRefresh} disabled={busy} title="Refresh cameras">
            <RefreshCw className={cn("h-3.5 w-3.5", busy && "animate-spin")} />
          </Button>
          {canWrite && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowAdd((v) => !v)}
              className="h-7 px-2 text-[11px]"
            >
              <Plus className="h-3 w-3" /> Add Camera
            </Button>
          )}
        </div>
      </div>

      {/* Add-camera form */}
      {showAdd && canWrite && (
        <div className="space-y-2 rounded-lg border border-gov-border bg-gov-panel/60 p-3">
          <div className="text-[11px] font-bold uppercase tracking-wide text-gov-saffron">
            Register new camera
          </div>
          <input
            value={addDraft.name}
            onChange={(e) => onAddDraftChange({ name: e.target.value })}
            placeholder="Camera name (e.g. Camera 06 - Ring Road)"
            className="w-full rounded-md border border-gov-border bg-gov-navy px-2.5 py-1.5 text-xs text-white placeholder:text-slate-500 focus:border-gov-saffron focus:outline-none"
          />
          <div className="grid grid-cols-2 gap-2">
            <select
              value={addDraft.sourceType}
              onChange={(e) => onAddDraftChange({ sourceType: e.target.value as SourceType })}
              className="rounded-md border border-gov-border bg-gov-navy px-2 py-1.5 text-xs text-white focus:border-gov-saffron focus:outline-none"
            >
              <option value="device">Local webcam</option>
              <option value="rtsp">RTSP / HTTP stream</option>
              <option value="file">Video file</option>
              <option value="demo">Demo (synthetic)</option>
            </select>
            {addDraft.sourceType === "device" ? (
              <select
                value={addDraft.deviceId}
                onChange={(e) => onAddDraftChange({ deviceId: e.target.value })}
                className="rounded-md border border-gov-border bg-gov-navy px-2 py-1.5 text-xs text-white focus:border-gov-saffron focus:outline-none"
              >
                <option value="">Select device…</option>
                {devices.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
                {devices.length === 0 && <option value="device:0">Webcam 0 (default)</option>}
              </select>
            ) : (
              <input
                value={addDraft.sourceUri}
                onChange={(e) => onAddDraftChange({ sourceUri: e.target.value })}
                placeholder={
                  addDraft.sourceType === "demo"
                    ? "demo"
                    : addDraft.sourceType === "file"
                    ? "/path/to/video.mp4"
                    : "rtsp://user:pass@ip:554/stream"
                }
                className="rounded-md border border-gov-border bg-gov-navy px-2.5 py-1.5 text-xs text-white placeholder:text-slate-500 focus:border-gov-saffron focus:outline-none"
              />
            )}
          </div>

          <div className="flex items-center gap-2 text-[11px]">
            <Button
              variant={picking ? "emerald" : "secondary"}
              size="sm"
              onClick={onBeginPick}
              className="h-7 flex-1 px-2 text-[11px]"
            >
              <MapPin className="h-3 w-3" />
              {picking ? "Click map…" : "Pick location on map"}
            </Button>
            {onProbe && (
              <Button
                variant="outline"
                size="sm"
                onClick={onProbe}
                disabled={probing}
                className="h-7 px-2 text-[11px]"
              >
                {probing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Radio className="h-3 w-3" />}
                Test feed
              </Button>
            )}
          </div>

          <div className="flex items-center gap-2 font-mono text-[10px] text-slate-400">
            <span>
              Lat: {addDraft.latitude != null ? addDraft.latitude.toFixed(5) : "—"}
            </span>
            <span>
              Lon: {addDraft.longitude != null ? addDraft.longitude.toFixed(5) : "—"}
            </span>
          </div>

          {probeResult && (
            <div
              className={cn(
                "rounded-md border px-2 py-1 text-[11px]",
                probeResult.status === "ok"
                  ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                  : "border-rose-500/40 bg-rose-500/10 text-rose-300"
              )}
            >
              {probeResult.message}
            </div>
          )}

          <div className="flex gap-2">
            <Button
              size="sm"
              variant="cyan"
              className="h-7 flex-1 text-[11px]"
              onClick={onSaveCamera}
              disabled={saving || !addDraft.name || addDraft.latitude == null}
            >
              {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <CheckCircle2 className="h-3 w-3" />}
              Save Camera
            </Button>
            <Button
              size="sm"
              variant="ghost"
              className="h-7 text-[11px]"
              onClick={() => {
                setShowAdd(false);
                onCancelAdd();
              }}
            >
              Cancel
            </Button>
          </div>

          <div className="flex items-center justify-between border-t border-gov-border/60 pt-2">
            <span className="text-[10px] text-slate-500">
              {devices.length} local device(s) detected
            </span>
            <button
              onClick={onRefreshDevices}
              className="text-[10px] font-semibold text-gov-saffron hover:underline"
            >
              Rescan devices
            </button>
          </div>
          {devicesNote && <p className="text-[10px] leading-snug text-slate-500">{devicesNote}</p>}
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-rose-500/40 bg-rose-500/10 px-2.5 py-2 text-[11px] text-rose-300">
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Camera list */}
      <div className="flex-1 space-y-2 overflow-y-auto pr-1">
        {cameras.length === 0 && (
          <div className="rounded-lg border border-dashed border-gov-border p-4 text-center text-[11px] text-slate-500">
            No cameras registered. Add one, or load the synthetic demo corridor.
          </div>
        )}
        {cameras.map((cam) => {
          const style = STATUS_STYLES[cam.status] || STATUS_STYLES.OFFLINE;
          const selected = cam.id === selectedCameraId;
          return (
            <div
              key={cam.id}
              onClick={() => onSelectCamera(cam.id)}
              className={cn(
                "cursor-pointer rounded-lg border p-2.5 transition-colors",
                selected
                  ? "border-gov-saffron/70 bg-gov-saffron/5"
                  : "border-gov-border bg-gov-panel/40 hover:border-gov-border/80 hover:bg-gov-panel/70"
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className={cn("h-2 w-2 shrink-0 rounded-full", style.dot, cam.status === "PROCESSING" && "animate-pulse")} />
                    <span className="truncate text-xs font-semibold text-white">{cam.name}</span>
                  </div>
                  <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 font-mono text-[10px] text-slate-400">
                    <span className={style.text}>{style.label}</span>
                    <span>· {cam.source_type}</span>
                    {cam.is_demo && <span className="text-gov-saffron">· DEMO</span>}
                    <span>· {cam.sighting_count ?? 0} sightings</span>
                  </div>
                  {cam.last_error && (
                    <div className="mt-0.5 truncate text-[10px] text-rose-400">{cam.last_error}</div>
                  )}
                </div>
                {canWrite && (
                  <div className="flex shrink-0 items-center gap-1">
                    {cam.runtime?.is_running ? (
                      <Button
                        variant="destructive"
                        size="icon"
                        className="h-6 w-6"
                        title="Stop processing"
                        onClick={(e) => {
                          e.stopPropagation();
                          onStop(cam.id);
                        }}
                      >
                        <Square className="h-3 w-3" />
                      </Button>
                    ) : (
                      <Button
                        variant="emerald"
                        size="icon"
                        className="h-6 w-6"
                        title="Start processing"
                        onClick={(e) => {
                          e.stopPropagation();
                          onStart(cam.id);
                        }}
                      >
                        <Play className="h-3 w-3" />
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 text-slate-400 hover:text-rose-400"
                      title="Delete camera"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete(cam.id);
                      }}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
