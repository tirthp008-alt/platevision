"use client";

import { useEffect, useState } from "react";
import { Cctv, Loader2, Video, Layers } from "lucide-react";
import { cn } from "@/lib/utils";
import { cameraFrameUrl, cameraStatus } from "@/lib/grid-api";
import { CameraRuntime, GridCamera } from "@/lib/grid-types";

function LiveTile({ camera, onReveal }: { camera: GridCamera; onReveal?: () => void }) {
  const [bust, setBust] = useState(() => Date.now());
  const [error, setError] = useState(false);
  const [runtime, setRuntime] = useState<CameraRuntime | null>(camera.runtime ?? null);
  const running = Boolean(camera.runtime?.is_running);

  useEffect(() => {
    if (!running) return;
    // The snapshot endpoint already renders vehicle + plate overlays server-side
    // (bounding boxes, track IDs, recognized plate text, OCR confidence).
    const imgTimer = setInterval(() => setBust(Date.now()), 900);
    const statusTimer = setInterval(async () => {
      try {
        setRuntime(await cameraStatus(camera.id));
      } catch {
        /* keep last known runtime */
      }
    }, 2500);
    return () => {
      clearInterval(imgTimer);
      clearInterval(statusTimer);
    };
  }, [running, camera.id]);

  const recent = runtime?.recent_sightings || [];

  return (
    <div className="overflow-hidden rounded-xl border border-gov-border bg-gov-navy">
      <div className="relative aspect-video w-full bg-black">
        {running ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={bust}
            src={cameraFrameUrl(camera.id, bust)}
            alt={`${camera.name} live annotated feed`}
            className="h-full w-full object-contain"
            onError={() => setError(true)}
            onLoad={() => setError(false)}
          />
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-slate-500">
            <Video className="h-6 w-6" />
            <span className="text-[11px]">Feed not running — start the camera to process</span>
          </div>
        )}
        {running && error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/70 text-[11px] text-slate-300">
            <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" /> Waiting for frames…
          </div>
        )}
        <div className="absolute left-2 top-2 flex items-center gap-1.5 rounded-md bg-black/60 px-2 py-1">
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              running ? "animate-pulse bg-rose-500" : "bg-slate-500"
            )}
          />
          <span className="font-mono text-[10px] font-bold text-white">
            {running ? "LIVE" : "IDLE"}
          </span>
        </div>
        {camera.is_demo && (
          <div className="absolute right-2 top-2 rounded-md bg-gov-saffron/90 px-1.5 py-0.5 font-mono text-[9px] font-bold text-gov-navy">
            SYNTHETIC DEMO
          </div>
        )}
      </div>

      <div className="space-y-1.5 p-2.5">
        <div className="flex items-center justify-between gap-2">
          <button
            onClick={onReveal}
            className="flex min-w-0 items-center gap-1.5 text-left hover:text-gov-saffron"
          >
            <Cctv className="h-3 w-3 shrink-0 text-gov-saffron" />
            <span className="truncate text-[11px] font-semibold text-white">{camera.name}</span>
          </button>
          <span className="shrink-0 font-mono text-[10px] text-slate-400">
            {runtime?.fps ? `${runtime.fps.toFixed(1)} fps` : "—"}
          </span>
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[10px] text-slate-400">
          <span>tracks: {runtime?.tracked_vehicles ?? 0}</span>
          <span>sightings: {runtime?.sighting_count ?? camera.sighting_count ?? 0}</span>
          <span>status: {runtime?.status ?? camera.status}</span>
        </div>
        {recent.length > 0 && (
          <div className="space-y-0.5 border-t border-gov-border/60 pt-1.5">
            <div className="flex items-center gap-1 text-[9px] uppercase tracking-wide text-slate-500">
              <Layers className="h-2.5 w-2.5" /> recent plate reads
            </div>
            {recent.slice(0, 3).map((s) => (
              <div key={s.sighting_id} className="flex items-center justify-between font-mono text-[10px]">
                <span className="text-slate-200">{s.plate || "—"}</span>
                <span className="text-slate-500">
                  #{s.track_id} · {(s.plate_confidence * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

interface LiveCameraGridProps {
  cameras: GridCamera[];
  selectedCameraId: string | null;
  onSelectCamera: (id: string) => void;
}

export function LiveCameraGrid({ cameras, selectedCameraId, onSelectCamera }: LiveCameraGridProps) {
  const active = cameras.filter((c) => c.runtime?.is_running || c.status === "PROCESSING");
  const shown = active.length > 0 ? active : cameras.slice(0, 4);

  if (shown.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-gov-border p-6 text-center text-xs text-slate-500">
        No camera feeds available. Register and start a camera to view live annotated
        detection overlays.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {shown.map((cam) => (
        <LiveTile
          key={cam.id}
          camera={cam}
          onReveal={() => onSelectCamera(cam.id)}
        />
      ))}
    </div>
  );
}
