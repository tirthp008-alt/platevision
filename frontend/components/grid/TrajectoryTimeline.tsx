"use client";

import { ArrowRight, Clock, MapPin, Car, ShieldQuestion, CheckCircle2 } from "lucide-react";
import { cn, formatPercentage } from "@/lib/utils";
import { Trajectory } from "@/lib/grid-types";
import { PlateChip, fmtTime } from "./VehicleSearchPanel";

interface TrajectoryTimelineProps {
  trajectory: Trajectory | null;
  onCenterMap: (lat: number, lng: number) => void;
  canViewFullPlate: boolean;
}

function confidenceColor(c: number): string {
  if (c >= 0.85) return "text-emerald-400";
  if (c >= 0.65) return "text-amber-400";
  return "text-rose-400";
}

export function TrajectoryTimeline({ trajectory, onCenterMap, canViewFullPlate }: TrajectoryTimelineProps) {
  if (!trajectory || trajectory.route.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-gov-border p-5 text-center text-xs text-slate-500">
        Search a plate to display its sighting timeline and cross-camera transitions.
      </div>
    );
  }

  const { route, transitions, unresolved_links } = trajectory;

  return (
    <div className="rounded-xl border border-gov-border bg-gov-panel/40 p-3">
      <div className="mb-3 flex items-center gap-2">
        <Clock className="h-3.5 w-3.5 text-gov-saffron" />
        <span className="text-[11px] font-bold uppercase tracking-wide text-slate-300">
          Visual Trajectory Timeline
        </span>
        <span className="ml-auto font-mono text-[10px] text-slate-500">
          {route.length} observed · {transitions.length} accepted · {unresolved_links.length} unresolved
        </span>
      </div>

      <ol className="relative space-y-0 border-l border-gov-border pl-5">
        {route.map((s, i) => {
          // Find a transition leaving this sighting (accepted).
          const outgoing = transitions.find((e) => e.source_sighting_id === s.sighting_id);
          return (
            <li key={s.sighting_id} className="relative pb-5 last:pb-0">
              <span className="absolute -left-[27px] flex h-4 w-4 items-center justify-center rounded-full border-2 border-gov-saffron bg-gov-navy">
                <MapPin className="h-2 w-2 text-gov-saffron" />
              </span>

              <button
                onClick={() =>
                  s.latitude != null && s.longitude != null && onCenterMap(s.latitude, s.longitude)
                }
                className="w-full rounded-lg border border-gov-border bg-gov-navy/50 p-2.5 text-left transition-colors hover:border-gov-saffron/50"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-white">{s.camera_name}</span>
                    <span className="rounded bg-emerald-500/15 px-1 py-0.5 text-[8px] font-bold text-emerald-400">
                      OBSERVED
                    </span>
                    {!canViewFullPlate && "*".includes(s.plate) && (
                      <span className="text-[9px] text-amber-400">masked</span>
                    )}
                  </div>
                  <span className="font-mono text-[11px] text-slate-300">{fmtTime(s.timestamp)}</span>
                </div>

                <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px]">
                  <span className="flex items-center gap-1 text-slate-400">
                    <Car className="h-2.5 w-2.5" /> Track #{s.vehicle_local_track_id}
                  </span>
                  <PlateChip plate={s.plate} className="!px-1.5 !py-0 !text-[11px]" />
                  <span className={cn("font-semibold", confidenceColor(s.plate_confidence))}>
                    OCR {formatPercentage(s.plate_confidence)}
                  </span>
                  {s.vehicle_type && <span className="text-slate-400">{s.vehicle_type}</span>}
                  {s.vehicle_color && (
                    <span className="text-slate-400">color: {s.vehicle_color}</span>
                  )}
                </div>
              </button>

              {outgoing && (
                <div className="mt-2 ml-1 flex items-center gap-2 text-[10px]">
                  <ArrowRight className="h-3.5 w-3.5 text-cyan-400" />
                  <span className="text-cyan-300">
                    INFERRED → {outgoing.destination_camera_name}
                  </span>
                  <span className="font-mono text-slate-500">
                    S={outgoing.combined_score.toFixed(3)} · plate{" "}
                    {formatPercentage(outgoing.plate_score)} · vis{" "}
                    {formatPercentage(outgoing.visual_similarity)} · Δt{" "}
                    {outgoing.time_delta.toFixed(1)}s / exp {outgoing.expected_travel_time.toFixed(1)}s
                  </span>
                </div>
              )}
            </li>
          );
        })}
      </ol>

      {unresolved_links.length > 0 && (
        <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/5 p-2">
          <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wide text-amber-300">
            <ShieldQuestion className="h-3 w-3" /> Unresolved links (human review)
          </div>
          <ul className="mt-1.5 space-y-1">
            {unresolved_links.slice(0, 6).map((e) => (
              <li key={e.id} className="flex flex-wrap items-center gap-x-2 font-mono text-[10px] text-slate-400">
                <span className="text-slate-300">
                  {e.source_camera_name} → {e.destination_camera_name}
                </span>
                <span>S={e.combined_score.toFixed(3)}</span>
                <span className="text-amber-400/80">{e.decision_reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export { CheckCircle2 };
