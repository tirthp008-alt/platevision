"use client";

import { Activity, Gauge, TrendingUp, Car } from "lucide-react";
import { cn } from "@/lib/utils";
import { SystemStatus, TrafficAnalytics } from "@/lib/grid-types";

function congestionColor(r: number): string {
  if (r >= 0.8) return "text-rose-400";
  if (r >= 0.5) return "text-amber-400";
  return "text-emerald-400";
}

export function TrafficAnalyticsPanel({ traffic }: { traffic: TrafficAnalytics | null }) {
  if (!traffic) {
    return (
      <div className="rounded-xl border border-dashed border-gov-border p-5 text-center text-xs text-slate-500">
        Analytics will populate once cameras produce sightings and accepted transitions.
      </div>
    );
  }

  const totals = traffic.totals;
  const cards = [
    { label: "Sightings", value: totals.sightings, icon: Car, color: "text-cyan-400" },
    { label: "Accepted links", value: totals.accepted_transitions, icon: TrendingUp, color: "text-emerald-400" },
    { label: "Unique vehicles", value: totals.unique_vehicles, icon: Activity, color: "text-gov-saffron" },
    { label: "Active cameras", value: totals.active_cameras, icon: Gauge, color: "text-blue-400" },
  ];

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {cards.map((c) => {
          const Icon = c.icon;
          return (
            <div key={c.label} className="rounded-lg border border-gov-border bg-gov-panel/50 p-2.5">
              <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-slate-500">
                <Icon className={cn("h-3 w-3", c.color)} />
                {c.label}
              </div>
              <div className={cn("mt-1 font-mono text-xl font-bold", c.color)}>{c.value}</div>
            </div>
          );
        })}
      </div>

      <div className="rounded-lg border border-gov-border bg-gov-panel/40 p-3">
        <div className="mb-2 flex items-center gap-2">
          <TrendingUp className="h-3.5 w-3.5 text-gov-saffron" />
          <span className="text-[11px] font-bold uppercase tracking-wide text-slate-300">
            Corridor Segment Metrics
          </span>
          <span className="ml-auto text-[9px] text-slate-500">
            approximation — baseline capacity model
          </span>
        </div>
        {traffic.segments.length === 0 ? (
          <p className="text-[11px] text-slate-500">
            No accepted transitions yet, so no segment metrics are available.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-[10px]">
              <thead className="text-slate-500">
                <tr className="border-b border-gov-border/60">
                  <th className="py-1 pr-2 font-medium">Segment</th>
                  <th className="py-1 px-2 font-medium">N</th>
                  <th className="py-1 px-2 font-medium">Dist</th>
                  <th className="py-1 px-2 font-medium">Avg speed</th>
                  <th className="py-1 px-2 font-medium">Avg Δt</th>
                  <th className="py-1 px-2 font-medium">Flow/h</th>
                  <th className="py-1 pl-2 font-medium">Congestion</th>
                </tr>
              </thead>
              <tbody>
                {traffic.segments.map((s) => (
                  <tr key={s.segment} className="border-b border-gov-border/30 last:border-0">
                    <td className="py-1 pr-2 text-slate-200">
                      {(s.source_camera_name || s.source_camera).slice(0, 18)}
                      {" → "}
                      {(s.destination_camera_name || s.destination_camera).slice(0, 18)}
                    </td>
                    <td className="py-1 px-2 text-slate-300">{s.vehicle_count}</td>
                    <td className="py-1 px-2 text-slate-400">{s.distance_m.toFixed(0)}m</td>
                    <td className="py-1 px-2 text-slate-300">{s.average_speed_kmh.toFixed(1)} km/h</td>
                    <td className="py-1 px-2 text-slate-400">{s.average_travel_time.toFixed(1)}s</td>
                    <td className="py-1 px-2 text-slate-300">{s.flow_vph.toFixed(1)}</td>
                    <td className={cn("py-1 pl-2 font-semibold", congestionColor(s.congestion_ratio))}>
                      {(s.congestion_ratio * 100).toFixed(0)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export function SystemStatusBar({ status }: { status: SystemStatus | null }) {
  if (!status) {
    return (
      <div className="flex items-center gap-2 font-mono text-[10px] text-slate-500">
        <span className="h-2 w-2 rounded-full bg-slate-500" /> connecting to backend…
      </div>
    );
  }
  const a = status.aggregate;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10px]">
      <span className="flex items-center gap-1.5">
        <span
          className={cn(
            "h-2 w-2 rounded-full",
            a.cameras_processing > 0 ? "animate-pulse bg-emerald-400" : "bg-slate-500"
          )}
        />
        <span className="text-slate-300">
          {a.cameras_processing}/{a.cameras_registered} cameras
        </span>
      </span>
      <span className="text-slate-400">
        frames: <span className="text-slate-200">{a.frames_processed}</span>
      </span>
      <span className="text-slate-400">
        sightings: <span className="text-slate-200">{a.sightings_created}</span>
      </span>
      <span className="text-slate-400">
        fps: <span className="text-slate-200">{a.average_fps.toFixed(1)}</span>
      </span>
      <span className="text-slate-400">
        detector: <span className="text-slate-200">{status.detector}</span>
      </span>
      <span className="text-slate-400">
        events: <span className="text-slate-200">{status.event_backend}</span>
      </span>
      {status.demo_mode && (
        <span className="rounded bg-gov-saffron/20 px-1.5 py-0.5 text-[9px] font-bold text-gov-saffron">
          DEMO MODE
        </span>
      )}
    </div>
  );
}
