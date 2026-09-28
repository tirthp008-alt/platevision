"use client";

import { useState } from "react";
import {
  Search,
  Car,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  ArrowRight,
  Route,
  Loader2,
  ShieldQuestion,
  Eye,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn, formatPercentage } from "@/lib/utils";
import { Trajectory, TrajectoryEdge } from "@/lib/grid-types";

function fmtTime(ts?: number | null): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleTimeString();
}

function PlateChip({ plate, className }: { plate: string; className?: string }) {
  return (
    <span
      className={cn(
        "inline-block rounded border border-slate-600 bg-white px-2 py-0.5 font-mono text-sm font-bold tracking-widest text-slate-900",
        className
      )}
    >
      {plate || "---------"}
    </span>
  );
}

function DecisionBadge({ decision }: { decision: string }) {
  const map: Record<string, { cls: string; icon: React.ReactNode; label: string }> = {
    ACCEPTED: {
      cls: "border-emerald-500/40 bg-emerald-500/10 text-emerald-400",
      icon: <CheckCircle2 className="h-3 w-3" />,
      label: "ACCEPTED",
    },
    ABSTAINED: {
      cls: "border-amber-500/40 bg-amber-500/10 text-amber-400",
      icon: <ShieldQuestion className="h-3 w-3" />,
      label: "UNRESOLVED",
    },
    REJECTED: {
      cls: "border-rose-500/40 bg-rose-500/10 text-rose-400",
      icon: <XCircle className="h-3 w-3" />,
      label: "REJECTED",
    },
  };
  const m = map[decision] || map.REJECTED;
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-bold", m.cls)}>
      {m.icon}
      {m.label}
    </span>
  );
}

function EdgeEvidence({ e }: { e: TrajectoryEdge }) {
  const rows = [
    { label: "Plate evidence", value: formatPercentage(e.plate_score) },
    { label: "Visual similarity", value: formatPercentage(e.visual_similarity) },
    { label: "Observed Δt", value: `${e.time_delta.toFixed(1)} s` },
    { label: "Expected travel time", value: `${e.expected_travel_time.toFixed(1)} s` },
    { label: "Combined score S", value: e.combined_score.toFixed(3) },
  ];
  return (
    <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 rounded-md bg-gov-navy/60 p-2 font-mono text-[10px]">
      {rows.map((r) => (
        <div key={r.label} className="flex items-center justify-between gap-2">
          <span className="text-slate-500">{r.label}</span>
          <span className="text-slate-200">{r.value}</span>
        </div>
      ))}
      <div className="col-span-2 mt-1 border-t border-gov-border/60 pt-1 text-slate-400">
        {e.decision_reason}
      </div>
    </div>
  );
}

interface VehicleSearchPanelProps {
  onSearch: (plate: string) => void;
  searching: boolean;
  error?: string | null;
  trajectory: Trajectory | null;
  searched: boolean;
  onSelectSighting?: (lat: number, lng: number) => void;
  canViewFullPlate: boolean;
}

export function VehicleSearchPanel({
  onSearch,
  searching,
  error,
  trajectory,
  searched,
  onSelectSighting,
  canViewFullPlate,
}: VehicleSearchPanelProps) {
  const [query, setQuery] = useState("GJ01AB1234");

  const submit = () => {
    const q = query.trim();
    if (q) onSearch(q);
  };

  const route = trajectory?.route || [];
  const transitions = trajectory?.transitions || [];
  const unresolved = trajectory?.unresolved_links || [];

  return (
    <div className="space-y-3">
      {/* Search field */}
      <div className="flex items-center gap-2 rounded-xl border border-gov-border bg-gov-panel/60 p-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === "Enter" && submit()}
            placeholder="Enter vehicle number plate (e.g. GJ01AB1234)"
            className="w-full rounded-lg border border-gov-border bg-gov-navy py-2 pl-9 pr-3 font-mono text-sm tracking-widest text-white placeholder:font-sans placeholder:tracking-normal placeholder:text-slate-500 focus:border-gov-saffron focus:outline-none"
          />
        </div>
        <Button variant="cyan" onClick={submit} disabled={searching} className="h-9">
          {searching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
          Search
        </Button>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {searched && !trajectory && !error && (
        <div className="rounded-lg border border-dashed border-gov-border p-4 text-center text-xs text-slate-400">
          No sightings match this plate. The system will not fabricate a route when evidence is insufficient.
        </div>
      )}

      {trajectory && (
        <div className="space-y-3">
          {/* Identity summary */}
          <div className="rounded-xl border border-gov-border bg-gov-panel/50 p-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <Car className="h-5 w-5 text-gov-saffron" />
                <div>
                  <PlateChip plate={trajectory.identity.display_plate} />
                  {!canViewFullPlate && (
                    <span className="ml-2 text-[10px] text-amber-400">
                      masked — full plate requires elevated permission
                    </span>
                  )}
                  <div className="mt-1 font-mono text-[10px] text-slate-400">
                    {trajectory.identity.id} · {trajectory.identity.status} ·{" "}
                    {trajectory.identity.sighting_count} sightings
                  </div>
                </div>
              </div>
              <div className="flex gap-4 font-mono text-[10px]">
                <div>
                  <div className="text-slate-500">First seen</div>
                  <div className="text-slate-200">
                    {route[0]?.camera_name ?? "—"} · {fmtTime(trajectory.first_seen)}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500">Last seen</div>
                  <div className="text-slate-200">
                    {route[route.length - 1]?.camera_name ?? "—"} · {fmtTime(trajectory.last_seen)}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500">Travel time</div>
                  <div className="text-slate-200">
                    {trajectory.inferred_total_travel_time != null
                      ? `${trajectory.inferred_total_travel_time.toFixed(0)} s`
                      : "—"}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Route chain */}
          <div className="rounded-xl border border-gov-border bg-gov-panel/40 p-3">
            <div className="mb-2 flex items-center gap-2">
              <Route className="h-3.5 w-3.5 text-cyan-400" />
              <span className="text-[11px] font-bold uppercase tracking-wide text-slate-300">
                Reconstructed Route
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              {route.map((s, i) => (
                <div key={s.sighting_id} className="flex items-center gap-1.5">
                  <button
                    onClick={() => s.latitude != null && s.longitude != null && onSelectSighting?.(s.latitude, s.longitude)}
                    className="group rounded-lg border border-gov-border bg-gov-navy px-2.5 py-1.5 text-left transition-colors hover:border-gov-saffron/60"
                    title="Center map on this camera"
                  >
                    <div className="flex items-center gap-1.5">
                      <span className="rounded bg-emerald-500/15 px-1 py-0.5 text-[8px] font-bold text-emerald-400">
                        OBSERVED
                      </span>
                      <span className="text-[11px] font-semibold text-white">{s.camera_name}</span>
                    </div>
                    <div className="mt-0.5 font-mono text-[10px] text-slate-400">
                      {fmtTime(s.timestamp)} · <PlateChip plate={s.plate} className="!px-1 !py-0 !text-[10px]" />{" "}
                      <span className="text-slate-500">
                        {formatPercentage(s.plate_confidence)} · Track #{s.vehicle_local_track_id}
                      </span>
                    </div>
                  </button>
                  {i < route.length - 1 && (
                    <ArrowRight className="h-4 w-4 shrink-0 text-cyan-400/70" />
                  )}
                </div>
              ))}
            </div>
            {trajectory.has_inferred_links && (
              <p className="mt-2 border-t border-gov-border/60 pt-2 text-[10px] leading-snug text-slate-500">
                OBSERVED = actual camera detection. Transitions between observations are
                INFERRED from plate + visual + temporal + road-network evidence, never
                presented as direct observations.
              </p>
            )}
          </div>

          {/* Accepted transitions with evidence */}
          <div className="rounded-xl border border-gov-border bg-gov-panel/40 p-3">
            <div className="mb-2 flex items-center gap-2">
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
              <span className="text-[11px] font-bold uppercase tracking-wide text-slate-300">
                Accepted Links ({transitions.length})
              </span>
            </div>
            {transitions.length === 0 ? (
              <p className="text-[11px] text-slate-500">
                No accepted cross-camera links — the sightings at this identity are not yet
                confidently connected.
              </p>
            ) : (
              <div className="space-y-2">
                {transitions.map((e) => (
                  <div key={e.id} className="rounded-lg border border-gov-border/70 bg-gov-navy/40 p-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-1.5 text-[11px] text-slate-200">
                        <span className="font-semibold">{e.source_camera_name}</span>
                        <ArrowRight className="h-3 w-3 text-cyan-400" />
                        <span className="font-semibold">{e.destination_camera_name}</span>
                      </div>
                      <DecisionBadge decision={e.decision} />
                    </div>
                    <EdgeEvidence e={e} />
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Unresolved candidate links */}
          <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-3">
            <div className="mb-2 flex items-center gap-2">
              <ShieldQuestion className="h-3.5 w-3.5 text-amber-400" />
              <span className="text-[11px] font-bold uppercase tracking-wide text-amber-300">
                Unresolved Candidates ({unresolved.length})
              </span>
            </div>
            {unresolved.length === 0 ? (
              <p className="text-[11px] text-slate-500">
                No ambiguous candidate links involved these sightings.
              </p>
            ) : (
              <div className="space-y-2">
                {unresolved.map((e) => (
                  <div key={e.id} className="rounded-lg border border-amber-500/20 bg-gov-navy/40 p-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-1.5 text-[11px] text-slate-200">
                        <span className="font-semibold">{e.source_camera_name}</span>
                        <ArrowRight className="h-3 w-3 text-amber-400" />
                        <span className="font-semibold">{e.destination_camera_name}</span>
                      </div>
                      <DecisionBadge decision={e.decision} />
                    </div>
                    <EdgeEvidence e={e} />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export { DecisionBadge, PlateChip, fmtTime, EdgeEvidence };
