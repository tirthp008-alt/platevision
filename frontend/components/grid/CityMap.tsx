"use client";

import { Fragment, useEffect, useMemo } from "react";
import L from "leaflet";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polyline,
  CircleMarker,
  useMap,
  useMapEvents,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { GridCamera, HeatmapPoint, Trajectory, TrajectoryEdge } from "@/lib/grid-types";

export type MapMode = "view" | "pick";

const STATUS_COLORS: Record<string, string> = {
  ONLINE: "#34d399",
  PROCESSING: "#22d3ee",
  CONNECTING: "#fbbf24",
  OFFLINE: "#64748b",
  ERROR: "#f43f5e",
};

function cameraIcon(status: string, selected: boolean) {
  const color = STATUS_COLORS[status] || "#64748b";
  const ring = selected ? "#FF9933" : "rgba(255,255,255,0.65)";
  return L.divIcon({
    className: "",
    html: `<div style="
        width:22px;height:22px;border-radius:50%;
        background:${color};
        border:3px solid ${ring};
        box-shadow:0 0 10px ${color}aa;
        display:flex;align-items:center;justify-content:center;">
        <div style="width:6px;height:6px;border-radius:50%;background:#0A192F"></div>
      </div>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
  });
}

function ClickHandler({ onPick }: { onPick: (lat: number, lng: number) => void }) {
  useMapEvents({
    click(e) {
      onPick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

function Recenter({ target }: { target: [number, number] | null }) {
  const map = useMap();
  useEffect(() => {
    if (target) map.setView(target, Math.max(map.getZoom(), 14), { animate: true });
  }, [target, map]);
  return null;
}

/** Arrow marker at the midpoint of an edge to indicate travel direction. */
function EdgeArrow({ from, to, color }: { from: [number, number]; to: [number, number]; color: string }) {
  const mid: [number, number] = [(from[0] + to[0]) / 2, (from[1] + to[1]) / 2];
  const angle = (Math.atan2(to[0] - from[0], to[1] - from[1]) * 180) / Math.PI;
  const icon = L.divIcon({
    className: "",
    html: `<div style="transform:rotate(${90 - angle}deg);color:${color};font-size:16px;line-height:16px;text-shadow:0 0 4px #000">&#10148;</div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8],
  });
  return <Marker position={mid} icon={icon} interactive={false} />;
}

interface CityMapProps {
  cameras: GridCamera[];
  trajectory?: Trajectory | null;
  selectedCameraId?: string | null;
  onSelectCamera?: (id: string) => void;
  mode?: MapMode;
  onPickLocation?: (lat: number, lng: number) => void;
  pickedLocation?: { lat: number; lng: number } | null;
  heatmapPoints?: HeatmapPoint[] | null;
  showHeatmap?: boolean;
  recenterTo?: [number, number] | null;
  center?: [number, number];
  zoom?: number;
}

export default function CityMap({
  cameras,
  trajectory,
  selectedCameraId,
  onSelectCamera,
  mode = "view",
  onPickLocation,
  pickedLocation,
  heatmapPoints,
  showHeatmap = false,
  recenterTo = null,
  center = [23.0225, 72.5714],
  zoom = 13,
}: CityMapProps) {
  const cameraById = useMemo(() => {
    const m = new Map<string, GridCamera>();
    cameras.forEach((c) => m.set(c.id, c));
    return m;
  }, [cameras]);

  const acceptedEdges: TrajectoryEdge[] = trajectory?.transitions || [];
  const unresolvedEdges: TrajectoryEdge[] = trajectory?.unresolved_links || [];

  const edgePath = (e: TrajectoryEdge): [number, number][] => {
    if (e.path_geometry && e.path_geometry.length >= 2) {
      return e.path_geometry as [number, number][];
    }
    const a = cameraById.get(e.source_camera_id);
    const b = cameraById.get(e.destination_camera_id);
    if (!a || !b) return [];
    return [
      [a.latitude, a.longitude],
      [b.latitude, b.longitude],
    ];
  };

  const routePositions: [number, number][] = (trajectory?.route || [])
    .filter((s) => s.latitude != null && s.longitude != null)
    .map((s) => [s.latitude as number, s.longitude as number]);

  return (
    <div className="relative h-full w-full overflow-hidden rounded-xl border border-gov-border">
      <MapContainer
        center={center}
        zoom={zoom}
        scrollWheelZoom
        style={{ height: "100%", width: "100%", background: "#0A192F" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <Recenter target={recenterTo} />
        {mode === "pick" && onPickLocation && <ClickHandler onPick={onPickLocation} />}

        {/* Traffic heatmap overlay */}
        {showHeatmap &&
          heatmapPoints?.map((p) => (
            <CircleMarker
              key={`heat-${p.camera_id}`}
              center={[p.latitude, p.longitude]}
              radius={14 + p.intensity * 30}
              pathOptions={{
                color: "#FF9933",
                fillColor: "#FF9933",
                fillOpacity: 0.12 + p.intensity * 0.35,
                stroke: false,
              }}
              interactive={false}
            />
          ))}

        {/* Unresolved / abstained candidate links (dashed, red) */}
        {unresolvedEdges.map((e) => {
          const path = edgePath(e);
          if (path.length < 2) return null;
          return (
            <Polyline
              key={`unres-${e.id}`}
              positions={path}
              pathOptions={{
                color: "#f43f5e",
                weight: 2,
                opacity: 0.75,
                dashArray: "6 6",
              }}
            />
          );
        })}

        {/* Accepted (inferred) trajectory edges along the road network */}
        {acceptedEdges.map((e) => {
          const path = edgePath(e);
          if (path.length < 2) return null;
          return (
            <Fragment key={`edge-${e.id}`}>
              <Polyline
                positions={path}
                pathOptions={{ color: "#22d3ee", weight: 4, opacity: 0.9 }}
              />
              <EdgeArrow from={path[0]} to={path[path.length - 1]} color="#22d3ee" />
            </Fragment>
          );
        })}

        {/* Observed sightings along the route */}
        {routePositions.map((pos, i) => (
          <CircleMarker
            key={`obs-${i}`}
            center={pos}
            radius={6}
            pathOptions={{
              color: "#FF9933",
              fillColor: "#FF9933",
              fillOpacity: 0.9,
              weight: 2,
            }}
          />
        ))}

        {/* Camera markers */}
        {cameras.map((cam) => (
          <Marker
            key={cam.id}
            position={[cam.latitude, cam.longitude]}
            icon={cameraIcon(cam.status, cam.id === selectedCameraId)}
            eventHandlers={{
              click: () => onSelectCamera?.(cam.id),
            }}
          >
            <Popup>
              <div className="space-y-1 text-xs">
                <div className="font-bold text-sm">{cam.name}</div>
                <div>
                  Status: <span className="font-mono">{cam.status}</span>
                </div>
                <div className="font-mono text-[10px]">
                  {cam.latitude.toFixed(5)}, {cam.longitude.toFixed(5)}
                </div>
                <div>
                  Source: <span className="font-mono">{cam.source_type}</span>
                  {cam.source_uri ? ` — ${cam.source_uri}` : ""}
                </div>
                <div>Sightings: {cam.sighting_count ?? 0}</div>
                {cam.runtime?.is_running && (
                  <div>Tracking: {cam.runtime.tracked_vehicles ?? 0} vehicles</div>
                )}
              </div>
            </Popup>
          </Marker>
        ))}

        {/* Pending camera placement in pick mode */}
        {pickedLocation && (
          <CircleMarker
            center={[pickedLocation.lat, pickedLocation.lng]}
            radius={10}
            pathOptions={{
              color: "#FF9933",
              fillColor: "#FF9933",
              fillOpacity: 0.4,
              weight: 2,
              dashArray: "4 4",
            }}
          />
        )}
      </MapContainer>

      {mode === "pick" && (
        <div className="pointer-events-none absolute left-3 top-3 z-[1000] rounded-lg border border-gov-saffron/60 bg-gov-navy/90 px-3 py-1.5 text-[11px] font-semibold text-gov-saffron shadow-lg">
          Click the map to place the camera location
        </div>
      )}
    </div>
  );
}
