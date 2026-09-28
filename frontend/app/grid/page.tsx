"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  KeyRound,
  Layers,
  Map as MapIcon,
  Cctv,
  Radar,
  Loader2,
  Sparkles,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import * as api from "@/lib/grid-api";
import {
  CameraDeviceInfo,
  GridCamera,
  HeatmapPoint,
  SystemStatus,
  TrafficAnalytics,
  Trajectory,
} from "@/lib/grid-types";
import { CameraManagerPanel, AddCameraDraft } from "@/components/grid/CameraManagerPanel";
import { VehicleSearchPanel } from "@/components/grid/VehicleSearchPanel";
import { TrajectoryTimeline } from "@/components/grid/TrajectoryTimeline";
import { LiveCameraGrid } from "@/components/grid/LiveCameraGrid";
import { TrafficAnalyticsPanel, SystemStatusBar } from "@/components/grid/TrafficAnalyticsPanel";

const CityMap = dynamic(() => import("@/components/grid/CityMap"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center rounded-xl border border-gov-border bg-gov-panel/40">
      <Loader2 className="h-5 w-5 animate-spin text-gov-saffron" />
    </div>
  ),
});

const EMPTY_DRAFT: AddCameraDraft = {
  name: "",
  sourceType: "demo",
  sourceUri: "demo",
  deviceId: "",
  latitude: null,
  longitude: null,
};

export default function DrishtiGridPage() {
  const [cameras, setCameras] = useState<GridCamera[]>([]);
  const [devices, setDevices] = useState<CameraDeviceInfo[]>([]);
  const [devicesNote, setDevicesNote] = useState<string | undefined>();
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [addDraft, setAddDraft] = useState<AddCameraDraft>(EMPTY_DRAFT);
  const [picking, setPicking] = useState(false);
  const [saving, setSaving] = useState(false);
  const [probing, setProbing] = useState(false);
  const [probeResult, setProbeResult] = useState<{ status: string; message: string } | null>(null);

  const [apiKeyInput, setApiKeyInput] = useState("");
  const [showKeyPanel, setShowKeyPanel] = useState(false);

  const [trajectory, setTrajectory] = useState<Trajectory | null>(null);
  const [searching, setSearching] = useState(false);
  const [searched, setSearched] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  const [traffic, setTraffic] = useState<TrafficAnalytics | null>(null);
  const [heatPoints, setHeatPoints] = useState<HeatmapPoint[] | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [recenterTo, setRecenterTo] = useState<[number, number] | null>(null);
  const [showAnalytics, setShowAnalytics] = useState(true);

  const canWrite = true; // backend enforces; a viewer key surfaces a 403 in the UI

  const refresh = useCallback(async () => {
    setBusy(true);
    try {
      const data = await api.listCameras();
      setCameras(data.cameras);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, []);

  const refreshDevices = useCallback(async () => {
    try {
      const data = await api.listCameraDevices();
      setDevices(data.devices);
      setDevicesNote(data.note);
    } catch (e) {
      setDevices([]);
      setDevicesNote((e as Error).message);
    }
  }, []);

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await api.systemStatus());
    } catch {
      setStatus(null);
    }
  }, []);

  const refreshAnalytics = useCallback(async () => {
    try {
      const [t, h] = await Promise.all([api.trafficAnalytics(3600), api.heatmap(3600)]);
      setTraffic(t);
      setHeatPoints(h.points);
    } catch {
      /* analytics are best-effort */
    }
  }, []);

  useEffect(() => {
    setApiKeyInput(api.getApiKey() || "");
    refresh();
    refreshDevices();
    refreshStatus();
    refreshAnalytics();
    setShowKeyPanel(!api.getApiKey());
    const timer = setInterval(() => {
      refresh();
      refreshStatus();
    }, 4000);
    const anTimer = setInterval(refreshAnalytics, 15000);
    return () => {
      clearInterval(timer);
      clearInterval(anTimer);
    };
  }, [refresh, refreshDevices, refreshStatus, refreshAnalytics]);

  // ------- Add camera flow -------
  const handleSaveCamera = async () => {
    if (addDraft.latitude == null || addDraft.longitude == null) {
      setError("Pick a location on the map before saving.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await api.createCamera({
        name: addDraft.name,
        source_type: addDraft.sourceType,
        source_uri: addDraft.sourceType === "device" ? "" : addDraft.sourceUri || "demo",
        device_id: addDraft.sourceType === "device" ? addDraft.deviceId || "device:0" : undefined,
        latitude: addDraft.latitude,
        longitude: addDraft.longitude,
        is_demo: addDraft.sourceType === "demo",
      });
      setAddDraft(EMPTY_DRAFT);
      setPicking(false);
      setProbeResult(null);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const handleProbe = async () => {
    setProbing(true);
    try {
      const res = await api.probeCameraSource({
        source_type: addDraft.sourceType,
        source_uri: addDraft.sourceType === "device" ? undefined : addDraft.sourceUri,
        device_id: addDraft.sourceType === "device" ? addDraft.deviceId || "device:0" : undefined,
      });
      setProbeResult({ status: res.status, message: res.message });
    } catch (e) {
      setProbeResult({ status: "error", message: (e as Error).message });
    } finally {
      setProbing(false);
    }
  };

  const handleStart = async (id: string) => {
    try {
      await api.startCamera(id);
      await refresh();
      refreshStatus();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleStop = async (id: string) => {
    try {
      await api.stopCamera(id);
      await refresh();
      refreshStatus();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await api.deleteCamera(id);
      if (selectedCameraId === id) setSelectedCameraId(null);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const loadDemo = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.setupDemoCameras();
      await refresh();
      refreshStatus();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  // ------- Plate search -------
  const handleSearch = async (plate: string) => {
    setSearching(true);
    setSearchError(null);
    setSearched(true);
    try {
      const res = await api.searchPlate(plate);
      const top = res.matches[0];
      if (!top) {
        setTrajectory(null);
        return;
      }
      const traj = await api.vehicleTrajectory(top.identity.id);
      setTrajectory(traj);
      const first = traj.route.find((s) => s.latitude != null && s.longitude != null);
      if (first) setRecenterTo([first.latitude as number, first.longitude as number]);
    } catch (e) {
      setTrajectory(null);
      setSearchError((e as Error).message);
    } finally {
      setSearching(false);
    }
  };

  const saveApiKey = () => {
    api.setApiKey(apiKeyInput);
    setShowKeyPanel(false);
    refresh();
    refreshStatus();
  };

  const canViewFullPlate =
    !trajectory || !trajectory.route.some((s) => (s.plate || "").includes("*"));

  const center: [number, number] = useMemo(() => {
    const withCoords = cameras.filter((c) => c.latitude && c.longitude);
    if (withCoords.length === 0) return [23.0225, 72.5714];
    const lat = withCoords.reduce((a, c) => a + c.latitude, 0) / withCoords.length;
    const lon = withCoords.reduce((a, c) => a + c.longitude, 0) / withCoords.length;
    return [lat, lon];
  }, [cameras]);

  return (
    <div className="space-y-4 pb-20">
      {/* Header */}
      <div className="flex flex-col gap-2 rounded-2xl border border-gov-border bg-gradient-to-br from-gov-deepnavy via-gov-panel to-gov-navy p-4 shadow-xl sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-gov-saffron to-orange-600 text-gov-navy">
            <Radar className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-lg font-black tracking-tight text-white sm:text-xl">DRISHTI GRID</h1>
            <p className="text-[10px] font-medium text-slate-400 sm:text-[11px]">
              City-Wide Multi-Camera ANPR Trajectory Tracking &amp; Urban Traffic Analytics
            </p>
          </div>
        </div>
        <div className="flex flex-col items-start gap-2 sm:items-end">
          <SystemStatusBar status={status} />
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2 text-[11px]"
              onClick={() => setShowKeyPanel((v) => !v)}
            >
              <KeyRound className="h-3 w-3" />
              {api.getApiKey() ? "Access key set" : "Set access key"}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              className="h-7 px-2 text-[11px]"
              onClick={loadDemo}
              disabled={busy}
            >
              <Sparkles className="h-3 w-3" />
              Load demo corridor
            </Button>
          </div>
        </div>
      </div>

      {/* Access key panel */}
      {showKeyPanel && (
        <div className="flex flex-col gap-2 rounded-xl border border-gov-border bg-gov-panel/50 p-3 sm:flex-row sm:items-center">
          <ShieldCheck className="h-4 w-4 shrink-0 text-gov-saffron" />
          <div className="flex-1">
            <p className="text-[11px] text-slate-300">
              API key for role-based access. Sent as{" "}
              <span className="font-mono text-gov-saffron">X-API-Key</span>; stored only in this
              browser session. Without a key you have read-only (viewer) access and plates are
              masked.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="password"
              value={apiKeyInput}
              onChange={(e) => setApiKeyInput(e.target.value)}
              placeholder="X-API-Key"
              className="w-48 rounded-md border border-gov-border bg-gov-navy px-2 py-1.5 font-mono text-xs text-white focus:border-gov-saffron focus:outline-none"
            />
            <Button size="sm" variant="cyan" className="h-8" onClick={saveApiKey}>
              Apply
            </Button>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
          {error}
        </div>
      )}

      {/* Main grid: camera panel + map */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[300px_1fr]">
        <div className="rounded-2xl border border-gov-border bg-gov-panel/30 p-3 lg:max-h-[620px]">
          <CameraManagerPanel
            cameras={cameras}
            devices={devices}
            devicesNote={devicesNote}
            selectedCameraId={selectedCameraId}
            onSelectCamera={(id) => {
              setSelectedCameraId(id);
              const cam = cameras.find((c) => c.id === id);
              if (cam) setRecenterTo([cam.latitude, cam.longitude]);
            }}
            onStart={handleStart}
            onStop={handleStop}
            onDelete={handleDelete}
            onRefreshDevices={refreshDevices}
            onRefresh={refresh}
            addDraft={addDraft}
            onAddDraftChange={(patch) => setAddDraft((d) => ({ ...d, ...patch }))}
            onBeginPick={() => setPicking(true)}
            picking={picking}
            onSaveCamera={handleSaveCamera}
            onCancelAdd={() => {
              setAddDraft(EMPTY_DRAFT);
              setProbeResult(null);
            }}
            saving={saving}
            canWrite={canWrite}
            busy={busy}
            error={null}
            onProbe={handleProbe}
            probing={probing}
            probeResult={probeResult}
          />
        </div>

        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MapIcon className="h-4 w-4 text-gov-saffron" />
              <h2 className="text-sm font-bold text-white">City Map</h2>
            </div>
            <button
              onClick={() => setShowHeatmap((v) => !v)}
              className={cn(
                "flex items-center gap-1.5 rounded-lg border px-2 py-1 text-[10px] font-semibold transition-colors",
                showHeatmap
                  ? "border-gov-saffron/60 bg-gov-saffron/10 text-gov-saffron"
                  : "border-gov-border text-slate-400 hover:text-white"
              )}
            >
              <Layers className="h-3 w-3" /> Traffic heatmap
            </button>
          </div>

          <div className="h-[420px] w-full sm:h-[520px]">
            <CityMap
              cameras={cameras}
              trajectory={trajectory}
              selectedCameraId={selectedCameraId}
              onSelectCamera={(id) => setSelectedCameraId(id)}
              mode={picking ? "pick" : "view"}
              onPickLocation={(lat, lng) => {
                setAddDraft((d) => ({ ...d, latitude: lat, longitude: lng }));
                setPicking(false);
              }}
              pickedLocation={
                addDraft.latitude != null && addDraft.longitude != null
                  ? { lat: addDraft.latitude, lng: addDraft.longitude }
                  : null
              }
              heatmapPoints={heatPoints}
              showHeatmap={showHeatmap}
              recenterTo={recenterTo}
              center={center}
            />
          </div>
        </div>
      </div>

      {/* Search + trajectory */}
      <VehicleSearchPanel
        onSearch={handleSearch}
        searching={searching}
        error={searchError}
        trajectory={trajectory}
        searched={searched}
        canViewFullPlate={canViewFullPlate}
        onSelectSighting={(lat, lng) => setRecenterTo([lat, lng])}
      />

      {/* Timeline */}
      {trajectory && (
        <TrajectoryTimeline
          trajectory={trajectory}
          onCenterMap={(lat, lng) => setRecenterTo([lat, lng])}
          canViewFullPlate={canViewFullPlate}
        />
      )}

      {/* Live camera dashboard */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Cctv className="h-4 w-4 text-gov-saffron" />
          <h2 className="text-sm font-bold text-white">Live Camera Dashboard</h2>
          <span className="text-[10px] text-slate-500">
            annotated snapshots: vehicle boxes, track IDs, plate boxes, recognized text
          </span>
        </div>
        <LiveCameraGrid
          cameras={cameras}
          selectedCameraId={selectedCameraId}
          onSelectCamera={(id) => setSelectedCameraId(id)}
        />
      </div>

      {/* Analytics */}
      <div className="space-y-2">
        <button
          onClick={() => setShowAnalytics((v) => !v)}
          className="flex w-full items-center gap-2"
        >
          <Radar className="h-4 w-4 text-gov-saffron" />
          <h2 className="text-sm font-bold text-white">Traffic Analytics</h2>
          {showAnalytics ? (
            <ChevronUp className="h-3.5 w-3.5 text-slate-400" />
          ) : (
            <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
          )}
        </button>
        {showAnalytics && <TrafficAnalyticsPanel traffic={traffic} />}
      </div>

      {/* Deployment note */}
      <div className="rounded-xl border border-gov-border bg-gov-panel/30 p-3 text-[10px] leading-relaxed text-slate-500">
        <strong className="text-slate-400">Privacy &amp; scope:</strong> This dashboard is
        intended for authorised / synthetic / demo traffic footage. Camera frames are stored as
        lightweight annotated JPEG snapshots on the edge; raw video is never transmitted to the
        browser. Plate searches are written to the audit log, and plates are masked for roles
        without the <span className="font-mono">view_full_plate</span> permission.
        {status?.demo_mode && (
          <>
            {" "}
            <span className="text-gov-saffron">
              Demo mode is active — camera feeds labelled SYNTHETIC DEMO are rendered synthetic
              scenes, processed by the real detection/OCR/fusion pipeline (no fabricated
              results).
            </span>
          </>
        )}
      </div>
    </div>
  );
}
