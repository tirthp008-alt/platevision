"use client";

import { useState, useEffect, useRef } from "react";
import {
  Cctv,
  Play,
  Square,
  RefreshCw,
  Download,
  ShieldCheck,
  Clock,
  Car,
  Activity,
  Layers,
  Radio,
  Copy,
  Check,
  Search,
  Filter,
  Sliders,
  AlertTriangle,
  Zap,
  Info,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { getFormatStatusDetails } from "@/lib/utils";

interface CCTVEvent {
  plate_number: string;
  formatted_number: string;
  confidence: number;
  format_status: "valid" | "possible" | "uncertain";
  first_seen: number;
  last_seen: number;
  crop_base64: string;
}

interface ProbeResult {
  status: "ok" | "error";
  protocol: string;
  resolution?: string;
  fps?: number;
  latency_ms?: number;
  message: string;
}

const STREAM_PRESETS = [
  {
    id: "demo_traffic",
    title: "NH-48 Delhi–Gurugram Expressway",
    type: "Highway Stream",
    description: "4-Lane high-speed multi-vehicle corridor simulation",
  },
  {
    id: "demo_toll",
    title: "Mumbai–Pune Expressway Toll Plaza",
    type: "Toll Plaza Feed",
    description: "Multi-lane toll plaza ANPR entry sensor simulation",
  },
  {
    id: "demo_blr",
    title: "Bengaluru Outer Ring Road Junction",
    type: "City Junction",
    description: "Urban arterial multi-plate dense traffic monitoring",
  },
];

const INDIAN_STATE_FILTERS = [
  { label: "All States", value: "" },
  { label: "Gujarat (GJ)", value: "GJ" },
  { label: "Maharashtra (MH)", value: "MH" },
  { label: "Delhi (DL)", value: "DL" },
  { label: "Karnataka (KA)", value: "KA" },
  { label: "Bharat Series (BH)", value: "BH" },
];

export function CCTVStreamViewer() {
  const [streamUrlInput, setStreamUrlInput] = useState<string>("demo_traffic");
  const [selectedPreset, setSelectedPreset] = useState<string>("demo_traffic");
  const [isStreaming, setIsStreaming] = useState<boolean>(true);
  const [statusState, setStatusState] = useState<string>("LIVE STREAMING");
  const [fps, setFps] = useState<number>(24.0);
  const [analysisFps, setAnalysisFps] = useState<number>(3);
  const [totalEvents, setTotalEvents] = useState<number>(0);
  const [events, setEvents] = useState<CCTVEvent[]>([]);
  const [copiedText, setCopiedText] = useState<string | null>(null);

  // Probe / Connection Test states
  const [isProbing, setIsProbing] = useState<boolean>(false);
  const [probeResult, setProbeResult] = useState<ProbeResult | null>(null);

  // Filtering states
  const [stateFilter, setStateFilter] = useState<string>("");
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState<string>("");

  const feedTimestamp = useRef<number>(Date.now());

  // Probe stream connection
  const handleTestConnection = async () => {
    setIsProbing(true);
    setProbeResult(null);

    try {
      const res = await fetch("/api/cctv/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stream_url: streamUrlInput }),
      });
      const data = await res.json();
      setProbeResult(data);
    } catch (e: any) {
      setProbeResult({
        status: "error",
        protocol: "Unknown",
        message: e.message || "Failed to reach backend connection probe.",
      });
    } finally {
      setIsProbing(false);
    }
  };

  // Start live ingestion
  const handleStart = async (url: string = streamUrlInput, targetFps: number = analysisFps) => {
    try {
      setStatusState("CONNECTING");
      await fetch("/api/cctv/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stream_url: url, analysis_fps: targetFps }),
      });
      setIsStreaming(true);
      feedTimestamp.current = Date.now();
    } catch (e) {
      console.error("Failed to start traffic stream:", e);
      setStatusState("ERROR");
    }
  };

  // Stop live ingestion
  const handleStop = async () => {
    try {
      await fetch("/api/cctv/stop", { method: "POST" });
      setIsStreaming(false);
      setStatusState("OFFLINE");
    } catch (e) {
      console.error("Failed to stop traffic stream:", e);
    }
  };

  const handleSelectPreset = (presetId: string) => {
    setSelectedPreset(presetId);
    setStreamUrlInput(presetId);
    setProbeResult(null);
    handleStart(presetId, analysisFps);
  };

  const handleFpsChange = (newFps: number) => {
    setAnalysisFps(newFps);
    if (isStreaming) {
      handleStart(streamUrlInput, newFps);
    }
  };

  // Poll event ledger and status every 1.2s
  useEffect(() => {
    let mounted = true;

    const pollStatus = async () => {
      try {
        const queryParams = new URLSearchParams();
        queryParams.set("limit", "50");
        if (stateFilter) queryParams.set("state_code", stateFilter);
        if (statusFilter) queryParams.set("status_filter", statusFilter);

        const res = await fetch(`/api/cctv/events?${queryParams.toString()}`);
        if (res.ok && mounted) {
          const data = await res.json();
          setEvents(data.events || []);
          setTotalEvents(data.total_events || 0);
          if (data.status_state) setStatusState(data.status_state);
        }
      } catch (e) {}
    };

    pollStatus();
    const interval = setInterval(pollStatus, 1200);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [stateFilter, statusFilter, isStreaming]);

  const handleCopy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedText(text);
      setTimeout(() => setCopiedText(null), 2000);
    } catch (e) {}
  };

  const exportEventsCSV = () => {
    if (events.length === 0) return;
    const rows = [
      ["Timestamp (UTC)", "Registration Number", "Validation Status", "Confidence"],
      ...events.map((e) => [
        new Date(e.last_seen * 1000).toISOString(),
        e.formatted_number || e.plate_number,
        e.format_status,
        `${Math.round(e.confidence * 100)}%`,
      ]),
    ];
    const csvContent = "data:text/csv;charset=utf-8," + rows.map((r) => r.join(",")).join("\n");
    const a = document.createElement("a");
    a.href = encodeURI(csvContent);
    a.download = `anpr_traffic_log_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const filteredEvents = events.filter((e) => {
    if (!searchQuery) return true;
    const q = searchQuery.toUpperCase();
    return (
      e.plate_number.includes(q) ||
      (e.formatted_number && e.formatted_number.toUpperCase().includes(q))
    );
  });

  return (
    <div className="space-y-6 w-full max-w-7xl mx-auto pb-24 font-sans">
      {/* Top Stream Configuration Panel */}
      <div className="rounded-2xl bg-gov-deepnavy/90 border border-gov-border shadow-xl backdrop-blur-md overflow-hidden">
        {/* Panel Header */}
        <div className="p-4 sm:p-5 border-b border-gov-border flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gov-saffron/15 border border-gov-saffron/30 text-gov-saffron">
              <Cctv className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-white tracking-tight font-mono">
                  Live Traffic & Surveillance Stream Portal
                </h2>
                {/* Live Status Badge */}
                <span
                  className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold border flex items-center gap-1.5 ${
                    statusState === "LIVE STREAMING"
                      ? "bg-emerald-500/20 border-emerald-500/40 text-emerald-400"
                      : statusState === "CONNECTING" || statusState === "RECONNECTING"
                      ? "bg-gov-saffron/20 border-gov-saffron/40 text-gov-saffron animate-pulse"
                      : "bg-slate-800 border-slate-700 text-slate-400"
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      statusState === "LIVE STREAMING" ? "bg-emerald-400 animate-ping" : "bg-gov-saffron"
                    }`}
                  />
                  {statusState}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Ingest HLS/M3U8, RTSP, MJPEG, and permitted public traffic feeds with real-time multi-plate detection.
              </p>
            </div>
          </div>

          {/* Master Stream Connection Actions */}
          <div className="flex items-center gap-2">
            {isStreaming ? (
              <Button
                variant="destructive"
                size="sm"
                onClick={handleStop}
                className="font-bold text-xs"
              >
                <Square className="w-3.5 h-3.5 mr-1.5" /> Stop Ingestion
              </Button>
            ) : (
              <Button
                variant="default"
                size="sm"
                onClick={() => handleStart()}
                className="bg-gov-saffron hover:bg-gov-saffronDark text-gov-navy font-bold text-xs shadow-md"
              >
                <Play className="w-3.5 h-3.5 mr-1.5" /> Start Live Ingestion
              </Button>
            )}
          </div>
        </div>

        {/* Source Selector & Presets */}
        <div className="p-4 sm:p-5 space-y-4 bg-gov-navy/50">
          {/* Preset Buttons */}
          <div className="space-y-1.5">
            <label className="text-[11px] font-mono font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
              <Radio className="w-3.5 h-3.5 text-gov-saffron" />
              <span>Select Authorized Surveillance Feed:</span>
            </label>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {STREAM_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => handleSelectPreset(preset.id)}
                  className={`p-3 rounded-xl border text-left transition-all ${
                    selectedPreset === preset.id
                      ? "border-gov-saffron bg-gov-saffron/10 shadow-md shadow-gov-saffron/5"
                      : "border-gov-border bg-gov-panel/60 hover:bg-gov-panel hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-mono text-gov-saffron font-bold uppercase">
                      {preset.type}
                    </span>
                    {selectedPreset === preset.id && (
                      <span className="w-2 h-2 rounded-full bg-gov-saffron" />
                    )}
                  </div>
                  <h4 className="text-xs font-bold text-white tracking-tight">{preset.title}</h4>
                  <p className="text-[10px] text-slate-400 mt-0.5 line-clamp-1">{preset.description}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Custom Stream URL & Connection Probe Controls */}
          <div className="space-y-2 pt-2 border-t border-gov-border">
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5">
              <div className="relative flex-1">
                <input
                  type="text"
                  placeholder="Enter HLS (.m3u8), RTSP (rtsp://...), MJPEG, or permitted public video URL"
                  value={streamUrlInput}
                  onChange={(e) => {
                    setStreamUrlInput(e.target.value);
                    setSelectedPreset("custom");
                  }}
                  className="w-full bg-gov-panel border border-gov-border rounded-xl px-3.5 py-2 text-xs font-mono text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-gov-saffron focus:border-gov-saffron"
                />
              </div>

              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleTestConnection}
                  disabled={isProbing}
                  className="border-gov-border bg-gov-panel text-slate-200 hover:border-gov-saffron text-xs font-semibold"
                >
                  {isProbing ? (
                    <RefreshCw className="w-3.5 h-3.5 animate-spin mr-1.5 text-gov-saffron" />
                  ) : (
                    <Activity className="w-3.5 h-3.5 mr-1.5 text-gov-saffron" />
                  )}
                  <span>Test Connection</span>
                </Button>

                <Button
                  type="button"
                  size="sm"
                  onClick={() => handleStart(streamUrlInput)}
                  className="bg-gov-saffron hover:bg-gov-saffronDark text-gov-navy font-bold text-xs"
                >
                  Connect Stream
                </Button>
              </div>
            </div>

            {/* Probe Feedback Banner */}
            {probeResult && (
              <div
                className={`p-3 rounded-xl border text-xs flex items-start gap-2.5 transition-all ${
                  probeResult.status === "ok"
                    ? "bg-emerald-950/30 border-emerald-500/40 text-emerald-300"
                    : "bg-rose-950/30 border-rose-500/40 text-rose-300"
                }`}
              >
                {probeResult.status === "ok" ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                ) : (
                  <XCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                )}
                <div className="space-y-0.5 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-bold font-mono uppercase text-[11px]">
                      {probeResult.protocol}
                    </span>
                    {probeResult.resolution && (
                      <span className="px-1.5 py-0.2 rounded bg-black/40 font-mono text-[10px]">
                        Resolution: {probeResult.resolution}
                      </span>
                    )}
                    {probeResult.latency_ms !== undefined && (
                      <span className="px-1.5 py-0.2 rounded bg-black/40 font-mono text-[10px]">
                        Ping: {probeResult.latency_ms}ms
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-slate-300 leading-relaxed">{probeResult.message}</p>
                </div>
              </div>
            )}
          </div>

          {/* Analysis Frame-Rate Selector Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-gov-border text-xs">
            <div className="flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-gov-saffron" />
              <span className="font-mono text-[11px] text-slate-300 font-bold">AI Analysis Rate:</span>
              <div className="flex items-center gap-1 bg-gov-panel p-1 rounded-lg border border-gov-border">
                {[
                  { label: "1 FPS (Eco)", value: 1 },
                  { label: "3 FPS (Balanced)", value: 3 },
                  { label: "5 FPS (Precision)", value: 5 },
                ].map((rate) => (
                  <button
                    key={rate.value}
                    type="button"
                    onClick={() => handleFpsChange(rate.value)}
                    className={`px-2.5 py-1 rounded text-[10px] font-mono font-bold transition-colors ${
                      analysisFps === rate.value
                        ? "bg-gov-saffron text-gov-navy"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {rate.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-3 text-[11px] font-mono text-slate-400">
              <span>Stream FPS: <strong className="text-white">{fps.toFixed(1)}</strong></span>
              <span>•</span>
              <span>Logged Events: <strong className="text-gov-saffron">{totalEvents}</strong></span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Monitoring Grid: Live Video View on Left, ANPR Event Ledger on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Live Video Canvas */}
        <div className="lg:col-span-7 space-y-3">
          <div className="relative aspect-video w-full rounded-2xl bg-black border border-gov-border overflow-hidden shadow-2xl flex items-center justify-center">
            {isStreaming ? (
              <img
                src={`/api/cctv/feed?t=${feedTimestamp.current}`}
                alt="Live Traffic Surveillance Video Feed"
                className="w-full h-full object-contain block"
              />
            ) : (
              <div className="flex flex-col items-center justify-center p-8 text-center space-y-3">
                <Cctv className="w-12 h-12 text-slate-600" />
                <h4 className="text-base font-bold text-white">Surveillance Stream Ingestion Paused</h4>
                <p className="text-xs text-slate-400 max-w-sm">
                  Click &apos;Connect Stream&apos; to resume real-time multi-plate ANPR monitoring and recording.
                </p>
                <Button
                  variant="default"
                  size="sm"
                  onClick={() => handleStart()}
                  className="bg-gov-saffron text-gov-navy font-bold hover:bg-gov-saffronDark text-xs"
                >
                  <Play className="w-3.5 h-3.5 mr-1" /> Resume Live Feed
                </Button>
              </div>
            )}

            {/* Top HUD Watermark */}
            {isStreaming && (
              <div className="absolute top-3 left-3 bg-gov-deepnavy/85 backdrop-blur-md px-3 py-1 rounded-lg border border-gov-border flex items-center gap-2 text-xs font-mono text-gov-saffron shadow-lg">
                <Activity className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
                <span>ANPR SENSOR S-01 | {analysisFps} FPS AI INGESTION</span>
              </div>
            )}
          </div>

          {/* Privacy & Compliance Warning Box */}
          <div className="p-3.5 rounded-xl bg-gov-panel/60 border border-gov-border text-[11px] text-slate-400 flex items-start gap-2.5">
            <Info className="w-4 h-4 text-gov-saffron shrink-0 mt-0.5" />
            <p className="leading-relaxed">
              <strong>Authorized Surveillance Compliance:</strong> Ensure all connected RTSP/IP camera endpoints comply with local public data collection regulations and the Digital Personal Data Protection (DPDP) Act 2023. Video frames are processed in transient system memory.
            </p>
          </div>
        </div>

        {/* Right Column: Live ANPR Event Ledger */}
        <div className="lg:col-span-5 space-y-3">
          {/* Ledger Header & Filters */}
          <div className="p-4 rounded-xl bg-gov-deepnavy border border-gov-border space-y-3 shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white font-mono flex items-center gap-2">
                  <Layers className="w-4 h-4 text-gov-saffron" />
                  Live Detected Vehicles ({filteredEvents.length})
                </h3>
                <span className="text-[10px] text-slate-400 font-mono">Continuous RTO format audit log</span>
              </div>

              {filteredEvents.length > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={exportEventsCSV}
                  className="border-gov-border bg-gov-panel text-slate-200 hover:border-gov-saffron text-xs"
                >
                  <Download className="w-3.5 h-3.5 mr-1 text-gov-saffron" /> Export CSV
                </Button>
              )}
            </div>

            {/* Filter & Search Bar */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 border-t border-gov-border">
              {/* Search Plate Box */}
              <div className="relative">
                <Search className="w-3 h-3 text-slate-400 absolute left-2.5 top-2.5" />
                <input
                  type="text"
                  placeholder="Search plate e.g. GJ01..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-gov-panel border border-gov-border rounded-lg pl-7 pr-2 py-1.5 text-xs font-mono text-slate-200 placeholder-slate-500 focus:outline-none focus:border-gov-saffron"
                />
              </div>

              {/* State Filter Dropdown */}
              <select
                value={stateFilter}
                onChange={(e) => setStateFilter(e.target.value)}
                className="w-full bg-gov-panel border border-gov-border rounded-lg px-2 py-1.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-gov-saffron"
              >
                {INDIAN_STATE_FILTERS.map((st) => (
                  <option key={st.value} value={st.value}>
                    {st.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Event Stream Cards */}
          <div className="space-y-2 max-h-[480px] overflow-y-auto pr-1">
            {filteredEvents.length === 0 && (
              <div className="p-8 text-center rounded-xl bg-gov-panel/40 border border-gov-border text-xs text-slate-400 space-y-2">
                <Car className="w-8 h-8 text-slate-600 mx-auto" />
                <p>Watching live stream for passing vehicles...</p>
              </div>
            )}

            {filteredEvents.map((ev, i) => {
              const statusDetails = getFormatStatusDetails(ev.format_status);
              const timeStr = new Date(ev.last_seen * 1000).toLocaleTimeString("en-IN", {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
                hour12: true,
              });

              return (
                <div
                  key={`${ev.plate_number}_${i}`}
                  className="p-3 rounded-xl border border-gov-border bg-gov-panel/80 hover:border-gov-saffron/60 transition-all flex items-center justify-between gap-3 shadow-md"
                >
                  {/* Left: Thumbnail & Registration */}
                  <div className="flex items-center gap-3">
                    {ev.crop_base64 && (
                      <div className="w-16 h-8 bg-black rounded border border-slate-700 overflow-hidden flex items-center justify-center shrink-0">
                        <img
                          src={ev.crop_base64}
                          alt="Plate Crop"
                          className="max-h-full max-w-full object-contain"
                        />
                      </div>
                    )}

                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-sm font-black font-mono tracking-wider text-white">
                          {ev.formatted_number || ev.plate_number}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-[10px] text-slate-400 font-mono">
                        <span className="flex items-center gap-1">
                          <Clock className="w-2.5 h-2.5 text-slate-500" />
                          {timeStr}
                        </span>
                        <span>•</span>
                        <span className={`px-1.5 py-0.2 rounded font-bold ${statusDetails.colorClass}`}>
                          {statusDetails.label}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Right: Confidence Score & One-Click Copy */}
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-bold text-gov-saffron bg-gov-navy px-2 py-0.5 rounded border border-gov-border">
                      {Math.round(ev.confidence * 100)}%
                    </span>
                    <button
                      type="button"
                      onClick={() => handleCopy(ev.formatted_number || ev.plate_number)}
                      title="Copy registration number"
                      className="p-1.5 rounded-lg bg-gov-navy border border-gov-border text-slate-400 hover:text-white hover:border-gov-saffron transition-colors"
                    >
                      {copiedText === (ev.formatted_number || ev.plate_number) ? (
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
