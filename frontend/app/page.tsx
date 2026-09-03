"use client";

import Link from "next/link";
import {
  Camera,
  Upload,
  Cctv,
  Film,
  Lock,
  ArrowRight,
  Sparkles,
  Layers,
  History as HistoryIcon,
  ShieldCheck,
  Radio,
  FileCheck2,
  Car,
  Activity,
  CheckCircle2,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useLocalHistory } from "@/hooks/useLocalHistory";

export default function DashboardPage() {
  const { history } = useLocalHistory();

  return (
    <div className="space-y-10 pb-20 font-sans">
      {/* Official Government Portal Hero Section */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-gov-deepnavy via-gov-panel to-gov-navy border border-gov-border p-6 sm:p-10 shadow-2xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-gov-saffron/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-10 -left-10 w-72 h-72 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl space-y-4">
          {/* Institutional Badge */}
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-gov-saffron/30 bg-gov-saffron/10 text-gov-saffron text-xs font-mono font-bold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>National Automated Number Plate Recognition (ANPR) System</span>
          </div>

          <h1 className="text-3xl sm:text-5xl font-black text-white tracking-tight leading-tight">
            High-Precision Vehicle <br className="hidden sm:inline" />
            <span className="bg-gradient-to-r from-gov-saffron via-orange-400 to-amber-300 bg-clip-text text-transparent">
              Traffic Vision & Recognition
            </span>
          </h1>

          <p className="text-sm sm:text-base text-slate-300 leading-relaxed max-w-2xl">
            Official-standard deep learning OCR and edge computer-vision engine for high-speed multi-plate detection across Live Traffic Streams, Mobile Cameras, Photos, and Recorded Video Media.
          </p>

          {/* Quick Metrics Bar */}
          <div className="flex flex-wrap items-center gap-4 sm:gap-6 pt-3 border-t border-gov-border/80 text-xs font-mono">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
              <span className="text-slate-300">AI Core: <strong className="text-emerald-400">YOLOv8 + PaddleOCR</strong></span>
            </div>
            <div className="hidden sm:inline text-slate-600">•</div>
            <div>
              <span className="text-slate-300">Indian RTO Formats: <strong className="text-gov-saffron">HSRP & Bharat (BH)</strong></span>
            </div>
            <div className="hidden sm:inline text-slate-600">•</div>
            <div>
              <span className="text-slate-300">Privacy Standard: <strong className="text-white">DPDP 2023 Compliant</strong></span>
            </div>
          </div>
        </div>
      </div>

      {/* 3 Prominent Surveillance Service Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 max-w-6xl mx-auto">
        {/* Service 1: Live Traffic & CCTV Streams */}
        <Link href="/cctv" className="group block">
          <Card className="h-full border-gov-border bg-gov-deepnavy/90 group-hover:border-gov-saffron group-hover:shadow-gov-saffron/10 group-hover:shadow-2xl transition-all duration-300 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-gov-saffron/10 rounded-full blur-2xl group-hover:bg-gov-saffron/20 transition-all pointer-events-none" />
            <CardHeader className="space-y-3">
              <div className="w-12 h-12 rounded-2xl bg-gov-saffron/15 border border-gov-saffron/30 text-gov-saffron flex items-center justify-center group-hover:scale-110 transition-transform">
                <Cctv className="w-6 h-6" />
              </div>
              <div>
                <CardTitle className="text-lg text-white group-hover:text-gov-saffron transition-colors flex items-center justify-between font-mono">
                  <span>Live Traffic Stream</span>
                  <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-gov-saffron group-hover:translate-x-1 transition-all" />
                </CardTitle>
                <CardDescription className="mt-1 text-xs text-slate-400">
                  Connect RTSP / HLS IP cameras, traffic feeds, or highway presets with real-time multi-plate logging.
                </CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-1.5 text-[10px] font-mono text-slate-400">
                <span className="px-2 py-0.5 rounded bg-gov-panel border border-gov-border">
                  HLS / RTSP / MJPEG
                </span>
                <span className="px-2 py-0.5 rounded bg-gov-panel border border-gov-border">
                  Live Event Ledger
                </span>
              </div>
            </CardContent>
          </Card>
        </Link>

        {/* Service 2: Phone Live Camera */}
        <Link href="/camera" className="group block">
          <Card className="h-full border-gov-border bg-gov-deepnavy/90 group-hover:border-blue-500 group-hover:shadow-blue-500/10 group-hover:shadow-2xl transition-all duration-300 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/10 rounded-full blur-2xl group-hover:bg-blue-500/20 transition-all pointer-events-none" />
            <CardHeader className="space-y-3">
              <div className="w-12 h-12 rounded-2xl bg-blue-500/15 border border-blue-500/30 text-blue-400 flex items-center justify-center group-hover:scale-110 transition-transform">
                <Camera className="w-6 h-6" />
              </div>
              <div>
                <CardTitle className="text-lg text-white group-hover:text-blue-300 transition-colors flex items-center justify-between font-mono">
                  <span>Phone Camera Scan</span>
                  <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-blue-400 group-hover:translate-x-1 transition-all" />
                </CardTitle>
                <CardDescription className="mt-1 text-xs text-slate-400">
                  Field scanning using device rear camera with live HUD bounding boxes and continuous 450ms detection loop.
                </CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-1.5 text-[10px] font-mono text-slate-400">
                <span className="px-2 py-0.5 rounded bg-gov-panel border border-gov-border">
                  Rear / Front Switch
                </span>
                <span className="px-2 py-0.5 rounded bg-gov-panel border border-gov-border">
                  Torch & Zoom
                </span>
              </div>
            </CardContent>
          </Card>
        </Link>

        {/* Service 3: Photo & Video Upload */}
        <Link href="/upload" className="group block">
          <Card className="h-full border-gov-border bg-gov-deepnavy/90 group-hover:border-emerald-500 group-hover:shadow-emerald-500/10 group-hover:shadow-2xl transition-all duration-300 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 rounded-full blur-2xl group-hover:bg-emerald-500/20 transition-all pointer-events-none" />
            <CardHeader className="space-y-3">
              <div className="w-12 h-12 rounded-2xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 flex items-center justify-center group-hover:scale-110 transition-transform">
                <Upload className="w-6 h-6" />
              </div>
              <div>
                <CardTitle className="text-lg text-white group-hover:text-emerald-300 transition-colors flex items-center justify-between font-mono">
                  <span>Photo & Video Upload</span>
                  <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-emerald-400 group-hover:translate-x-1 transition-all" />
                </CardTitle>
                <CardDescription className="mt-1 text-xs text-slate-400">
                  Upload vehicle photos, dashcam video clips (MP4/MOV), or snap photos directly for multi-plate extraction.
                </CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-1.5 text-[10px] font-mono text-slate-400">
                <span className="px-2 py-0.5 rounded bg-gov-panel border border-gov-border">
                  Photos & Videos
                </span>
                <span className="px-2 py-0.5 rounded bg-gov-panel border border-gov-border">
                  Instant Camera Snap
                </span>
              </div>
            </CardContent>
          </Card>
        </Link>
      </div>

      {/* Trust, Security & Compliance Strip */}
      <div className="max-w-6xl mx-auto rounded-2xl border border-gov-border bg-gov-panel/70 p-4 sm:p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="p-2.5 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 shrink-0 mt-0.5">
            <Lock className="w-4 h-4" />
          </div>
          <div className="space-y-0.5">
            <h4 className="text-xs sm:text-sm font-bold text-white flex items-center gap-2">
              <span>Volatile Memory Architecture (DPDP Act 2023 Compliant)</span>
            </h4>
            <p className="text-xs text-slate-400 leading-relaxed max-w-2xl">
              All video frames and uploaded images are processed exclusively in volatile RAM and automatically purged. No vehicle media is stored on permanent disk storage.
            </p>
          </div>
        </div>
        <Link
          href="/about"
          className="text-xs text-gov-saffron hover:underline font-mono shrink-0 flex items-center gap-1 font-bold"
        >
          <span>Security Specs</span> &rarr;
        </Link>
      </div>

      {/* Indian Vehicle Registration Standards Quick Reference */}
      <div className="max-w-6xl mx-auto space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-bold text-white font-mono flex items-center gap-2">
            <FileCheck2 className="w-4 h-4 text-gov-saffron" />
            Standard Indian Registration Plate Formats
          </h3>
          <Link href="/about" className="text-xs text-gov-saffron hover:underline font-mono">
            View All Series &rarr;
          </Link>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
          <div className="p-3.5 rounded-xl bg-gov-deepnavy border border-gov-border space-y-1.5">
            <span className="text-[10px] font-mono text-gov-saffron font-bold uppercase">Standard Private</span>
            <div className="bg-white text-black font-black font-mono px-2 py-1 rounded text-center tracking-wider text-sm border border-slate-400">
              GJ 01 AB 1234
            </div>
            <p className="text-[11px] text-slate-400">White background with black alphanumeric text (2L-2D-2L-4D).</p>
          </div>

          <div className="p-3.5 rounded-xl bg-gov-deepnavy border border-gov-border space-y-1.5">
            <span className="text-[10px] font-mono text-blue-400 font-bold uppercase">Bharat (BH) Series</span>
            <div className="bg-white text-black font-black font-mono px-2 py-1 rounded text-center tracking-wider text-sm border border-slate-400">
              22 BH 1234 AA
            </div>
            <p className="text-[11px] text-slate-400">Pan-India seamless registration for defense & corporate fleet.</p>
          </div>

          <div className="p-3.5 rounded-xl bg-gov-deepnavy border border-gov-border space-y-1.5">
            <span className="text-[10px] font-mono text-emerald-400 font-bold uppercase">Electric Vehicle (EV)</span>
            <div className="bg-emerald-700 text-white font-black font-mono px-2 py-1 rounded text-center tracking-wider text-sm border border-emerald-500">
              MH 12 EV 9999
            </div>
            <p className="text-[11px] text-slate-400">Green background with white lettering for clean fuel vehicles.</p>
          </div>

          <div className="p-3.5 rounded-xl bg-gov-deepnavy border border-gov-border space-y-1.5">
            <span className="text-[10px] font-mono text-amber-400 font-bold uppercase">Commercial / Transport</span>
            <div className="bg-amber-400 text-black font-black font-mono px-2 py-1 rounded text-center tracking-wider text-sm border border-amber-600">
              DL 1C AB 5678
            </div>
            <p className="text-[11px] text-slate-400">Yellow background with black lettering for commercial carriers.</p>
          </div>
        </div>
      </div>

      {/* Recent Local Scans Ledger Widget */}
      {history.length > 0 && (
        <div className="max-w-6xl mx-auto space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white font-mono flex items-center gap-2">
              <HistoryIcon className="w-4 h-4 text-gov-saffron" />
              Recent Field Audit Scans
            </h3>
            <Link href="/history" className="text-xs text-gov-saffron hover:underline font-mono">
              View All ({history.length}) &rarr;
            </Link>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {history.slice(0, 3).map((item) => (
              <div
                key={item.id}
                className="p-3 rounded-xl border border-gov-border bg-gov-panel/80 flex items-center justify-between gap-3 shadow-md"
              >
                <div>
                  <span className="text-sm font-bold font-mono text-white block">
                    {item.detection.formatted_text || item.detection.normalized_text}
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">
                    {new Date(item.timestamp).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })} • {item.source}
                  </span>
                </div>
                <span className="text-xs font-mono font-bold text-gov-saffron bg-gov-navy px-2 py-0.5 rounded border border-gov-border">
                  {Math.round(item.detection.detection_confidence * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
