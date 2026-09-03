import {
  ShieldCheck,
  Lock,
  EyeOff,
  Cpu,
  CheckCircle2,
  AlertTriangle,
  FileCode2,
  Globe,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata = {
  title: "About & Privacy - PlateVision",
  description: "PlateVision architecture, privacy standards, and Indian ANPR format guide",
};

export default function AboutPage() {
  return (
    <div className="space-y-8 max-w-4xl mx-auto pb-24">
      {/* Top Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          About <span className="text-cyan-400 font-mono">PlateVision</span>
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 max-w-xl mx-auto">
          High-performance Automated Number Plate Recognition (ANPR) designed for privacy, speed, and real-world robustness.
        </p>
      </div>

      {/* Privacy Notice Card */}
      <div className="rounded-2xl border border-cyan-500/40 bg-navy-900/90 p-5 sm:p-6 space-y-4 shadow-xl">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">Privacy & Responsible Use Guidelines</h2>
            <p className="text-xs text-cyan-300 font-mono">Zero-Retention Security Architecture</p>
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs leading-relaxed font-medium">
          “Vehicle registration numbers may be personal or sensitive information. Scan only vehicles you are authorized to process.”
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-300 pt-1">
          <div className="p-3 rounded-xl bg-navy-950/60 border border-slate-800 space-y-1">
            <span className="font-bold text-white flex items-center gap-1.5">
              <Lock className="w-3.5 h-3.5 text-emerald-400" />
              In-Memory Transient Processing
            </span>
            <p className="text-slate-400 text-[11px] leading-relaxed">
              Images received by the backend are processed exclusively in RAM and are never permanently persisted to disk or databases.
            </p>
          </div>

          <div className="p-3 rounded-xl bg-navy-950/60 border border-slate-800 space-y-1">
            <span className="font-bold text-white flex items-center gap-1.5">
              <EyeOff className="w-3.5 h-3.5 text-cyan-400" />
              No Tracking or Surveillance
            </span>
            <p className="text-slate-400 text-[11px] leading-relaxed">
              PlateVision contains no owner lookup databases, facial recognition, geolocation tracking, or centralized logging.
            </p>
          </div>
        </div>
      </div>

      {/* Indian Plate Format Guide */}
      <Card className="border-slate-800 bg-navy-900/80">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Globe className="w-5 h-5 text-cyan-400" />
            Indian Vehicle Registration Formats
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-xs text-slate-300">
          <p className="leading-relaxed">
            PlateVision features position-aware character normalization customized for Indian High Security Registration Plates (HSRP):
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="p-3 rounded-xl bg-navy-950 border border-slate-800 space-y-1.5">
              <span className="font-mono text-cyan-300 font-bold text-sm">Standard State Format</span>
              <p className="font-mono text-white text-xs bg-navy-900 px-2 py-1 rounded border border-slate-800">
                GJ 01 AB 1234 / MH 12 DE 1433
              </p>
              <ul className="list-disc list-inside text-[11px] text-slate-400 space-y-0.5 pt-1">
                <li><strong>SS (2 Letters):</strong> State/UT Code (e.g. GJ, DL, MH, KA)</li>
                <li><strong>DD (2 Digits):</strong> District RTO Office</li>
                <li><strong>LL (1-3 Letters):</strong> Series identifier</li>
                <li><strong>NNNN (4 Digits):</strong> Vehicle registration sequence</li>
              </ul>
            </div>

            <div className="p-3 rounded-xl bg-navy-950 border border-slate-800 space-y-1.5">
              <span className="font-mono text-cyan-300 font-bold text-sm">Bharat Series (BH)</span>
              <p className="font-mono text-white text-xs bg-navy-900 px-2 py-1 rounded border border-slate-800">
                22 BH 1234 AA
              </p>
              <ul className="list-disc list-inside text-[11px] text-slate-400 space-y-0.5 pt-1">
                <li><strong>YY (2 Digits):</strong> Year of registration (e.g. 22, 23, 24)</li>
                <li><strong>BH (2 Letters):</strong> Bharat national series code</li>
                <li><strong>NNNN (4 Digits):</strong> Randomized 4-digit number</li>
                <li><strong>LL (2 Letters):</strong> Running series suffix</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Architecture Overview */}
      <Card className="border-slate-800 bg-navy-900/80">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Cpu className="w-5 h-5 text-cyan-400" />
            Computer Vision Pipeline Architecture
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-xs text-slate-300">
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-center">
            <div className="p-3 rounded-xl bg-navy-950 border border-slate-800">
              <span className="text-cyan-400 font-mono font-bold block">01</span>
              <span className="font-bold text-white block mt-1">Ingestion</span>
              <span className="text-[10px] text-slate-400">EXIF Transpose & Safe Decode</span>
            </div>
            <div className="p-3 rounded-xl bg-navy-950 border border-slate-800">
              <span className="text-cyan-400 font-mono font-bold block">02</span>
              <span className="font-bold text-white block mt-1">Detection</span>
              <span className="text-[10px] text-slate-400">ONNX YOLO / Heuristic Provider</span>
            </div>
            <div className="p-3 rounded-xl bg-navy-950 border border-slate-800">
              <span className="text-cyan-400 font-mono font-bold block">03</span>
              <span className="font-bold text-white block mt-1">Cropping</span>
              <span className="text-[10px] text-slate-400">Padding & Boundary Clamp</span>
            </div>
            <div className="p-3 rounded-xl bg-navy-950 border border-slate-800">
              <span className="text-cyan-400 font-mono font-bold block">04</span>
              <span className="font-bold text-white block mt-1">Multi-OCR</span>
              <span className="text-[10px] text-slate-400">CLAHE + Otsu Variant Scoring</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
