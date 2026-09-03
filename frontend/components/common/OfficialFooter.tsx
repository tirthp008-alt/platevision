"use client";

import Link from "next/link";
import { ShieldCheck, Lock, Layers, ExternalLink, FileText, CheckCircle2 } from "lucide-react";

export function OfficialFooter() {
  return (
    <footer className="w-full bg-gov-deepnavy border-t border-gov-border text-slate-400 text-xs mt-16 pb-20 md:pb-8">
      {/* Tricolor Stripe Accent at Footer Top */}
      <div className="tricolor-stripe" />

      <div className="container mx-auto max-w-7xl px-4 sm:px-6 py-10 space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Column 1: Portal Overview */}
          <div className="space-y-3 md:col-span-2">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gov-saffron text-gov-navy font-bold font-mono text-sm">
                PV
              </div>
              <div>
                <span className="text-base font-black text-white font-mono tracking-tight">PLATE<span className="text-gov-saffron">VISION</span></span>
                <span className="ml-1.5 rounded bg-gov-saffron/20 px-1.5 py-0.5 text-[10px] font-bold text-gov-saffron font-mono">ANPR</span>
              </div>
            </div>
            <p className="text-slate-400 leading-relaxed text-xs max-w-md">
              High-accuracy Edge AI and Deep OCR system for automated Indian and international vehicle registration plate detection, cropping, and validation across phone cameras, CCTV feeds, and traffic media.
            </p>
            <div className="flex flex-wrap gap-2 pt-1">
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-gov-panel border border-gov-border text-[10px] text-emerald-400 font-mono">
                <CheckCircle2 className="w-3 h-3 text-emerald-400" /> DPDP Act 2023 Compliant
              </span>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-gov-panel border border-gov-border text-[10px] text-gov-saffron font-mono">
                <Lock className="w-3 h-3 text-gov-saffron" /> 256-Bit RAM Processing
              </span>
            </div>
          </div>

          {/* Column 2: Navigation Links */}
          <div className="space-y-2.5">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200 font-mono border-b border-gov-border pb-1">
              Surveillance Services
            </h4>
            <ul className="space-y-1.5 text-xs">
              <li>
                <Link href="/cctv" className="hover:text-gov-saffron transition-colors flex items-center gap-1.5">
                  Live Traffic & CCTV Stream
                </Link>
              </li>
              <li>
                <Link href="/camera" className="hover:text-gov-saffron transition-colors flex items-center gap-1.5">
                  Phone Camera Live Scan
                </Link>
              </li>
              <li>
                <Link href="/upload" className="hover:text-gov-saffron transition-colors flex items-center gap-1.5">
                  Photo & Video File Upload
                </Link>
              </li>
              <li>
                <Link href="/history" className="hover:text-gov-saffron transition-colors flex items-center gap-1.5">
                  Local Audit History Ledger
                </Link>
              </li>
            </ul>
          </div>

          {/* Column 3: Standards & Formats */}
          <div className="space-y-2.5">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200 font-mono border-b border-gov-border pb-1">
              Registration Standards
            </h4>
            <ul className="space-y-1.5 text-xs font-mono text-slate-400">
              <li>• Standard State RTO: <span className="text-slate-300">GJ 01 AB 1234</span></li>
              <li>• Bharat Series: <span className="text-slate-300">22 BH 1234 AA</span></li>
              <li>• Electric Vehicles: <span className="text-slate-300">Green Plate HSRP</span></li>
              <li>• Commercial: <span className="text-slate-300">Yellow Plate HSRP</span></li>
              <li>
                <Link href="/about" className="text-gov-saffron hover:underline text-[11px] block pt-1 font-sans">
                  View Full Registration Format Specs &rarr;
                </Link>
              </li>
            </ul>
          </div>
        </div>

        {/* Disclaimer Box */}
        <div className="p-4 rounded-xl bg-gov-panel border border-gov-border text-[11px] leading-relaxed text-slate-400 space-y-1.5">
          <div className="flex items-center gap-2 text-slate-300 font-bold">
            <FileText className="w-3.5 h-3.5 text-gov-saffron" />
            <span>Public Portal Disclaimer & Information Notice:</span>
          </div>
          <p>
            PlateVision is an automated computer vision & ANPR utility built for authorized traffic surveillance, fleet inspection, parking management, and public computer-vision research. This interface is inspired by standard Indian public infrastructure portals for high clarity and accessibility; it is a non-governmental technology utility and does not represent or claim official government affiliation unless under authorized service contract.
          </p>
          <p className="text-slate-500 font-mono text-[10px]">
            Zero Permanent Disk Storage Policy: All uploaded images and live stream frames are processed strictly in volatile system RAM with instant memory eviction.
          </p>
        </div>

        {/* Bottom copyright line */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2 text-[11px] text-slate-500 border-t border-gov-border/60">
          <span>&copy; {new Date().getFullYear()} PlateVision ANPR Utility. All rights reserved.</span>
          <div className="flex items-center gap-3">
            <Link href="/about" className="hover:text-slate-300 transition-colors">Privacy Policy</Link>
            <span>•</span>
            <Link href="/about" className="hover:text-slate-300 transition-colors">Accessibility Statement</Link>
            <span>•</span>
            <span className="font-mono text-gov-saffron">v1.2.0 (Deep Neural ONNX)</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
