"use client";

import { useState } from "react";
import { ShieldCheck, Lock, EyeOff, Check } from "lucide-react";
import { Button } from "@/components/ui/button";

interface PrivacyConsentModalProps {
  isOpen: boolean;
  onConsent: (storeCrops: boolean) => void;
}

export function PrivacyConsentModal({ isOpen, onConsent }: PrivacyConsentModalProps) {
  const [optInCrops, setOptInCrops] = useState(false);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-navy-950/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="max-w-md w-full rounded-2xl border border-cyan-500/30 bg-navy-900 p-6 shadow-2xl text-slate-200 space-y-5">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight">Privacy & Authorized Use</h2>
            <p className="text-xs text-cyan-400 font-mono">PlateVision Security Protocol</p>
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs leading-relaxed">
          <strong>Important Notice:</strong> Vehicle registration numbers may be personal or sensitive information. Scan only vehicles you are authorized to process.
        </div>

        <div className="space-y-2.5 text-xs text-slate-300">
          <div className="flex items-start gap-2.5">
            <Lock className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span><strong>No Server Storage:</strong> Images and plate crops are processed transiently in-memory and are never permanently saved on our servers.</span>
          </div>
          <div className="flex items-start gap-2.5">
            <EyeOff className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
            <span><strong>Client-Side History:</strong> Scan metadata remains solely inside your browser. No surveillance or vehicle owner tracking is performed.</span>
          </div>
        </div>

        <label className="flex items-center gap-3 p-3 rounded-xl border border-slate-800 bg-navy-950/60 cursor-pointer hover:border-slate-700 transition-colors">
          <input
            type="checkbox"
            checked={optInCrops}
            onChange={(e) => setOptInCrops(e.target.checked)}
            className="w-4 h-4 rounded border-slate-700 text-cyan-500 focus:ring-cyan-400 bg-navy-900"
          />
          <span className="text-xs text-slate-300">
            Opt-in to store plate crop thumbnails locally in browser history (optional)
          </span>
        </label>

        <div className="pt-2">
          <Button
            variant="cyan"
            size="lg"
            className="w-full"
            onClick={() => onConsent(optInCrops)}
          >
            <Check className="w-4 h-4" />
            I Acknowledge & Agree
          </Button>
        </div>
      </div>
    </div>
  );
}
