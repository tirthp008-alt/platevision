"use client";

import { CameraOff, ShieldAlert, Lock, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PermissionState } from "@/hooks/useCamera";

interface CameraPermissionHelpProps {
  permissionState: PermissionState;
  errorMessage: string | null;
  onRetry: () => void;
}

export function CameraPermissionHelp({
  permissionState,
  errorMessage,
  onRetry,
}: CameraPermissionHelpProps) {
  if (permissionState === "granted" || permissionState === "idle") {
    return null;
  }

  return (
    <div className="flex flex-col items-center justify-center p-6 text-center max-w-md mx-auto space-y-4 rounded-2xl border border-slate-800 bg-navy-900/90 shadow-2xl">
      <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-400">
        {permissionState === "denied" ? (
          <ShieldAlert className="w-10 h-10" />
        ) : (
          <CameraOff className="w-10 h-10" />
        )}
      </div>

      <div className="space-y-1.5">
        <h3 className="text-lg font-bold text-white">
          {permissionState === "denied"
            ? "Camera Access Required"
            : permissionState === "unsupported"
            ? "Camera Not Supported"
            : "Requesting Permission..."}
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          {errorMessage ||
            "Please grant camera access to scan vehicle number plates in real time."}
        </p>
      </div>

      {permissionState === "denied" && (
        <div className="p-3.5 rounded-xl bg-navy-950/70 border border-slate-800 text-left space-y-2 text-xs text-slate-300 w-full">
          <p className="font-semibold text-cyan-400 flex items-center gap-1.5">
            <Lock className="w-3.5 h-3.5" /> How to allow camera permission:
          </p>
          <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-400">
            <li>Tap the <strong>Lock / Tune icon</strong> in your browser address bar</li>
            <li>Change Camera permission to <strong>Allow</strong></li>
            <li>Ensure the site is loaded over <strong>HTTPS</strong> or localhost</li>
          </ul>
        </div>
      )}

      <Button variant="cyan" onClick={onRetry} className="w-full">
        <RefreshCw className="w-4 h-4 mr-2" />
        Retry Camera Access
      </Button>
    </div>
  );
}
