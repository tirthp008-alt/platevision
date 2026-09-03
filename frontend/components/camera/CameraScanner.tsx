"use client";

import { useEffect, useState } from "react";
import {
  Camera,
  Flashlight,
  FlashlightOff,
  FlipHorizontal,
  Play,
  Square,
  ZoomIn,
  ZoomOut,
  Activity,
  Maximize2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useCamera } from "@/hooks/useCamera";
import { useLiveDetection } from "@/hooks/useLiveDetection";
import { useLocalHistory } from "@/hooks/useLocalHistory";
import { ScannerOverlay } from "./ScannerOverlay";
import { CameraPermissionHelp } from "./CameraPermissionHelp";
import { DetectionResponse } from "@/lib/types";
import { DetectionResults } from "@/components/results/DetectionResults";

export function CameraScanner() {
  const {
    videoRef,
    isStreaming,
    permissionState,
    facingMode,
    hasTorch,
    isTorchOn,
    zoomRange,
    errorMessage,
    startCamera,
    stopCamera,
    toggleCamera,
    toggleTorch,
    setZoom,
    captureFrame,
  } = useCamera();

  const { addScanRecord } = useLocalHistory();
  const [activeResult, setActiveResult] = useState<DetectionResponse | null>(null);
  const [capturedImageDataUrl, setCapturedImageDataUrl] = useState<string | null>(null);

  const {
    isLiveEnabled,
    setIsLiveEnabled,
    status,
    statusText,
    lastDetections,
    isProcessing,
    isLocked,
    lockedResult,
    resumeScanning,
    triggerManualScan,
  } = useLiveDetection({
    isStreaming,
    captureFrame,
    intervalMs: 450,
    confidenceThreshold: 0.65,
    autoLockOnHighConfidence: false,
    onDetection: (response) => {
      if (response.detections.length > 0) {
        response.detections.forEach((det) => {
          addScanRecord(det, "camera");
        });
      }
    },
  });

  // Auto-start camera when page opens
  useEffect(() => {
    startCamera("environment");
    return () => {
      stopCamera();
    };
  }, []);

  const handleManualCapture = async () => {
    const frame = captureFrame(1600);
    if (frame) {
      setCapturedImageDataUrl(frame);
      await triggerManualScan();
    }
  };

  const handleInspectResults = (response: DetectionResponse) => {
    setActiveResult(response);
  };

  return (
    <div className="flex flex-col gap-5 w-full max-w-4xl mx-auto pb-24">
      {/* Scanner Viewport Box */}
      <div className="relative w-full aspect-[4/3] sm:aspect-[16/9] max-h-[70vh] bg-navy-950 rounded-2xl sm:rounded-3xl border border-slate-800/80 overflow-hidden shadow-2xl flex items-center justify-center">
        {/* Video Element */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover transition-opacity duration-300 ${
            isStreaming ? "opacity-100" : "opacity-0"
          }`}
        />

        {/* Permission Help & Fallbacks */}
        {permissionState !== "granted" && (
          <div className="absolute inset-0 flex items-center justify-center p-4 bg-navy-950/90 z-20">
            <CameraPermissionHelp
              permissionState={permissionState}
              errorMessage={errorMessage}
              onRetry={() => startCamera(facingMode)}
            />
          </div>
        )}

        {/* Live HUD Overlay when streaming */}
        {isStreaming && (
          <ScannerOverlay
            status={status}
            statusText={statusText}
            detections={lastDetections}
            isScanning={isLiveEnabled || isProcessing}
          />
        )}

        {/* Top Floating Controls on Video */}
        {isStreaming && (
          <div className="absolute top-3 right-3 flex items-center gap-2 z-30">
            {hasTorch && (
              <button
                onClick={toggleTorch}
                title="Toggle Flash"
                className={`p-2.5 rounded-xl border backdrop-blur-md transition-colors ${
                  isTorchOn
                    ? "bg-amber-500/30 border-amber-400 text-amber-300"
                    : "bg-navy-900/70 border-slate-700 text-slate-300 hover:text-white"
                }`}
              >
                {isTorchOn ? <Flashlight className="w-5 h-5" /> : <FlashlightOff className="w-5 h-5" />}
              </button>
            )}

            <button
              onClick={toggleCamera}
              title="Flip Camera (Front/Rear)"
              className="p-2.5 rounded-xl border border-slate-700 bg-navy-900/70 backdrop-blur-md text-slate-300 hover:text-white transition-colors"
            >
              <FlipHorizontal className="w-5 h-5" />
            </button>
          </div>
        )}

        {/* Zoom Slider Overlay if supported */}
        {isStreaming && zoomRange && (
          <div className="absolute bottom-3 left-4 right-4 sm:left-auto sm:right-4 z-30 flex items-center gap-2 bg-navy-900/80 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-700">
            <ZoomOut className="w-4 h-4 text-slate-400" />
            <input
              type="range"
              min={zoomRange.min}
              max={zoomRange.max}
              step={zoomRange.step}
              value={zoomRange.current}
              onChange={(e) => setZoom(parseFloat(e.target.value))}
              className="w-24 sm:w-32 accent-cyan-400"
            />
            <ZoomIn className="w-4 h-4 text-slate-400" />
            <span className="text-[10px] font-mono text-cyan-300">{zoomRange.current.toFixed(1)}x</span>
          </div>
        )}
      </div>

      {/* Main Scanner Control Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-navy-900/90 p-4 rounded-2xl border border-slate-800 shadow-xl backdrop-blur-md">
        {/* Toggle Live Scan */}
        <Button
          variant={isLiveEnabled ? "cyan" : "secondary"}
          onClick={() => setIsLiveEnabled(!isLiveEnabled)}
          disabled={!isStreaming}
          className="w-full"
        >
          <Activity className="w-4 h-4" />
          <span>{isLiveEnabled ? "Live: ON" : "Live: PAUSED"}</span>
        </Button>

        {/* Manual Capture Frame */}
        <Button
          variant="default"
          onClick={handleManualCapture}
          disabled={!isStreaming || isProcessing}
          className="w-full"
        >
          <Camera className="w-4 h-4" />
          <span>Capture Plate</span>
        </Button>

        {/* Start / Stop Camera */}
        {isStreaming ? (
          <Button
            variant="destructive"
            onClick={stopCamera}
            className="w-full"
          >
            <Square className="w-4 h-4" />
            <span>Stop Camera</span>
          </Button>
        ) : (
          <Button
            variant="emerald"
            onClick={() => startCamera(facingMode)}
            className="w-full"
          >
            <Play className="w-4 h-4" />
            <span>Start Camera</span>
          </Button>
        )}

        {/* Reset / Resume */}
        <Button
          variant="outline"
          onClick={resumeScanning}
          disabled={!isLocked}
          className="w-full"
        >
          <span>Resume Stream</span>
        </Button>
      </div>

      {/* Real-time Detection Result Drawer / View */}
      {lastDetections.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
              Recognized License Plates ({lastDetections.length})
            </h3>
            <span className="text-xs font-mono text-cyan-400 bg-cyan-950/60 px-2.5 py-1 rounded-full border border-cyan-800">
              Live Feed
            </span>
          </div>

          <DetectionResults
            detections={lastDetections}
            originalImageUrl={capturedImageDataUrl || undefined}
            onRescan={resumeScanning}
          />
        </div>
      )}
    </div>
  );
}
