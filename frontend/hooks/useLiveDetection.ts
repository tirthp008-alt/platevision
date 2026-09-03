"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { detectFrameBase64 } from "@/lib/api";
import { DetectionItem, DetectionResponse } from "@/lib/types";
import { triggerVibration } from "@/lib/utils";

export type ScannerStatus =
  | "idle"
  | "align_plate"
  | "scanning"
  | "plate_detected"
  | "low_confidence"
  | "no_plate"
  | "error";

interface UseLiveDetectionProps {
  isStreaming: boolean;
  captureFrame: (maxDimension?: number) => string | null;
  intervalMs?: number;
  confidenceThreshold?: number;
  autoLockOnHighConfidence?: boolean;
  onDetection?: (response: DetectionResponse) => void;
}

export function useLiveDetection({
  isStreaming,
  captureFrame,
  intervalMs = 450,
  confidenceThreshold = 0.65,
  autoLockOnHighConfidence = false,
  onDetection,
}: UseLiveDetectionProps) {
  const [isLiveEnabled, setIsLiveEnabled] = useState(true);
  const [status, setStatus] = useState<ScannerStatus>("align_plate");
  const [statusText, setStatusText] = useState<string>("Align vehicle plate in frame");
  const [lastDetections, setLastDetections] = useState<DetectionItem[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isLocked, setIsLocked] = useState(false);
  const [lockedResult, setLockedResult] = useState<DetectionResponse | null>(null);

  const processingRef = useRef(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const updateStatus = useCallback(
    (newStatus: ScannerStatus, customText?: string) => {
      setStatus(newStatus);
      if (customText) {
        setStatusText(customText);
        return;
      }
      switch (newStatus) {
        case "align_plate":
          setStatusText("Align vehicle plate in frame");
          break;
        case "scanning":
          setStatusText("Scanning for number plates...");
          break;
        case "plate_detected":
          setStatusText("Plate detected");
          break;
        case "low_confidence":
          setStatusText("Plate detected (Low confidence)");
          break;
        case "no_plate":
          setStatusText("No plate detected");
          break;
        case "error":
          setStatusText("Processing error");
          break;
        default:
          setStatusText("Ready");
      }
    },
    []
  );

  const runDetectionCycle = useCallback(async () => {
    if (!isStreaming || !isLiveEnabled || isLocked || processingRef.current) {
      return;
    }

    // Check document visibility
    if (typeof document !== "undefined" && document.hidden) {
      return;
    }

    const frameBase64 = captureFrame(800);
    if (!frameBase64) return;

    processingRef.current = true;
    setIsProcessing(true);
    updateStatus("scanning");

    abortControllerRef.current = new AbortController();

    try {
      const response = await detectFrameBase64(
        frameBase64,
        abortControllerRef.current.signal
      );

      if (response && response.detections && response.detections.length > 0) {
        setLastDetections(response.detections);
        const topDetection = response.detections[0];
        const combinedConf = (topDetection.detection_confidence + topDetection.ocr_confidence) / 2;

        if (combinedConf >= confidenceThreshold) {
          updateStatus("plate_detected", `Plate detected: ${topDetection.formatted_text || topDetection.normalized_text}`);
          triggerVibration(80);

          if (autoLockOnHighConfidence && combinedConf >= 0.85) {
            setIsLocked(true);
            setLockedResult(response);
          }
        } else {
          updateStatus("low_confidence", "Plate detected (Low confidence)");
        }

        if (onDetection) {
          onDetection(response);
        }
      } else {
        setLastDetections([]);
        updateStatus("no_plate", "Searching for vehicle plates...");
      }
    } catch (err: any) {
      if (err.name !== "AbortError") {
        console.warn("Live detection request failed:", err);
        updateStatus("error", err.message || "Detection failed");
      }
    } finally {
      processingRef.current = false;
      setIsProcessing(false);
    }
  }, [
    isStreaming,
    isLiveEnabled,
    isLocked,
    captureFrame,
    updateStatus,
    confidenceThreshold,
    autoLockOnHighConfidence,
    onDetection,
  ]);

  // Main polling loop
  useEffect(() => {
    if (!isStreaming || !isLiveEnabled || isLocked) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    timerRef.current = setInterval(() => {
      runDetectionCycle();
    }, intervalMs);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [isStreaming, isLiveEnabled, isLocked, intervalMs, runDetectionCycle]);

  // Pause when tab is inactive
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden && abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);

  const resumeScanning = useCallback(() => {
    setIsLocked(false);
    setLockedResult(null);
    setLastDetections([]);
    updateStatus("align_plate");
  }, [updateStatus]);

  return {
    isLiveEnabled,
    setIsLiveEnabled,
    status,
    statusText,
    lastDetections,
    isProcessing,
    isLocked,
    lockedResult,
    resumeScanning,
    triggerManualScan: runDetectionCycle,
  };
}
