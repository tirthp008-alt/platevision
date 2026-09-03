"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type CameraFacingMode = "environment" | "user";
export type PermissionState = "idle" | "requesting" | "granted" | "denied" | "unsupported";

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [permissionState, setPermissionState] = useState<PermissionState>("idle");
  const [facingMode, setFacingMode] = useState<CameraFacingMode>("environment");
  const [isStreaming, setIsStreaming] = useState(false);
  const [hasTorch, setHasTorch] = useState(false);
  const [isTorchOn, setIsTorchOn] = useState(false);
  const [zoomRange, setZoomRange] = useState<{ min: number; max: number; step: number; current: number } | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (e) {}
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setIsTorchOn(false);
    setHasTorch(false);
    setZoomRange(null);
  }, []);

  const startCamera = useCallback(
    async (targetFacingMode: CameraFacingMode = facingMode) => {
      setErrorMessage(null);
      if (typeof window === "undefined" || !navigator.mediaDevices?.getUserMedia) {
        setPermissionState("unsupported");
        setErrorMessage("Camera access is not supported by this browser. Please use Chrome, Safari, or Edge.");
        return;
      }

      setPermissionState("requesting");
      stopCamera();

      try {
        const constraints: MediaStreamConstraints = {
          video: {
            facingMode: { ideal: targetFacingMode },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
          audio: false,
        };

        let stream: MediaStream;
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (err: any) {
          // Fallback to basic video constraint without facingMode if specific request failed
          loggerFallback: {
            try {
              stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
              break loggerFallback;
            } catch (fallbackErr: any) {
              throw err;
            }
          }
        }

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // Ensure playsinline for iOS Safari
          videoRef.current.setAttribute("playsinline", "true");
          videoRef.current.setAttribute("webkit-playsinline", "true");
          await videoRef.current.play();
        }

        const videoTrack = stream.getVideoTracks()[0];
        if (videoTrack) {
          const capabilities: any = typeof videoTrack.getCapabilities === "function" ? videoTrack.getCapabilities() : {};

          // Check Torch / Flash support
          if (capabilities.torch) {
            setHasTorch(true);
          }

          // Check Zoom support
          if (capabilities.zoom) {
            setZoomRange({
              min: capabilities.zoom.min || 1,
              max: capabilities.zoom.max || 5,
              step: capabilities.zoom.step || 0.1,
              current: 1,
            });
          }
        }

        setFacingMode(targetFacingMode);
        setPermissionState("granted");
        setIsStreaming(true);
      } catch (err: any) {
        console.error("Camera access error:", err);
        stopCamera();
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
          setPermissionState("denied");
          setErrorMessage("Camera permission was denied. Please allow camera access in browser settings.");
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
          setPermissionState("unsupported");
          setErrorMessage("No camera device found on this system.");
        } else {
          setPermissionState("denied");
          setErrorMessage(`Unable to access camera: ${err.message || "Unknown error"}`);
        }
      }
    },
    [facingMode, stopCamera]
  );

  const toggleCamera = useCallback(async () => {
    const nextFacingMode = facingMode === "environment" ? "user" : "environment";
    await startCamera(nextFacingMode);
  }, [facingMode, startCamera]);

  const toggleTorch = useCallback(async () => {
    if (!streamRef.current || !hasTorch) return;
    const track = streamRef.current.getVideoTracks()[0];
    if (track) {
      try {
        const nextState = !isTorchOn;
        await (track as any).applyConstraints({
          advanced: [{ torch: nextState }],
        });
        setIsTorchOn(nextState);
      } catch (e) {
        console.warn("Failed to toggle torch:", e);
      }
    }
  }, [hasTorch, isTorchOn]);

  const setZoom = useCallback(
    async (zoomValue: number) => {
      if (!streamRef.current || !zoomRange) return;
      const track = streamRef.current.getVideoTracks()[0];
      if (track) {
        try {
          await (track as any).applyConstraints({
            advanced: [{ zoom: zoomValue }],
          });
          setZoomRange((prev) => (prev ? { ...prev, current: zoomValue } : null));
        } catch (e) {
          console.warn("Failed to set zoom:", e);
        }
      }
    },
    [zoomRange]
  );

  const captureFrame = useCallback(
    (maxDimension: number = 1280): string | null => {
      if (!videoRef.current || !isStreaming) return null;
      const video = videoRef.current;
      const w = video.videoWidth;
      const h = video.videoHeight;
      if (w === 0 || h === 0) return null;

      const canvas = document.createElement("canvas");
      let outW = w;
      let outH = h;
      if (Math.max(w, h) > maxDimension) {
        if (w > h) {
          outW = maxDimension;
          outH = Math.round((h * maxDimension) / w);
        } else {
          outH = maxDimension;
          outW = Math.round((w * maxDimension) / h);
        }
      }

      canvas.width = outW;
      canvas.height = outH;
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(video, 0, 0, outW, outH);
      return canvas.toDataURL("image/jpeg", 0.88);
    },
    [isStreaming]
  );

  // Stop camera when unmounting
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return {
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
  };
}
