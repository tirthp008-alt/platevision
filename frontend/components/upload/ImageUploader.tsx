"use client";

import { useState, useRef, ChangeEvent, DragEvent } from "react";
import {
  Upload,
  Camera,
  Image as ImageIcon,
  Film,
  Trash2,
  Scan,
  Loader2,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { SampleImages } from "./SampleImages";
import { detectImageFile, detectVideoFile } from "@/lib/api";
import { validateAndReadFile } from "@/lib/utils";
import { DetectionResponse } from "@/lib/types";
import { useLocalHistory } from "@/hooks/useLocalHistory";
import { DetectionResults } from "@/components/results/DetectionResults";

export function ImageUploader() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isVideo, setIsVideo] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [results, setResults] = useState<DetectionResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const videoInputRef = useRef<HTMLInputElement | null>(null);
  const cameraInputRef = useRef<HTMLInputElement | null>(null);
  const { addScanRecord } = useLocalHistory();

  const handleFile = async (file: File) => {
    setErrorMessage(null);
    setResults(null);

    const isVideoFile = file.type.startsWith("video/") || /\.(mp4|mov|avi|webm|mkv)$/i.test(file.name);
    setIsVideo(isVideoFile);

    if (isVideoFile) {
      // 50 MB video max limit
      if (file.size > 50 * 1024 * 1024) {
        setErrorMessage("Video file size exceeds 50 MB limit.");
        return;
      }
      const videoUrl = URL.createObjectURL(file);
      setSelectedFile(file);
      setPreviewUrl(videoUrl);
      return;
    }

    const { valid, error, dataUrl } = await validateAndReadFile(file, 15);
    if (!valid || !dataUrl) {
      setErrorMessage(error || "Invalid file.");
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(dataUrl);
  };

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    if (isVideo && previewUrl && previewUrl.startsWith("blob:")) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setIsVideo(false);
    setErrorMessage(null);
    setResults(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (videoInputRef.current) videoInputRef.current.value = "";
    if (cameraInputRef.current) cameraInputRef.current.value = "";
  };

  const handleProcess = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setErrorMessage(null);

    try {
      let response: DetectionResponse;
      if (isVideo) {
        response = await detectVideoFile(selectedFile);
      } else {
        response = await detectImageFile(selectedFile);
      }

      setResults(response);

      if (response.detections && response.detections.length > 0) {
        response.detections.forEach((det) => {
          addScanRecord(det, "upload");
        });
      }
    } catch (err: any) {
      console.error("Detection error:", err);
      setErrorMessage(err.message || "Failed to process vehicle media.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSelectSample = (file: File, dataUrl: string) => {
    setIsVideo(false);
    setSelectedFile(file);
    setPreviewUrl(dataUrl);
    setErrorMessage(null);
    setResults(null);
  };

  return (
    <div className="space-y-6 w-full max-w-4xl mx-auto pb-24">
      {/* Hidden File Inputs */}
      {/* 1. Images */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/heic,image/heif"
        onChange={onFileChange}
        className="hidden"
      />
      {/* 2. Videos */}
      <input
        ref={videoInputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/webm,video/x-msvideo,video/mkv"
        onChange={onFileChange}
        className="hidden"
      />
      {/* 3. Direct Camera Snap */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={onFileChange}
        className="hidden"
      />

      {/* Upload Box / Dropzone */}
      {!previewUrl && (
        <div
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          className={`relative rounded-3xl border-2 border-dashed p-8 sm:p-12 text-center transition-all duration-200 cursor-pointer ${
            isDragging
              ? "border-cyan-400 bg-cyan-500/10 scale-[1.01]"
              : "border-slate-800 bg-navy-900/60 hover:bg-navy-900 hover:border-slate-700"
          }`}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="flex flex-col items-center justify-center space-y-4 max-w-md mx-auto">
            <div className="p-4 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 shadow-lg shadow-cyan-500/10">
              <Upload className="w-8 h-8" />
            </div>

            <div className="space-y-1.5">
              <h3 className="text-lg font-bold text-white tracking-tight">
                Upload Vehicle Photo or Video
              </h3>
              <p className="text-xs sm:text-sm text-slate-400">
                Drag and drop your media file here, or select an input mode:
              </p>
            </div>

            {/* Direct Multi-Action Buttons */}
            <div className="flex flex-wrap gap-2.5 justify-center pt-2">
              <Button
                type="button"
                variant="cyan"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
              >
                <ImageIcon className="w-4 h-4" /> Browse Photo
              </Button>

              <Button
                type="button"
                variant="emerald"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  cameraInputRef.current?.click();
                }}
              >
                <Camera className="w-4 h-4" /> Camera Snap
              </Button>

              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  videoInputRef.current?.click();
                }}
              >
                <Film className="w-4 h-4" /> Upload Video
              </Button>
            </div>

            <span className="text-[11px] font-mono text-slate-500 block pt-2">
              Photos (JPEG, PNG, WEBP, HEIC up to 15MB) • Videos (MP4, MOV, WebM up to 50MB)
            </span>
          </div>
        </div>
      )}

      {/* Selected Media Preview & Process Bar */}
      {previewUrl && !results && (
        <div className="space-y-4">
          <div className="relative rounded-2xl bg-navy-950 border border-slate-800 p-2 overflow-hidden shadow-2xl flex items-center justify-center max-h-[60vh]">
            {isVideo ? (
              <video
                src={previewUrl}
                controls
                className="max-h-[55vh] w-auto object-contain rounded-xl"
              />
            ) : (
              <img
                src={previewUrl}
                alt="Uploaded Vehicle Preview"
                className="max-h-[55vh] w-auto object-contain rounded-xl"
              />
            )}

            {/* Remove / Change Floating Button */}
            <button
              onClick={handleClear}
              disabled={isProcessing}
              title="Remove media"
              className="absolute top-4 right-4 p-2 rounded-xl bg-navy-950/80 border border-slate-700 text-slate-300 hover:text-rose-400 hover:border-rose-500 transition-colors backdrop-blur-md z-10"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>

          {/* Action Row */}
          <div className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-2xl bg-navy-900 border border-slate-800">
            <div className="flex items-center gap-2 text-xs font-mono text-slate-300 truncate max-w-[200px] sm:max-w-xs">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span className="truncate">
                {selectedFile?.name || (isVideo ? "Video Clip" : "Sample Image")}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleClear}
                disabled={isProcessing}
              >
                Replace File
              </Button>

              <Button
                variant="cyan"
                size="default"
                onClick={handleProcess}
                disabled={isProcessing}
                className="min-w-[170px]"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin mr-2" />
                    <span>{isVideo ? "Scanning Video..." : "Detecting Plates..."}</span>
                  </>
                ) : (
                  <>
                    <Scan className="w-4 h-4 mr-2" />
                    <span>{isVideo ? "Analyze Video" : "Detect Number Plate"}</span>
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Error State Banner */}
      {errorMessage && (
        <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Instant Sample Vehicles Selector (when no file selected) */}
      {!previewUrl && !results && (
        <SampleImages onSelectSample={handleSelectSample} />
      )}

      {/* Results View */}
      {results && (
        <DetectionResults
          detections={results.detections}
          originalImageUrl={isVideo ? undefined : (previewUrl || undefined)}
          processingTimeMs={results.processing_time_ms}
          onUploadAnother={handleClear}
        />
      )}
    </div>
  );
}
