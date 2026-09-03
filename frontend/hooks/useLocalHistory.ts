"use client";

import { useCallback, useEffect, useState } from "react";
import { DetectionItem, ScanHistoryRecord } from "@/lib/types";

const STORAGE_KEY = "platevision_history_v1";
const CONSENT_KEY = "platevision_consent_v1";
const PREVIEW_STORE_KEY = "platevision_store_crops_v1";
const MAX_HISTORY_ITEMS = 50;

export function useLocalHistory() {
  const [history, setHistory] = useState<ScanHistoryRecord[]>([]);
  // Pre-grant consent by default per user instruction
  const [hasConsent, setHasConsent] = useState<boolean>(true);
  const [storeCropsLocally, setStoreCropsLocally] = useState<boolean>(true);
  const [isLoaded, setIsLoaded] = useState<boolean>(false);

  // Load from localStorage on mount
  useEffect(() => {
    try {
      localStorage.setItem(CONSENT_KEY, "true");
      localStorage.setItem(PREVIEW_STORE_KEY, "true");
      setHasConsent(true);
      setStoreCropsLocally(true);

      const storedHistory = localStorage.getItem(STORAGE_KEY);
      if (storedHistory) {
        setHistory(JSON.parse(storedHistory));
      }
    } catch (e) {
      console.warn("Could not load history from localStorage:", e);
    } finally {
      setIsLoaded(true);
    }
  }, []);

  const grantConsent = useCallback(() => {
    try {
      localStorage.setItem(CONSENT_KEY, "true");
      setHasConsent(true);
    } catch (e) {
      console.warn("Failed to save consent:", e);
    }
  }, []);

  const updateCropStorageOpt = useCallback((enable: boolean) => {
    try {
      localStorage.setItem(PREVIEW_STORE_KEY, enable ? "true" : "false");
      setStoreCropsLocally(enable);
    } catch (e) {
      console.warn("Failed to save crop storage opt:", e);
    }
  }, []);

  const addScanRecord = useCallback(
    (detection: DetectionItem, source: "camera" | "upload") => {
      const newRecord: ScanHistoryRecord = {
        id: detection.id || `rec_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
        timestamp: Date.now(),
        source,
        detection: { ...detection },
        crop_thumbnail: detection.crop_base64,
      };

      setHistory((prev) => {
        const filtered = prev.filter(
          (r) =>
            r.detection.normalized_text !== detection.normalized_text ||
            Date.now() - r.timestamp > 3000
        );
        const updated = [newRecord, ...filtered].slice(0, MAX_HISTORY_ITEMS);
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
        } catch (e) {
          console.warn("Failed to persist history:", e);
        }
        return updated;
      });
    },
    []
  );

  const deleteRecord = useCallback((id: string) => {
    setHistory((prev) => {
      const updated = prev.filter((r) => r.id !== id);
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      } catch (e) {
        console.warn("Failed to update history:", e);
      }
      return updated;
    });
  }, []);

  const clearAllHistory = useCallback(() => {
    setHistory([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      console.warn("Failed to clear history:", e);
    }
  }, []);

  const exportHistoryJSON = useCallback(() => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(history, null, 2));
    const a = document.createElement("a");
    a.href = dataStr;
    a.download = `platevision_history_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [history]);

  return {
    history,
    hasConsent,
    storeCropsLocally,
    isLoaded,
    grantConsent,
    updateCropStorageOpt,
    addScanRecord,
    deleteRecord,
    clearAllHistory,
    exportHistoryJSON,
  };
}
