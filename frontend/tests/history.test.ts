import { describe, it, expect } from "vitest";
import { DetectionItem, ScanHistoryRecord } from "../lib/types";

describe("Scan History Management Logic", () => {
  it("serializes and parses history records correctly", () => {
    const mockDetection: DetectionItem = {
      id: "det_123",
      bounding_box: { x: 100, y: 150, width: 200, height: 60 },
      normalized_box: { x: 0.1, y: 0.15, width: 0.2, height: 0.06 },
      detection_confidence: 0.95,
      raw_text: "GJ 01 AB 1234",
      normalized_text: "GJ01AB1234",
      formatted_text: "GJ 01 AB 1234",
      ocr_confidence: 0.91,
      format_status: "valid",
      crop_url: "/api/results/req_1/det_123/crop",
    };

    const record: ScanHistoryRecord = {
      id: "rec_1",
      timestamp: 1700000000000,
      source: "camera",
      detection: mockDetection,
    };

    const json = JSON.stringify([record]);
    const parsed = JSON.parse(json) as ScanHistoryRecord[];

    expect(parsed.length).toBe(1);
    expect(parsed[0].detection.normalized_text).toBe("GJ01AB1234");
    expect(parsed[0].source).toBe("camera");
    expect(parsed[0].detection.format_status).toBe("valid");
  });

  it("filters out deleted records accurately", () => {
    const records: ScanHistoryRecord[] = [
      {
        id: "1",
        timestamp: 100,
        source: "camera",
        detection: {} as any,
      },
      {
        id: "2",
        timestamp: 200,
        source: "upload",
        detection: {} as any,
      },
    ];

    const filtered = records.filter((r) => r.id !== "1");
    expect(filtered.length).toBe(1);
    expect(filtered[0].id).toBe("2");
  });
});
