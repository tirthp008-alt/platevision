import { describe, it, expect } from "vitest";
import { formatPercentage, getConfidenceColor, getFormatStatusDetails } from "../lib/utils";

describe("Frontend Utility Functions", () => {
  it("formats percentages correctly", () => {
    expect(formatPercentage(0.942)).toBe("94%");
    expect(formatPercentage(0.5)).toBe("50%");
    expect(formatPercentage(1.0)).toBe("100%");
    expect(formatPercentage(0)).toBe("0%");
  });

  it("assigns appropriate confidence colors", () => {
    const high = getConfidenceColor(0.92);
    expect(high.badgeText).toContain("emerald");

    const medium = getConfidenceColor(0.72);
    expect(medium.badgeText).toContain("amber");

    const low = getConfidenceColor(0.45);
    expect(low.badgeText).toContain("rose");
  });

  it("returns correct format status badge details", () => {
    const valid = getFormatStatusDetails("valid");
    expect(valid.label).toBe("Standard Format Verified");
    expect(valid.colorClass).toContain("emerald");

    const possible = getFormatStatusDetails("possible");
    expect(possible.label).toBe("Likely Plate Format");
    expect(possible.colorClass).toContain("cyan");

    const uncertain = getFormatStatusDetails("uncertain");
    expect(uncertain.label).toBe("Uncertain Format");
    expect(uncertain.colorClass).toContain("amber");
  });
});
