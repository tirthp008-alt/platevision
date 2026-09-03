"use client";

import { Car, Sparkles, Layers } from "lucide-react";

export interface SampleImageOption {
  id: string;
  title: string;
  category: "Car" | "Multi-Car" | "BH-Series" | "SUV";
  expectedPlate: string;
  platesCount: number;
}

// Generate single or multiple vehicle scenes with distinct license plates
function generateSampleScene(sceneId: string): string {
  if (typeof window === "undefined") return "";

  const canvas = document.createElement("canvas");
  canvas.width = 1000;
  canvas.height = 600;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  // Background Sky & Road
  const grad = ctx.createLinearGradient(0, 0, 0, 600);
  grad.addColorStop(0, "#1e293b");
  grad.addColorStop(0.55, "#0f172a");
  grad.addColorStop(1, "#334155");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 1000, 600);

  // Asphalt road & lane stripes
  ctx.fillStyle = "#1e293b";
  ctx.fillRect(0, 420, 1000, 180);
  ctx.fillStyle = "#f8fafc";
  for (let i = 20; i < 1000; i += 140) {
    ctx.fillRect(i, 510, 70, 8);
  }

  const drawPlate = (x: number, y: number, w: number, h: number, text: string) => {
    // White plate background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x, y, w, h);

    // Blue IND stripe
    ctx.fillStyle = "#1e40af";
    ctx.fillRect(x, y, 18, h);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 8px monospace";
    ctx.fillText("IND", x + 1, y + h / 1.7);

    // Registration text
    ctx.fillStyle = "#000000";
    ctx.font = "bold 20px monospace";
    ctx.fillText(text, x + 26, y + h / 1.5);
  };

  const drawCar = (cx: number, cy: number, w: number, h: number, color: string, plateText: string) => {
    // Body
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(cx, cy, w, h, 18);
    ctx.fill();

    // Cabin / Windshield
    ctx.fillStyle = "#090d16";
    ctx.beginPath();
    ctx.roundRect(cx + w * 0.15, cy - h * 0.45, w * 0.7, h * 0.5, [20, 20, 0, 0]);
    ctx.fill();

    // Headlights
    ctx.fillStyle = "#fef08a";
    ctx.fillRect(cx + 15, cy + 20, w * 0.15, 20);
    ctx.fillRect(cx + w - w * 0.15 - 15, cy + 20, w * 0.15, 20);

    // Bumper Grill
    ctx.fillStyle = "#090d16";
    ctx.fillRect(cx + w * 0.2, cy + 35, w * 0.6, 50);

    // License plate in center of bumper
    const pw = Math.min(220, w * 0.55);
    const ph = 48;
    const px = cx + (w - pw) / 2;
    const py = cy + 42;
    drawPlate(px, py, pw, ph, plateText);
  };

  if (sceneId === "multi_traffic") {
    // Left Car (Blue Sedan)
    drawCar(80, 260, 380, 150, "#1d4ed8", "GJ 01 AB 1234");
    // Right Car (Grey SUV)
    drawCar(530, 240, 400, 170, "#475569", "MH 12 DE 1433");
  } else if (sceneId === "multi_parking") {
    // 3 Cars in parking row
    drawCar(40, 280, 280, 130, "#047857", "DL 01 CA 1234");
    drawCar(360, 270, 280, 140, "#b91c1c", "22 BH 1234 AA");
    drawCar(680, 280, 280, 130, "#334155", "KA 03 MN 4567");
  } else if (sceneId === "sample_dl") {
    drawCar(250, 220, 500, 190, "#0f172a", "DL 01 CA 1234");
  } else if (sceneId === "sample_bh") {
    drawCar(250, 220, 500, 190, "#047857", "22 BH 1234 AA");
  } else {
    drawCar(250, 220, 500, 190, "#2563eb", "GJ 01 AB 1234");
  }

  return canvas.toDataURL("image/jpeg", 0.94);
}

export const SAMPLE_VEHICLES: SampleImageOption[] = [
  {
    id: "multi_traffic",
    title: "Dual Traffic Scene",
    category: "Multi-Car",
    expectedPlate: "GJ01AB1234 + MH12DE1433",
    platesCount: 2,
  },
  {
    id: "multi_parking",
    title: "Triple Parking Scene",
    category: "Multi-Car",
    expectedPlate: "DL01 + 22BH + KA03 (3 Plates)",
    platesCount: 3,
  },
  {
    id: "sample_gj",
    title: "Gujarat Single Sedan",
    category: "Car",
    expectedPlate: "GJ 01 AB 1234",
    platesCount: 1,
  },
  {
    id: "sample_bh",
    title: "Bharat Series HSRP",
    category: "BH-Series",
    expectedPlate: "22 BH 1234 AA",
    platesCount: 1,
  },
];

interface SampleImagesProps {
  onSelectSample: (file: File, previewUrl: string) => void;
}

export function SampleImages({ onSelectSample }: SampleImagesProps) {
  const handleSelect = (sample: SampleImageOption) => {
    const dataUrl = generateSampleScene(sample.id);
    const arr = dataUrl.split(",");
    const mime = arr[0].match(/:(.*?);/)![1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    const file = new File([u8arr], `${sample.id}.jpg`, { type: mime });
    onSelectSample(file, dataUrl);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
        <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
        <span>Test instantly with single or multi-vehicle scenes:</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        {SAMPLE_VEHICLES.map((sample) => (
          <button
            key={sample.id}
            type="button"
            onClick={() => handleSelect(sample)}
            className={`flex flex-col items-start p-3 rounded-xl border transition-all text-left group ${
              sample.platesCount > 1
                ? "border-cyan-500/50 bg-cyan-950/20 hover:bg-cyan-950/40 shadow-md shadow-cyan-500/10"
                : "border-slate-800 bg-navy-900/60 hover:bg-navy-800 hover:border-slate-700"
            }`}
          >
            <div className="flex items-center justify-between w-full mb-1">
              <span className="text-[10px] font-mono text-cyan-400 font-bold uppercase flex items-center gap-1">
                {sample.platesCount > 1 ? (
                  <>
                    <Layers className="w-3 h-3 text-cyan-400" />
                    {sample.platesCount} Plates
                  </>
                ) : (
                  sample.category
                )}
              </span>
              <Car className="w-3.5 h-3.5 text-slate-400 group-hover:text-cyan-300 transition-colors" />
            </div>
            <span className="text-xs font-bold text-white tracking-tight line-clamp-1">
              {sample.title}
            </span>
            <span className="text-[11px] font-mono text-slate-400 mt-1">
              {sample.expectedPlate}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
