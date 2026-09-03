"use client";

import { useState, useEffect } from "react";
import { Globe, Clock, Sun, Moon, Eye, ShieldAlert } from "lucide-react";

export function OfficialUtilityBar() {
  const [fontSizeLevel, setFontSizeLevel] = useState<number>(0); // -1, 0, 1
  const [isHighContrast, setIsHighContrast] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<string>("");
  const [currentDate, setCurrentDate] = useState<string>("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      // Format in IST (Indian Standard Time)
      setCurrentTime(
        now.toLocaleTimeString("en-IN", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: true,
        })
      );
      setCurrentDate(
        now.toLocaleDateString("en-IN", {
          weekday: "short",
          day: "2-digit",
          month: "short",
          year: "numeric",
        })
      );
    };

    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleFontSize = (delta: number) => {
    let newLevel = 0;
    if (delta === 0) newLevel = 0;
    else newLevel = Math.max(-1, Math.min(2, fontSizeLevel + delta));

    setFontSizeLevel(newLevel);
    if (typeof document !== "undefined") {
      const basePx = 16 + newLevel * 2;
      document.documentElement.style.fontSize = `${basePx}px`;
    }
  };

  const toggleHighContrast = () => {
    setIsHighContrast((prev) => {
      const next = !prev;
      if (typeof document !== "undefined") {
        if (next) {
          document.documentElement.classList.add("high-contrast");
        } else {
          document.documentElement.classList.remove("high-contrast");
        }
      }
      return next;
    });
  };

  return (
    <aside aria-label="Accessibility & Official Portal Utility Bar" className="w-full bg-gov-deepnavy/95 border-b border-gov-border text-slate-300 text-[11px] font-sans">
      <div className="container mx-auto max-w-7xl px-4 sm:px-6 py-1.5 flex flex-wrap items-center justify-between gap-2">
        {/* Left: Portal Notice */}
        <div className="flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-gov-saffron animate-pulse" />
          <span className="font-medium text-slate-200 tracking-tight">
            National Automated Vehicle Vision & ANPR Utility
          </span>
          <span className="hidden md:inline text-slate-500">•</span>
          <span className="hidden md:inline text-slate-400 font-mono text-[10px]">
            HSRP RTO Pattern Standard 2024
          </span>
        </div>

        {/* Right: Accessibility Controls & IST Clock */}
        <div className="flex items-center gap-3 ml-auto">
          {/* Live IST Clock */}
          {currentTime && (
            <div className="hidden sm:flex items-center gap-1.5 text-gov-saffron font-mono bg-gov-navy/80 px-2 py-0.5 rounded border border-gov-border">
              <Clock className="w-3 h-3 text-gov-saffron" />
              <span>{currentDate} | {currentTime} IST</span>
            </div>
          )}

          {/* Text Size Accessibility Controls */}
          <div className="flex items-center gap-0.5 bg-gov-navy/80 rounded border border-gov-border p-0.5" title="Text Size Controls">
            <button
              type="button"
              onClick={() => handleFontSize(-1)}
              className={`px-1.5 py-0.5 rounded text-[10px] font-bold hover:bg-gov-border transition-colors ${
                fontSizeLevel === -1 ? "bg-gov-saffron text-gov-navy" : "text-slate-300"
              }`}
              title="Decrease Font Size"
            >
              A-
            </button>
            <button
              type="button"
              onClick={() => handleFontSize(0)}
              className={`px-1.5 py-0.5 rounded text-[10px] font-bold hover:bg-gov-border transition-colors ${
                fontSizeLevel === 0 ? "bg-gov-saffron text-gov-navy" : "text-slate-300"
              }`}
              title="Reset Font Size"
            >
              A
            </button>
            <button
              type="button"
              onClick={() => handleFontSize(1)}
              className={`px-1.5 py-0.5 rounded text-[10px] font-bold hover:bg-gov-border transition-colors ${
                fontSizeLevel >= 1 ? "bg-gov-saffron text-gov-navy" : "text-slate-300"
              }`}
              title="Increase Font Size"
            >
              A+
            </button>
          </div>

          {/* High Contrast Toggle */}
          <button
            type="button"
            onClick={toggleHighContrast}
            className={`flex items-center gap-1 px-2 py-0.5 rounded border text-[10px] font-mono transition-colors ${
              isHighContrast
                ? "bg-amber-400 text-black border-amber-400 font-bold"
                : "bg-gov-navy/80 text-slate-300 border-gov-border hover:border-gov-saffron"
            }`}
            title="Toggle High Contrast Mode"
          >
            <Eye className="w-3 h-3" />
            <span className="hidden sm:inline">Contrast</span>
          </button>

          {/* Language Indicator */}
          <div className="flex items-center gap-1 text-[11px] text-slate-300 font-medium">
            <Globe className="w-3 h-3 text-slate-400" />
            <span>English</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
