"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Camera, Upload, History, Info, Scan, Radio, Cctv, ShieldCheck } from "lucide-react";
import { StatusIndicator } from "./StatusIndicator";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { label: "Live Traffic Stream", href: "/cctv", icon: Cctv, badge: "LIVE" },
  { label: "Phone Camera", href: "/camera", icon: Camera },
  { label: "Upload Photo/Video", href: "/upload", icon: Upload },
  { label: "Audit Ledger", href: "/history", icon: History },
  { label: "Standards & Specs", href: "/about", icon: Info },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 w-full bg-gov-navy/95 border-b border-gov-border backdrop-blur-md shadow-lg">
      {/* Indian National Tricolor Ribbon at Header Top */}
      <div className="tricolor-stripe" />

      <div className="container mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6">
        {/* Brand Header */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-gov-saffron via-orange-600 to-amber-500 text-gov-navy font-extrabold shadow-md shadow-orange-500/20 group-hover:scale-105 transition-transform">
            <Scan className="h-5 w-5 text-gov-navy" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="text-lg font-black tracking-tight text-white font-mono">PLATE<span className="text-gov-saffron">VISION</span></span>
              <span className="rounded bg-gov-saffron/20 border border-gov-saffron/40 px-1.5 py-0.5 text-[9px] font-bold text-gov-saffron font-mono tracking-wider">
                ANPR PORTAL
              </span>
            </div>
            <p className="text-[10px] text-slate-400 -mt-0.5 hidden sm:block font-medium">
              National Traffic Vision & License Plate Recognition
            </p>
          </div>
        </Link>

        {/* Desktop Navigation Links */}
        <nav className="hidden lg:flex items-center gap-1 bg-gov-panel/80 p-1 rounded-xl border border-gov-border">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold tracking-tight transition-all duration-150 relative",
                  isActive
                    ? "bg-gov-saffron text-gov-navy shadow-sm font-bold"
                    : "text-slate-300 hover:text-white hover:bg-gov-border/60"
                )}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{item.label}</span>
                {item.badge && !isActive && (
                  <span className="px-1 py-0.2 rounded bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 text-[9px] font-mono animate-pulse">
                    {item.badge}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>

        {/* Right tools */}
        <div className="flex items-center gap-3">
          <StatusIndicator />
        </div>
      </div>
    </header>
  );
}
