"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Camera, Upload, History, Home, Cctv } from "lucide-react";
import { cn } from "@/lib/utils";

const MOBILE_NAV = [
  { label: "Dashboard", href: "/", icon: Home },
  { label: "Traffic Stream", href: "/cctv", icon: Cctv },
  { label: "Phone Cam", href: "/camera", icon: Camera },
  { label: "Upload", href: "/upload", icon: Upload },
  { label: "Audit Ledger", href: "/history", icon: History },
];

export function BottomNav() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Mobile Navigation Bar"
      className="fixed bottom-0 left-0 right-0 z-40 bg-gov-navy/95 backdrop-blur-lg border-t border-gov-border lg:hidden safe-area-pb shadow-2xl"
    >
      <div className="flex items-center justify-around h-16 max-w-lg mx-auto px-2">
        {MOBILE_NAV.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex flex-col items-center justify-center flex-1 py-1 transition-colors relative",
                isActive
                  ? "text-gov-saffron font-bold"
                  : "text-slate-400 hover:text-slate-200"
              )}
            >
              <div
                className={cn(
                  "p-1 rounded-xl transition-all",
                  isActive ? "bg-gov-saffron/15 text-gov-saffron" : ""
                )}
              >
                <Icon className="w-5 h-5" />
              </div>
              <span className="text-[10px] mt-0.5 font-medium">{item.label}</span>
              {isActive && (
                <div className="absolute top-0 w-8 h-0.5 bg-gov-saffron rounded-full" />
              )}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
