import * as React from "react";
import { cn } from "@/lib/utils";

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary" | "destructive" | "outline" | "cyan" | "emerald" | "amber";
}

function Badge({ className, variant = "default", ...props }: BadgeProps) {
  const variantClasses = {
    default: "border-transparent bg-electric-500/20 text-blue-300 border border-blue-500/40",
    secondary: "border-transparent bg-navy-800 text-slate-300 border border-slate-700",
    destructive: "border-transparent bg-rose-500/20 text-rose-300 border border-rose-500/40",
    outline: "text-slate-300 border border-slate-700",
    cyan: "border-cyan-500/40 bg-cyan-500/20 text-cyan-300",
    emerald: "border-emerald-500/40 bg-emerald-500/20 text-emerald-300",
    amber: "border-amber-500/40 bg-amber-500/20 text-amber-300",
  };

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold tracking-wide transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        variantClasses[variant],
        className
      )}
      {...props}
    />
  );
}

export { Badge };
