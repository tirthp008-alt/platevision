import * as React from "react";
import { cn } from "@/lib/utils";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "cyan" | "emerald";
  size?: "default" | "sm" | "lg" | "icon";
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", ...props }, ref) => {
    const variantClasses = {
      default: "bg-electric-500 text-white hover:bg-blue-600 shadow-md shadow-blue-500/20 active:scale-[0.98]",
      destructive: "bg-rose-600 text-white hover:bg-rose-700 shadow-md shadow-rose-600/20 active:scale-[0.98]",
      outline: "border border-slate-700 bg-transparent hover:bg-slate-800/60 text-slate-200 active:scale-[0.98]",
      secondary: "bg-navy-800 text-slate-200 hover:bg-navy-700 border border-slate-700/60 active:scale-[0.98]",
      ghost: "hover:bg-slate-800/40 text-slate-300 hover:text-white",
      cyan: "bg-cyan-500 text-navy-950 font-semibold hover:bg-cyan-400 shadow-md shadow-cyan-500/30 active:scale-[0.98]",
      emerald: "bg-emerald-600 text-white hover:bg-emerald-500 shadow-md shadow-emerald-600/30 active:scale-[0.98]",
    };

    const sizeClasses = {
      default: "h-11 px-5 py-2",
      sm: "h-9 px-3 text-xs rounded-lg",
      lg: "h-13 px-8 text-base rounded-xl font-medium",
      icon: "h-10 w-10 p-2 flex items-center justify-center rounded-xl",
    };

    return (
      <button
        className={cn(
          "inline-flex items-center justify-center gap-2 rounded-xl text-sm font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 disabled:pointer-events-none disabled:opacity-50",
          variantClasses[variant],
          sizeClasses[size],
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button };
