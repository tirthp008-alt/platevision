/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./app/**/*.{ts,tsx}",
    "./src/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        // Indian Government Portal Theme Palette
        gov: {
          navy: "#0A192F",
          deepnavy: "#0F2B5C",
          blue: "#1E3A8A",
          panel: "#13233F",
          border: "#1E3A5F",
          saffron: "#FF9933",
          saffronDark: "#EA580C",
          green: "#138808",
          greenDark: "#0E6606",
          gold: "#D97706",
          paper: "#F8FAFC",
          textMuted: "#94A3B8",
        },
        navy: {
          950: "#070C18",
          900: "#0B132B",
          850: "#131C38",
          800: "#1C2541",
          700: "#2C3D66",
          600: "#3A506B",
        },
        electric: {
          500: "#0070F3",
          400: "#38BDF8",
        },
        scanner: {
          cyan: "#00F0FF",
          saffron: "#FF9933",
          glow: "rgba(255, 153, 51, 0.4)",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        scan: {
          "0%": { top: "0%" },
          "50%": { top: "96%" },
          "100%": { top: "0%" },
        },
        pulseGlow: {
          "0%, 100%": { opacity: "0.4" },
          "50%": { opacity: "0.9" },
        },
      },
      animation: {
        scan: "scan 2.5s ease-in-out infinite",
        pulseGlow: "pulseGlow 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
