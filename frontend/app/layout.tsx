import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { OfficialUtilityBar } from "@/components/common/OfficialUtilityBar";
import { Navbar } from "@/components/common/Navbar";
import { OfficialFooter } from "@/components/common/OfficialFooter";
import { BottomNav } from "@/components/common/BottomNav";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "PlateVision — National Vehicle Vision & Traffic ANPR Portal",
  description:
    "Official-grade Automated Number Plate Recognition (ANPR) and Traffic Surveillance System for Indian and international vehicle registration plates.",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "PlateVision",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: "#0A192F",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} flex min-h-screen flex-col bg-gov-navy text-slate-100`}>
        {/* Top Official Utility & Accessibility Bar */}
        <OfficialUtilityBar />

        {/* Official Header & Navigation */}
        <Navbar />

        {/* Main Application Content Container */}
        <main className="flex-1 container mx-auto max-w-7xl px-4 sm:px-6 pt-6 pb-12">
          {children}
        </main>

        {/* Official Indian Portal Footer */}
        <OfficialFooter />

        {/* Mobile Navigation */}
        <BottomNav />
      </body>
    </html>
  );
}
