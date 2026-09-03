import { CCTVStreamViewer } from "@/components/cctv/CCTVStreamViewer";

export const metadata = {
  title: "Live CCTV Surveillance Feed - PlateVision",
  description: "Continuous real-time multi-plate detection from CCTV and RTSP IP camera streams",
};

export default function CCTVPage() {
  return (
    <div className="space-y-4">
      <div className="text-center space-y-1">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          CCTV & RTSP Live Surveillance Feed
        </h1>
        <p className="text-xs sm:text-sm text-slate-400">
          Connect your IP camera or test with real-time city traffic feeds.
        </p>
      </div>

      <CCTVStreamViewer />
    </div>
  );
}
