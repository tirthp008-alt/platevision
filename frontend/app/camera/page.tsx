import { CameraScanner } from "@/components/camera/CameraScanner";

export const metadata = {
  title: "Live Camera Scanner - PlateVision",
  description: "Real-time vehicle number plate recognition via phone camera",
};

export default function CameraPage() {
  return (
    <div className="space-y-4">
      <div className="text-center space-y-1">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Live Camera Scanner
        </h1>
        <p className="text-xs sm:text-sm text-slate-400">
          Point your rear camera at any vehicle license plate for real-time recognition.
        </p>
      </div>

      <CameraScanner />
    </div>
  );
}
