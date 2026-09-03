import { HistoryView } from "@/components/history/HistoryView";

export const metadata = {
  title: "Local Scan History - PlateVision",
  description: "View and manage locally stored license plate recognition records",
};

export default function HistoryPage() {
  return (
    <div className="space-y-4">
      <div className="text-center space-y-1">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Scan History
        </h1>
        <p className="text-xs sm:text-sm text-slate-400">
          Saved scans stored strictly in your local browser storage.
        </p>
      </div>

      <HistoryView />
    </div>
  );
}
