import { ImageUploader } from "@/components/upload/ImageUploader";

export const metadata = {
  title: "Upload Vehicle Photo - PlateVision",
  description: "Upload vehicle images for license plate detection and OCR analysis",
};

export default function UploadPage() {
  return (
    <div className="space-y-4">
      <div className="text-center space-y-1">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Upload Vehicle Photo
        </h1>
        <p className="text-xs sm:text-sm text-slate-400">
          Upload photo from files or choose one of our sample vehicle images.
        </p>
      </div>

      <ImageUploader />
    </div>
  );
}
