import { DetectionResponse, HealthStatus } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "";

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${BASE_URL}/api/health`);
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.statusText}`);
  }
  return res.json();
}

export async function detectImageFile(file: File): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("image", file);

  const res = await fetch(`${BASE_URL}/api/detect/image`, {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) {
    const errorMsg = data?.error?.message || "Failed to process image.";
    throw new Error(errorMsg);
  }

  return data;
}

export async function detectVideoFile(file: File): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("video", file);

  const res = await fetch(`${BASE_URL}/api/detect/video`, {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) {
    const errorMsg = data?.error?.message || "Failed to process video.";
    throw new Error(errorMsg);
  }

  return data;
}

export async function detectFrameBase64(
  frameBase64: string,
  signal?: AbortSignal
): Promise<DetectionResponse> {
  const res = await fetch(`${BASE_URL}/api/detect/frame`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ frame_base64: frameBase64 }),
    signal,
  });

  const data = await res.json();
  if (!res.ok) {
    const errorMsg = data?.error?.message || "Failed to process frame.";
    throw new Error(errorMsg);
  }

  return data;
}

export async function downloadCrop(
  cropUrlOrBase64: string,
  filename: string = "plate_crop.jpg"
) {
  if (cropUrlOrBase64.startsWith("data:")) {
    const a = document.createElement("a");
    a.href = cropUrlOrBase64;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    return;
  }

  const fullUrl = cropUrlOrBase64.startsWith("http")
    ? cropUrlOrBase64
    : `${BASE_URL}${cropUrlOrBase64}`;

  const res = await fetch(fullUrl);
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(objectUrl);
}
