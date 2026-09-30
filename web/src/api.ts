// Client for the diagnosis API (same origin as the web app).

export interface Task {
  id: string;
  name: string;
  modality: string;
  input_hint: string;
}

export interface Diagnosis {
  level: "high" | "low" | "uncertain";
  verdict: string;
  advice: string;
}

export class AuthError extends Error {}

const TOKEN_KEY = "diagnosis-access-token";

export function getToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setToken(token: string) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* private browsing: the token is kept only for this page view */
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(path, { ...init, headers });
  if (res.status === 401) throw new AuthError("An access code is required");
  if (!res.ok) {
    let message = "The service is temporarily unavailable. Please try again later.";
    try {
      message = (await res.json()).detail ?? message;
    } catch {
      /* non-JSON error page */
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export const fetchTasks = () => request<Task[]>("/api/tasks");

export function diagnose(taskId: string, photo: Blob): Promise<Diagnosis> {
  const form = new FormData();
  form.append("task_id", taskId);
  form.append("image", photo, "photo.jpg");
  return request<Diagnosis>("/api/diagnose", { method: "POST", body: form });
}

// Downscale a camera photo before upload: a 12-megapixel iPhone photo (several MB) becomes
// a JPEG of roughly 100-200 KB, which uploads in well under a second on a mobile network.
// The models look at 224-240 px, so 1024 px loses nothing. The canvas re-encode also turns
// HEIC into JPEG and applies the photo's orientation (browsers draw images upright).
export async function preparePhoto(file: File, maxSide = 1024): Promise<Blob> {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    canvas.getContext("2d")!.drawImage(img, 0, 0, canvas.width, canvas.height);
    return await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Could not process the image"))), "image/jpeg", 0.85),
    );
  } finally {
    URL.revokeObjectURL(url);
  }
}
