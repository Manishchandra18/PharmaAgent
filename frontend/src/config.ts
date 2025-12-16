// frontend/src/config.ts
// Centralized configuration for API endpoints

// Determine a sensible default base URL when one isn't provided via env vars.
// - In local development, default to the FastAPI server on port 8000
// - In deployed environments, default to the current origin
const getDefaultApiBaseUrl = () => {
  if (typeof window !== "undefined") {
    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
      return "http://localhost:8000";
    }
    return window.location.origin;
  }
  // Fallback for non-browser environments
  return "http://localhost:8000";
};

// Vite exposes env vars prefixed with VITE_ via import.meta.env
export const API_BASE_URL: string =
  (import.meta as any).env?.VITE_API_BASE_URL || getDefaultApiBaseUrl();

// Helper to build versioned API URLs
export const apiUrl = (path: string): string => {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
};


