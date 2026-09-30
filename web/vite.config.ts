import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.png", "apple-touch-icon.png"],
      manifest: {
        name: "Cancer Image Diagnosis",
        short_name: "Lesion Check",
        description: "Take a photo of a lesion and get AI-assisted diagnostic guidance",
        lang: "en",
        theme_color: "#0f766e",
        background_color: "#f8fafc",
        display: "standalone",
        start_url: "/",
        icons: [
          { src: "pwa-192x192.png", sizes: "192x192", type: "image/png" },
          { src: "pwa-512x512.png", sizes: "512x512", type: "image/png" },
        ],
      },
      workbox: {
        // only the app shell is cached; API calls (and photos) always go to the server
        navigateFallbackDenylist: [/^\/api\//],
      },
    }),
  ],
  // local development: `npm run dev` forwards API calls to a local uvicorn
  server: { proxy: { "/api": "http://localhost:8000" } },
});
