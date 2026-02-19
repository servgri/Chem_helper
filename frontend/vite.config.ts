import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Ketcher pulls Node packages (util/assert) that expect `process` in the browser.
export default defineConfig({
  plugins: [react()],
  define: {
    global: "globalThis",
    "process.env": JSON.stringify({
      NODE_ENV: process.env.NODE_ENV || "development",
    }),
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    watch: { usePolling: true },
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8511",
        changeOrigin: true,
      },
      "/admin": {
        target: "http://127.0.0.1:8511",
        changeOrigin: true,
      },
      "/static": {
        target: "http://127.0.0.1:8511",
        changeOrigin: true,
      },
    },
  },
  optimizeDeps: {
    include: ["ketcher-standalone", "ketcher-react", "ketcher-core", "ketcher-react > assert", "util"],
    esbuildOptions: {
      define: {
        global: "globalThis",
      },
    },
  },
});
