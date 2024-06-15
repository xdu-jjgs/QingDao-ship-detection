import { defineConfig } from "vite";
import path from "node:path";
import vue from "@vitejs/plugin-vue";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5174,
    headers: {
      "Permissions-Policy": "autoplay=self",
    },
  },
});
