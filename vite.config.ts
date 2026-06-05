import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  publicDir: "public",
  assetsInclude: ["**/*.wgsl"],
  build: {
    outDir: "dist",
    target: "esnext",
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        app: resolve(__dirname, "app.html"),
        bench: resolve(__dirname, "bench.html"),
      },
    },
  },
  server: { port: 8766, host: "127.0.0.1" },
});
