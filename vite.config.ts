import { defineConfig } from "vite";

export default defineConfig({
  publicDir: "public",
  assetsInclude: ["**/*.wgsl"],
  build: { outDir: "dist", target: "esnext" },
  server: { port: 8766, host: "127.0.0.1" },
});
