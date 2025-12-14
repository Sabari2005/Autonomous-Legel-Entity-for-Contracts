import path from 'path';

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
// https://vite.dev/config/
export default defineConfig({
  plugins: [react(),    
    tailwindcss(),],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      allowedHosts: ['09a3-2409-4072-2e06-2a14-3144-d1a2-73c8-7a43.ngrok-free.app'],
    },
})
