import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The API base URL is not configured here: the app reads VITE_API_URL from the
// environment or dashboard/.env (see .env.example) when it is built or served.
export default defineConfig({
  plugins: [react()],
})
