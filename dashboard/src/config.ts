/**
 * Runtime configuration, read once from Vite's build-time environment.
 *
 * VITE_API_URL has no fallback on purpose: when it is missing the dashboard shows a
 * configuration error instead of silently calling a guessed address. VITE_* values are
 * embedded in the bundle, so they must never contain secrets.
 */
function readApiUrl(): string | null {
  const raw = import.meta.env.VITE_API_URL
  if (typeof raw !== 'string' || raw.trim() === '') return null
  return raw.trim().replace(/\/+$/, '')
}

export const API_URL: string | null = readApiUrl()
