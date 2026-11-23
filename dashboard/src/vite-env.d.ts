/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the OpsPilot API, e.g. http://localhost:8000 (public, build-time). */
  readonly VITE_API_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
