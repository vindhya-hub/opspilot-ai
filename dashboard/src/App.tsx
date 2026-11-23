import { useEffect, useState } from 'react'
import { api } from './api.ts'
import { ReviewQueue } from './components/ReviewQueue.tsx'
import { StatusBar } from './components/StatusBar.tsx'
import { TicketForm } from './components/TicketForm.tsx'
import { TriageResult } from './components/TriageResult.tsx'
import { API_URL } from './config.ts'
import type { ModelInfo, Readiness, TriageResponse } from './types.ts'

interface ServiceStatus {
  readiness: Readiness | null
  modelInfo: ModelInfo | null
  error: string | null
}

const UNKNOWN_STATUS: ServiceStatus = { readiness: null, modelInfo: null, error: null }

async function fetchStatus(): Promise<ServiceStatus> {
  try {
    const readiness = await api.readiness()
    const modelInfo = readiness.status === 'ready' ? await api.modelInfo() : null
    return { readiness, modelInfo, error: null }
  } catch (error) {
    return { ...UNKNOWN_STATUS, error: error instanceof Error ? error.message : String(error) }
  }
}

export default function App() {
  const [status, setStatus] = useState<ServiceStatus>(UNKNOWN_STATUS)
  const [result, setResult] = useState<TriageResponse | null>(null)
  const [queueRefresh, setQueueRefresh] = useState(0)
  const configured = API_URL !== null

  useEffect(() => {
    if (!configured) return
    let cancelled = false
    void fetchStatus().then((next) => {
      if (!cancelled) setStatus(next)
    })
    return () => {
      cancelled = true
    }
  }, [configured])

  function retryStatus() {
    void fetchStatus().then(setStatus)
  }

  function handleTriaged(triage: TriageResponse) {
    setResult(triage)
    if (triage.review_item_id !== null) setQueueRefresh((n) => n + 1)
  }

  const statusError = status.error
  return (
    <div className="app">
      <StatusBar
        readiness={status.readiness}
        modelInfo={status.modelInfo}
        error={statusError}
        onRetry={retryStatus}
      />
      {!configured && (
        <p className="alert error config-error" role="alert">
          VITE_API_URL is not set, so the dashboard does not know where the API is. Copy{' '}
          <code>dashboard/.env.example</code> to <code>dashboard/.env</code> (or pass the build
          argument) and restart.
        </p>
      )}
      {statusError && <p className="alert error config-error">{statusError}</p>}
      <main className="layout">
        <TicketForm disabled={!configured} onTriaged={handleTriaged} />
        <TriageResult result={result} />
        <ReviewQueue disabled={!configured} refreshKey={queueRefresh} />
      </main>
      <footer className="muted small">
        Scores come from a PyTorch model trained on synthetic tickets; drafts cite the runbooks in{' '}
        <code>kb/</code>. Every high-risk ticket is reviewed by a person before anything is sent.
      </footer>
    </div>
  )
}
