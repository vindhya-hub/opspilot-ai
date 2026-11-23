import type { ModelInfo, Readiness } from '../types.ts'

interface StatusBarProps {
  readiness: Readiness | null
  modelInfo: ModelInfo | null
  error: string | null
  onRetry: () => void
}

function StatusPill({ readiness, error }: Pick<StatusBarProps, 'readiness' | 'error'>) {
  if (error) return <span className="status-pill error">API unreachable</span>
  if (readiness === null) return <span className="status-pill">Connecting…</span>
  if (readiness.status === 'ready') return <span className="status-pill ok">API ready</span>
  return <span className="status-pill warn">Model not loaded</span>
}

export function StatusBar({ readiness, modelInfo, error, onRetry }: StatusBarProps) {
  const metrics = modelInfo?.test_metrics
  return (
    <header className="topbar">
      <div className="brand">
        <img src="/favicon.svg" alt="" width={36} height={36} />
        <div>
          <h1>OpsPilot AI</h1>
          <p className="muted">Escalation-risk triage with cited guidance</p>
        </div>
      </div>
      <div className="status">
        <StatusPill readiness={readiness} error={error} />
        {(error || readiness?.status === 'not_ready') && (
          <button type="button" className="ghost small" onClick={onRetry}>
            Retry
          </button>
        )}
        {modelInfo && (
          <span className="muted small">
            Model {modelInfo.model_version} · review threshold{' '}
            {modelInfo.review_threshold.toFixed(2)}
            {metrics && ` · test F1 ${metrics.f1.toFixed(3)} on synthetic data`}
          </span>
        )}
      </div>
    </header>
  )
}
