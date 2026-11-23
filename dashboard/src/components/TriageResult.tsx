import { BAND_LABELS, ROUTE_LABELS, percent } from '../format.ts'
import type { RiskBand, TriageResponse } from '../types.ts'
import { CitationList } from './CitationList.tsx'
import { DraftBlock } from './DraftBlock.tsx'

function RiskMeter({ score, threshold, band }: { score: number; threshold: number; band: RiskBand }) {
  return (
    <div className="meter-wrap">
      <div
        className="meter"
        role="meter"
        aria-label="Escalation risk"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={score}
      >
        <div className={`meter-fill band-${band}`} style={{ width: `${score * 100}%` }} />
        <div className="meter-threshold" style={{ left: `${threshold * 100}%` }} />
      </div>
      <div className="meter-legend muted small">
        <span>0%</span>
        <span>review threshold {percent(threshold)}</span>
        <span>100%</span>
      </div>
    </div>
  )
}

export function TriageResult({ result }: { result: TriageResponse | null }) {
  if (result === null) {
    return (
      <section className="card placeholder">
        <h2>Triage result</h2>
        <p className="muted">
          Submit a ticket (or load an example) to see its escalation risk, where it is routed and a
          drafted reply that cites the runbooks it is based on.
        </p>
      </section>
    )
  }

  return (
    <section className="card" aria-live="polite">
      <div className="card-header">
        <h2>Triage result</h2>
        <span className="muted small">{result.ticket_id}</span>
      </div>

      <div className="score-row">
        <div className="score">
          <span className={`score-value band-text-${result.risk_band}`}>
            {percent(result.risk_score)}
          </span>
          <span className="muted small">escalation risk</span>
        </div>
        <div className="badges">
          <span className={`badge band-${result.risk_band}`}>{BAND_LABELS[result.risk_band]}</span>
          <span className={`badge route-${result.route}`}>{ROUTE_LABELS[result.route]}</span>
        </div>
      </div>
      <RiskMeter score={result.risk_score} threshold={result.review_threshold} band={result.risk_band} />

      {result.review_item_id && (
        <p className="alert info">
          Queued for human review as <code>{result.review_item_id}</code>.
        </p>
      )}

      {result.signals.length > 0 && (
        <div className="section">
          <h3>Signals</h3>
          <ul className="chips">
            {result.signals.map((signal) => (
              <li key={signal}>{signal}</li>
            ))}
          </ul>
        </div>
      )}

      <DraftBlock draft={result.draft} guidance={result.guidance} />
      <CitationList citations={result.citations} />

      <p className="muted small meta">
        Model {result.model_version} · composer {result.composer} · inference{' '}
        {result.timings.inference_ms.toFixed(1)} ms · total {result.timings.total_ms.toFixed(1)} ms
      </p>
    </section>
  )
}
