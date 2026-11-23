import { useEffect, useState, type FormEvent } from 'react'
import { api } from '../api.ts'
import { AREA_LABELS, CHANNEL_LABELS, TIER_LABELS, percent, timeAgo } from '../format.ts'
import type { QueueFilter, QueueResponse, Resolution, ReviewItem } from '../types.ts'
import { CitationList } from './CitationList.tsx'
import { DraftBlock } from './DraftBlock.tsx'

const FILTERS: { value: QueueFilter; label: string }[] = [
  { value: 'pending', label: 'Pending' },
  { value: 'resolved', label: 'Resolved' },
  { value: 'all', label: 'All' },
]

const RESOLUTIONS: { value: Resolution; label: string }[] = [
  { value: 'approved', label: 'Approve draft' },
  { value: 'edited', label: 'Edited before sending' },
  { value: 'escalated', label: 'Escalate further' },
  { value: 'dismissed', label: 'Dismiss' },
]

function ResolveForm({ itemId, onResolved }: { itemId: string; onResolved: () => void }) {
  const [resolution, setResolution] = useState<Resolution>('approved')
  const [reviewer, setReviewer] = useState('')
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await api.resolve(itemId, {
        resolution,
        ...(reviewer.trim() ? { reviewer: reviewer.trim() } : {}),
        ...(note.trim() ? { note: note.trim() } : {}),
      })
      onResolved()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setBusy(false)
    }
  }

  return (
    <form className="resolve-form" onSubmit={handleSubmit}>
      <label className="field">
        <span>Resolution</span>
        <select value={resolution} onChange={(e) => setResolution(e.target.value as Resolution)}>
          {RESOLUTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        <span>Reviewer</span>
        <input
          value={reviewer}
          maxLength={80}
          placeholder="Your name"
          onChange={(e) => setReviewer(e.target.value)}
        />
      </label>
      <label className="field grow">
        <span>Note</span>
        <input
          value={note}
          maxLength={2000}
          placeholder="What you changed or decided"
          onChange={(e) => setNote(e.target.value)}
        />
      </label>
      <button type="submit" className="primary" disabled={busy}>
        {busy ? 'Resolving…' : 'Resolve'}
      </button>
      {error && (
        <p className="alert error" role="alert">
          {error}
        </p>
      )}
    </form>
  )
}

function QueueItem({ item, onResolved }: { item: ReviewItem; onResolved: () => void }) {
  const [open, setOpen] = useState(false)
  return (
    <li className="queue-item">
      <button
        type="button"
        className="queue-summary"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span className={`score-pill band-${item.risk_band}`}>{percent(item.risk_score)}</span>
        <span className="queue-subject">{item.subject}</span>
        <span className="muted small">
          {TIER_LABELS[item.customer_tier]} · {AREA_LABELS[item.product_area]} ·{' '}
          {CHANNEL_LABELS[item.channel]}
        </span>
        <span className="muted small">{timeAgo(item.created_at)}</span>
        {item.status === 'resolved' && <span className="tag">{item.resolution}</span>}
        <span className="chevron" aria-hidden="true">
          {open ? '▾' : '▸'}
        </span>
      </button>

      {open && (
        <div className="queue-details">
          <p className="ticket-body">{item.body}</p>
          {item.signals.length > 0 && (
            <ul className="chips">
              {item.signals.map((signal) => (
                <li key={signal}>{signal}</li>
              ))}
            </ul>
          )}
          <DraftBlock draft={item.draft} guidance={item.guidance} />
          <CitationList citations={item.citations} />
          {item.status === 'pending' ? (
            <ResolveForm itemId={item.id} onResolved={onResolved} />
          ) : (
            <p className="alert info">
              {item.resolution} by {item.reviewer ?? 'a reviewer'}
              {item.resolved_at && ` ${timeAgo(item.resolved_at)}`}
              {item.note && `: ${item.note}`}
            </p>
          )}
        </div>
      )}
    </li>
  )
}

interface ReviewQueueProps {
  disabled: boolean
  /** Changes whenever a new ticket was queued, so the list reloads. */
  refreshKey: number
}

interface QueueRequest {
  filter: QueueFilter
  /** Bumped to refetch the same filter (a new object identity re-runs the effect). */
  attempt: number
}

export function ReviewQueue({ disabled, refreshKey }: ReviewQueueProps) {
  const [request, setRequest] = useState<QueueRequest>({ filter: 'pending', attempt: 0 })
  const [data, setData] = useState<QueueResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  // A new ticket was queued upstream: refetch. Adjusting state while rendering (instead
  // of in an effect) is React's recommended way to respond to a changed prop.
  const [seenRefreshKey, setSeenRefreshKey] = useState(refreshKey)
  if (refreshKey !== seenRefreshKey) {
    setSeenRefreshKey(refreshKey)
    setRequest((current) => ({ ...current, attempt: current.attempt + 1 }))
  }

  useEffect(() => {
    if (disabled) return
    let cancelled = false
    api
      .queue(request.filter)
      .then((response) => {
        if (!cancelled) {
          setData(response)
          setError(null)
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      })
    return () => {
      cancelled = true
    }
  }, [disabled, request])

  const filter = request.filter
  const setFilter = (next: QueueFilter) => setRequest({ filter: next, attempt: 0 })
  const reload = () => setRequest((current) => ({ ...current, attempt: current.attempt + 1 }))

  return (
    <section className="card queue">
      <div className="card-header">
        <h2>
          Human review queue{' '}
          {data && <span className="count">{data.pending_count} pending</span>}
        </h2>
        <div className="queue-controls">
          <div className="segmented" role="group" aria-label="Filter review items">
            {FILTERS.map((option) => (
              <button
                key={option.value}
                type="button"
                className={option.value === filter ? 'active' : ''}
                aria-pressed={option.value === filter}
                onClick={() => setFilter(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
          <button type="button" className="ghost small" onClick={reload} disabled={disabled}>
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <p className="alert error" role="alert">
          {error}
        </p>
      )}
      {data && data.items.length === 0 && (
        <p className="muted">
          {filter === 'pending'
            ? 'Nothing waiting for review. High-risk tickets appear here automatically.'
            : 'No review items yet.'}
        </p>
      )}
      {data && data.items.length > 0 && (
        <ul className="queue-list">
          {data.items.map((item) => (
            <QueueItem key={item.id} item={item} onResolved={reload} />
          ))}
        </ul>
      )}
    </section>
  )
}
