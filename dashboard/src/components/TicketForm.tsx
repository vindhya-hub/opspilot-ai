import { useState, type FormEvent } from 'react'
import { api } from '../api.ts'
import { EXAMPLES } from '../examples.ts'
import { AREA_LABELS, CHANNEL_LABELS, TIER_LABELS } from '../format.ts'
import type { Channel, CustomerTier, ProductArea, TicketInput, TriageResponse } from '../types.ts'

const MAX_SUBJECT = 200
const MAX_BODY = 5000

interface FormState {
  ticket_id: string
  subject: string
  body: string
  customer_tier: CustomerTier
  product_area: ProductArea
  channel: Channel
  prior_contacts_30d: string
  hours_open: string
}

const EMPTY_FORM: FormState = {
  ticket_id: '',
  subject: '',
  body: '',
  customer_tier: 'pro',
  product_area: 'billing',
  channel: 'email',
  prior_contacts_30d: '0',
  hours_open: '1',
}

function toForm(ticket: TicketInput): FormState {
  return {
    ...ticket,
    ticket_id: ticket.ticket_id ?? '',
    prior_contacts_30d: String(ticket.prior_contacts_30d),
    hours_open: String(ticket.hours_open),
  }
}

function toTicket(form: FormState): TicketInput {
  const ticketId = form.ticket_id.trim()
  return {
    ...(ticketId ? { ticket_id: ticketId } : {}),
    subject: form.subject,
    body: form.body,
    customer_tier: form.customer_tier,
    product_area: form.product_area,
    channel: form.channel,
    prior_contacts_30d: Number(form.prior_contacts_30d),
    hours_open: Number(form.hours_open),
  }
}

function options<T extends string>(labels: Record<T, string>) {
  return (Object.entries(labels) as [T, string][]).map(([value, label]) => (
    <option key={value} value={value}>
      {label}
    </option>
  ))
}

interface TicketFormProps {
  disabled: boolean
  onTriaged: (result: TriageResponse) => void
}

export function TicketForm({ disabled, onTriaged }: TicketFormProps) {
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function update<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((current) => ({ ...current, [key]: value }))
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      onTriaged(await api.triage(toTicket(form)))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="card">
      <div className="card-header">
        <h2>New ticket</h2>
        <div className="example-buttons">
          {EXAMPLES.map((example) => (
            <button
              key={example.label}
              type="button"
              className="ghost small"
              onClick={() => {
                setForm(toForm(example.ticket))
                setError(null)
              }}
            >
              {example.label}
            </button>
          ))}
        </div>
      </div>

      <form className="ticket-form" onSubmit={handleSubmit}>
        <label className="field">
          <span>
            Subject <span className="muted small">{form.subject.length}/{MAX_SUBJECT}</span>
          </span>
          <input
            value={form.subject}
            maxLength={MAX_SUBJECT}
            required
            placeholder="e.g. Charged twice this month"
            onChange={(e) => update('subject', e.target.value)}
          />
        </label>

        <label className="field">
          <span>
            Message <span className="muted small">{form.body.length}/{MAX_BODY}</span>
          </span>
          <textarea
            value={form.body}
            maxLength={MAX_BODY}
            rows={6}
            required
            placeholder="What the customer wrote"
            onChange={(e) => update('body', e.target.value)}
          />
        </label>

        <div className="field-row">
          <label className="field">
            <span>Customer tier</span>
            <select
              value={form.customer_tier}
              onChange={(e) => update('customer_tier', e.target.value as CustomerTier)}
            >
              {options(TIER_LABELS)}
            </select>
          </label>
          <label className="field">
            <span>Product area</span>
            <select
              value={form.product_area}
              onChange={(e) => update('product_area', e.target.value as ProductArea)}
            >
              {options(AREA_LABELS)}
            </select>
          </label>
          <label className="field">
            <span>Channel</span>
            <select
              value={form.channel}
              onChange={(e) => update('channel', e.target.value as Channel)}
            >
              {options(CHANNEL_LABELS)}
            </select>
          </label>
        </div>

        <div className="field-row">
          <label className="field">
            <span>Prior contacts (30 days)</span>
            <input
              type="number"
              min={0}
              max={1000}
              step={1}
              required
              value={form.prior_contacts_30d}
              onChange={(e) => update('prior_contacts_30d', e.target.value)}
            />
          </label>
          <label className="field">
            <span>Hours open</span>
            <input
              type="number"
              min={0}
              max={8760}
              step={0.5}
              required
              value={form.hours_open}
              onChange={(e) => update('hours_open', e.target.value)}
            />
          </label>
          <label className="field">
            <span>
              Ticket ID <span className="muted small">optional</span>
            </span>
            <input
              value={form.ticket_id}
              maxLength={64}
              pattern="[A-Za-z0-9._:\-]+"
              placeholder="auto"
              onChange={(e) => update('ticket_id', e.target.value)}
            />
          </label>
        </div>

        {error && (
          <p className="alert error" role="alert">
            {error}
          </p>
        )}

        <div className="form-actions">
          <button type="submit" className="primary" disabled={disabled || submitting}>
            {submitting ? 'Triaging…' : 'Triage ticket'}
          </button>
          <button
            type="button"
            className="ghost"
            onClick={() => {
              setForm(EMPTY_FORM)
              setError(null)
            }}
          >
            Clear
          </button>
        </div>
      </form>
    </section>
  )
}
