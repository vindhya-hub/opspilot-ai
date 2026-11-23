import { API_URL } from './config.ts'
import type {
  ModelInfo,
  QueueFilter,
  QueueResponse,
  Readiness,
  ResolveRequest,
  ReviewItem,
  TicketInput,
  TriageResponse,
} from './types.ts'

export class ApiError extends Error {
  readonly status: number
  readonly body: unknown

  constructor(status: number, message: string, body: unknown = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

interface ValidationIssue {
  loc?: (string | number)[]
  msg?: string
}

/** Human-readable message from a FastAPI error body (string detail or 422 issue list). */
function describeError(status: number, body: unknown): string {
  if (body !== null && typeof body === 'object' && 'detail' in body) {
    const { detail } = body as { detail: unknown }
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return (detail as ValidationIssue[])
        .map((issue) => {
          const field = (issue.loc ?? []).filter((part) => part !== 'body').join('.')
          return field ? `${field}: ${issue.msg ?? 'invalid'}` : (issue.msg ?? 'invalid')
        })
        .join('; ')
    }
  }
  return `Request failed with status ${status}`
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  if (API_URL === null) {
    throw new ApiError(0, 'VITE_API_URL is not configured')
  }
  // Only send a JSON content type with a body, so simple GETs need no CORS preflight.
  const headers = init.body === undefined ? undefined : { 'Content-Type': 'application/json' }
  let response: Response
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, headers })
  } catch {
    throw new ApiError(0, `Cannot reach the OpsPilot API at ${API_URL}`)
  }
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new ApiError(response.status, describeError(response.status, body), body)
  }
  return body as T
}

export const api = {
  /** Readiness never throws for "not ready" (503); only for unreachable/other errors. */
  async readiness(): Promise<Readiness> {
    try {
      return await request<Readiness>('/readyz')
    } catch (error) {
      if (error instanceof ApiError && error.status === 503 && error.body) {
        return error.body as Readiness
      }
      throw error
    }
  },
  modelInfo: () => request<ModelInfo>('/v1/model'),
  triage: (ticket: TicketInput) =>
    request<TriageResponse>('/v1/triage', { method: 'POST', body: JSON.stringify(ticket) }),
  queue: (status: QueueFilter) => request<QueueResponse>(`/v1/queue?status=${status}&limit=50`),
  resolve: (id: string, body: ResolveRequest) =>
    request<ReviewItem>(`/v1/queue/${encodeURIComponent(id)}/resolve`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
}
