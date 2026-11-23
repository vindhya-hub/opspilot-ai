// Mirrors the API schemas in opspilot/api/schemas.py.

export type CustomerTier = 'free' | 'pro' | 'business' | 'enterprise'
export type ProductArea =
  | 'billing'
  | 'account_access'
  | 'integrations'
  | 'data_export'
  | 'performance'
  | 'mobile_app'
export type Channel = 'email' | 'chat' | 'web_form' | 'phone' | 'social'
export type RiskBand = 'low' | 'medium' | 'high'
export type Route = 'auto_draft' | 'human_review'
export type Resolution = 'approved' | 'edited' | 'escalated' | 'dismissed'
export type QueueFilter = 'pending' | 'resolved' | 'all'

export interface TicketInput {
  ticket_id?: string
  subject: string
  body: string
  customer_tier: CustomerTier
  product_area: ProductArea
  channel: Channel
  prior_contacts_30d: number
  hours_open: number
}

export interface Citation {
  ref: number
  doc_id: string
  title: string
  section: string
  audience: 'customer' | 'internal'
  score: number
  excerpt: string
}

export interface TriageResponse {
  ticket_id: string
  risk_score: number
  risk_band: RiskBand
  route: Route
  review_threshold: number
  model_version: string
  signals: string[]
  draft: string
  guidance: string[]
  citations: Citation[]
  composer: string
  review_item_id: string | null
  timings: { inference_ms: number; drafting_ms: number; total_ms: number }
}

export interface ReviewItem {
  id: string
  ticket_id: string
  status: 'pending' | 'resolved'
  subject: string
  body: string
  customer_tier: CustomerTier
  product_area: ProductArea
  channel: Channel
  risk_score: number
  risk_band: RiskBand
  signals: string[]
  draft: string
  guidance: string[]
  citations: Citation[]
  created_at: string
  resolved_at: string | null
  resolution: Resolution | null
  reviewer: string | null
  note: string | null
}

export interface QueueResponse {
  items: ReviewItem[]
  pending_count: number
}

export interface ResolveRequest {
  resolution: Resolution
  reviewer?: string
  note?: string
}

export interface Readiness {
  status: string
  model_version: string | null
  kb_documents: number | null
  reason: string | null
}

export interface ModelInfo {
  model_version: string
  trained_at: string
  input_dim: number
  tuned_threshold: number
  review_threshold: number
  medium_threshold: number
  synthetic_data: boolean
  test_metrics: { f1: number; precision: number; recall: number; roc_auc?: number } | null
}
