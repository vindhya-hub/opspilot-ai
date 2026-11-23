import type { Channel, CustomerTier, ProductArea, RiskBand, Route } from './types.ts'

export const TIER_LABELS: Record<CustomerTier, string> = {
  free: 'Free',
  pro: 'Pro',
  business: 'Business',
  enterprise: 'Enterprise',
}

export const AREA_LABELS: Record<ProductArea, string> = {
  billing: 'Billing',
  account_access: 'Account access',
  integrations: 'Integrations',
  data_export: 'Data export',
  performance: 'Performance',
  mobile_app: 'Mobile app',
}

export const CHANNEL_LABELS: Record<Channel, string> = {
  email: 'Email',
  chat: 'Chat',
  web_form: 'Web form',
  phone: 'Phone',
  social: 'Social media',
}

export const BAND_LABELS: Record<RiskBand, string> = {
  low: 'Low risk',
  medium: 'Medium risk',
  high: 'High risk',
}

export const ROUTE_LABELS: Record<Route, string> = {
  auto_draft: 'Auto draft',
  human_review: 'Human review',
}

export function percent(probability: number): string {
  return `${(probability * 100).toFixed(probability >= 0.995 || probability < 0.1 ? 1 : 0)}%`
}

const relative = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })

export function timeAgo(iso: string, now: number = Date.now()): string {
  const seconds = Math.round((new Date(iso).getTime() - now) / 1000)
  const steps: [Intl.RelativeTimeFormatUnit, number][] = [
    ['day', 86_400],
    ['hour', 3_600],
    ['minute', 60],
  ]
  for (const [unit, size] of steps) {
    if (Math.abs(seconds) >= size) return relative.format(Math.round(seconds / size), unit)
  }
  return relative.format(seconds, 'second')
}
