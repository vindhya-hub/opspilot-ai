import type { TicketInput } from './types.ts'

export const EXAMPLES: { label: string; ticket: TicketInput }[] = [
  {
    label: 'High-risk example',
    ticket: {
      subject: 'Re: Charged twice this month',
      body:
        'I was charged twice for our Scale subscription this month and invoice INV-55120 shows ' +
        'a duplicate charge. Our payroll run is blocked until this is fixed. This is the third ' +
        'time I am contacting you about this. If this is not fixed I will dispute the charge ' +
        'with my bank.',
      customer_tier: 'enterprise',
      product_area: 'billing',
      channel: 'email',
      prior_contacts_30d: 3,
      hours_open: 9.5,
    },
  },
  {
    label: 'Low-risk example',
    ticket: {
      subject: 'CSV export missing columns',
      body:
        'The CSV export of our customer list is missing all of our custom field columns. It is ' +
        'a minor issue and we have a workaround. Thanks in advance for your help!',
      customer_tier: 'pro',
      product_area: 'data_export',
      channel: 'web_form',
      prior_contacts_30d: 0,
      hours_open: 2.5,
    },
  },
]
