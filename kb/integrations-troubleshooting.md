# Integrations and API Troubleshooting

## Sync stopped or failing

An integration usually stops syncing because the connected user lost permissions in the other app or the connection was revoked. Re-authorizing the connection under Settings > Integrations restarts the sync and backfills any records missed in the last seven days.

## Webhook delivery failures

Webhook deliveries that fail with 401 errors usually mean the receiving endpoint is validating an outdated signing secret. Once the receiving side uses the current signing secret, failed deliveries can be replayed from the webhook log for up to 72 hours. Webhooks that fail continuously for three days are paused automatically and must be re-enabled.

## API rate limits

API rate limits depend on the plan, and every response with status 429 includes a Retry-After header that says when to retry. Clients should retry with exponential backoff, and rate limit errors that appear without any change in request volume can be checked on the API usage page.

## OAuth tokens and duplicate records

An OAuth token that expires every few hours usually points to an identity provider session policy that revokes refresh tokens. Duplicate records after a sync are typically caused by two active connections to the same workspace or by a field mapping without a unique key, and removing the extra connection stops new duplicates.
