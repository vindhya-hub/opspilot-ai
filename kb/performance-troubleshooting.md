# Performance Troubleshooting

## Slow dashboards and pages

Slow dashboards are most often caused by very large date ranges, many widgets on a single dashboard or browser extensions that block scripts. Narrowing the date range and testing in a private browser window shows whether the slowness is specific to one account. When a whole team is affected, support collects a browser network log (HAR file) and checks the status page for an ongoing incident.

## Timeouts and errors when saving

Save timeouts on very large projects are a known limit, and splitting the project or archiving old items reduces the load. Server errors or missing results in search usually mean the search index needs rebuilding, which support can trigger and which finishes within a few hours.

## Crashes in reports

A reports page that crashes the browser tab when filtering is usually caused by a filter that returns a very large result set. Applying a shorter date range first avoids the crash while the issue is investigated.

## Region-wide problems

If the app is not loading for users in one region, the problem is handled as a possible incident under the outage communications process instead of as a single-account issue.
