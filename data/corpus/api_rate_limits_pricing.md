# Helix Dynamics — API Rate Limits & Pricing Tiers

This page describes the public Helix Relay Cloud API: its pricing plans, rate
limits, and overage policy. All prices are in US dollars.

## Pricing Plans

There are three self-serve pricing plans plus a custom Enterprise plan:

| Plan       | Monthly price | Included messages | Rate limit (req/sec) |
|------------|---------------|-------------------|----------------------|
| Developer  | $0 (free)     | 100,000 / month   | 10 req/sec           |
| Startup    | $249 / month  | 10 million / month| 100 req/sec          |
| Scale      | $1,200 / month| 100 million / month| 1,000 req/sec       |
| Enterprise | custom        | custom            | custom               |

## Overage Pricing

Once a plan's included message quota is exhausted, additional messages are billed
at **$0.40 per 100,000 messages** on the Startup plan and **$0.25 per 100,000
messages** on the Scale plan. The Developer (free) plan does **not** allow
overages; requests are rejected with HTTP 429 once the quota is reached.

## Rate Limiting Behavior

Rate limits are enforced per API key using a token-bucket algorithm with a burst
allowance of **2x** the steady-state limit. When a client exceeds its rate limit,
the API returns **HTTP 429** with a `Retry-After` header indicating the number of
seconds to wait.

## Authentication

API requests authenticate with a bearer token in the `Authorization` header.
API keys are prefixed with `hxk_` and can be rotated from the Cloud console.
Each account may hold up to **25 active API keys**.

## Billing Cycle

Billing runs on the first day of each calendar month. Annual prepayment receives
a **two months free** discount (effectively 10 months' cost for 12 months).
