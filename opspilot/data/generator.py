"""Deterministic, seeded generator of synthetic support tickets.

Each ticket comes from a small generative story with two *latent* variables that the
model never observes directly:

* ``severity`` (0-1) of the underlying issue drives repeat contacts, ticket age and the
  impact phrases that appear in the body;
* ``frustration`` builds up with severity, repeat contacts and SLA breaches, and leaks
  into the text as frustrated (or calm) phrasing.

The ``escalated`` label is a noisy logistic function of the latent variables, the issue
type and the structured fields (tier, product area, channel, SLA breach), including an
interaction (frustrated or SLA-breached *key accounts* escalate more), followed by
random label flips. Observed features are informative but imperfect proxies of the
label, so the task is learnable without being trivial.

CLI: ``python -m opspilot.data.generator --n 30 --out data/sample_tickets.jsonl``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from opspilot.domain import CHANNELS, PRODUCT_AREAS, SLA_HOURS, TIERS

DEFAULT_SEED = 42

TIER_PROBS = (0.36, 0.30, 0.21, 0.13)
AREA_PROBS = (0.22, 0.17, 0.18, 0.12, 0.17, 0.14)
CHANNEL_PROBS = (0.40, 0.26, 0.17, 0.10, 0.07)
KEY_ACCOUNT_TIERS = ("business", "enterprise")

# Contribution of each structured field to the escalation logit.
TIER_EFFECT = {"free": -0.6, "pro": 0.0, "business": 0.35, "enterprise": 0.8}
AREA_EFFECT = {
    "billing": 0.1,
    "account_access": 0.3,
    "integrations": 0.0,
    "data_export": 0.0,
    "performance": 0.2,
    "mobile_app": -0.3,
}
CHANNEL_EFFECT = {"email": -0.1, "chat": 0.0, "web_form": -0.2, "phone": 0.35, "social": 0.55}

INTERCEPT = -3.4
SEVERITY_WEIGHT = 2.8
FRUSTRATION_WEIGHT = 1.0
KEY_ACCOUNT_BREACH_BONUS = 0.9  # SLA breach on a business/enterprise account
KEY_ACCOUNT_FRUSTRATION_WEIGHT = 0.9  # frustration matters more for key accounts
REPEAT_CONTACT_BONUS = 0.8  # policy: 3+ contacts in 30 days (kb/sla-escalation-policy.md)
LOGIT_NOISE_SD = 0.1
SHARPNESS = 8.0
LABEL_FLIP_RATE = 0.02
FRUSTRATION_LATENT_SD = 0.35  # part of frustration not explained by observable fields

# How faithfully the latent state shows up in the text (higher = noisier read-out).
IMPACT_NOISE_SD = 0.04
FRUSTRATION_NOISE_SD = 0.1

PLANS = ("Starter", "Team", "Business", "Scale")
INTEGRATIONS = ("Slack", "Salesforce", "HubSpot", "Zapier", "GitHub", "Jira")
REPORTS = ("usage report", "audit log", "customer list", "invoice history", "project archive")
DEVICES = ("iPhone 15", "Pixel 8", "Galaxy S23", "iPad Air", "OnePlus 12")
REGIONS = ("Europe", "the US East region", "Australia", "India", "Brazil")

# (subject, issue sentence, issue-type risk) templates per product area.
ISSUES: dict[str, tuple[tuple[str, str, float], ...]] = {
    "billing": (
        (
            "Charged twice this month",
            "I was charged twice for our {plan} subscription this month and invoice "
            "{invoice} shows a duplicate charge of ${amount}.",
            0.4,
        ),
        (
            "Refund for accidental upgrade",
            "We upgraded to the annual {plan} plan by mistake and need a refund of "
            "${amount} for the difference.",
            0.0,
        ),
        (
            "Invoice {invoice} has the wrong VAT",
            "Invoice {invoice} added VAT even though our VAT ID is on file, so the total "
            "is off by ${amount}.",
            -0.5,
        ),
        (
            "Payment failed after card update",
            "I updated the company card but the renewal for our {plan} plan still failed "
            "and the account shows a payment warning.",
            -0.2,
        ),
        (
            "Unexpected price increase",
            "Our {plan} renewal was ${amount} higher than last year and nobody told us "
            "about a price change.",
            0.3,
        ),
    ),
    "account_access": (
        (
            "Locked out after changing phones",
            "I switched to a new phone and lost my authenticator app, so two-factor login "
            "no longer works.",
            0.5,
        ),
        (
            "Password reset email never arrives",
            "I requested a password reset several times but the email never arrives, not "
            "even in spam.",
            -0.2,
        ),
        (
            "Suspicious login alert",
            "We got an alert about a login from a country where we have no staff and we "
            "are worried the account is compromised.",
            1.2,
        ),
        (
            "SSO login loop for our team",
            "Our team gets stuck in a redirect loop when signing in with SSO through our "
            "identity provider.",
            0.6,
        ),
        (
            "Need to remove a former admin",
            "A former employee is still listed as account owner and we cannot remove "
            "their admin access.",
            0.1,
        ),
    ),
    "integrations": (
        (
            "{integration} sync stopped",
            "The {integration} integration stopped syncing yesterday and no new records "
            "have come through since.",
            0.3,
        ),
        (
            "Webhooks failing with 401",
            "Our webhook endpoint started receiving 401 errors and deliveries to our "
            "system are failing.",
            0.4,
        ),
        (
            "API rate limit errors",
            "We keep hitting 429 rate limit errors from the API even though our request "
            "volume has not changed.",
            -0.1,
        ),
        (
            "OAuth token keeps expiring",
            "The {integration} connection asks us to re-authorize every few hours because "
            "the OAuth token expires.",
            -0.3,
        ),
        (
            "Duplicate records from {integration}",
            "Every sync with {integration} creates duplicate records in our workspace.",
            0.0,
        ),
    ),
    "data_export": (
        (
            "Export job stuck at 0%",
            "The export of our {report} has been stuck at 0% for hours.",
            0.0,
        ),
        (
            "CSV export missing columns",
            "The CSV export of our {report} is missing all of our custom field columns.",
            -0.5,
        ),
        (
            "GDPR deletion request",
            "We need all personal data for one of our former users deleted under GDPR and "
            "want written confirmation.",
            0.8,
        ),
        (
            "Need a full account export",
            "We need a complete export of our {report} and attachments before our contract review.",
            0.3,
        ),
        (
            "Export file will not open",
            "The export file we downloaded for the {report} is corrupted and will not open.",
            -0.4,
        ),
    ),
    "performance": (
        (
            "Dashboards extremely slow",
            "Dashboards take more than a minute to load for everyone on our team.",
            0.2,
        ),
        (
            "Timeouts when saving",
            "We get timeout errors whenever we try to save changes to larger projects.",
            0.3,
        ),
        (
            "Search returns errors",
            "Search returns server errors or no results for terms that definitely exist.",
            0.0,
        ),
        (
            "App not loading in {region}",
            "The web app is not loading at all for our users in {region}.",
            1.0,
        ),
        (
            "Reports page keeps crashing",
            "The reports page crashes the browser tab whenever we filter by date.",
            -0.2,
        ),
    ),
    "mobile_app": (
        (
            "App crashes on launch",
            "The mobile app crashes immediately on launch on my {device} after the latest update.",
            0.2,
        ),
        (
            "Mobile changes not syncing",
            "Changes I make on my {device} do not sync to the web app.",
            0.0,
        ),
        (
            "No push notifications",
            "I stopped getting push notifications on my {device} even though they are enabled.",
            -0.6,
        ),
        (
            "Cannot upload photos from phone",
            "Uploading photos from my {device} fails with a generic error.",
            -0.3,
        ),
        (
            "Dark mode text unreadable",
            "In dark mode some text in the mobile app is unreadable on my {device}.",
            -1.0,
        ),
    ),
}

# Impact phrases, graded by the (noisily observed) latent severity.
HIGH_IMPACT_PHRASES = (
    "Our whole team is blocked right now.",
    "This is affecting all of our users.",
    "Production is effectively down for us.",
    "We are losing data because of this.",
    "Our payroll run is blocked until this is fixed.",
    "Our own customers cannot complete orders.",
    "We have a board meeting tomorrow that depends on this.",
)
MODERATE_IMPACT_PHRASES = (
    "A few people on the team are affected.",
    "It is slowing our work down.",
    "We can work around it for a little while.",
    "It is starting to cause problems for us.",
)
LOW_IMPACT_PHRASES = (
    "It is a minor issue and we have a workaround.",
    "Not urgent, just wanted to flag it.",
    "Only one user is affected.",
    "It is not blocking anything for now.",
    "Whenever you get a chance is fine.",
)
# Tone phrases, graded by the (noisily observed) latent frustration.
CALM_PHRASES = (
    "Thanks in advance for your help!",
    "Appreciate any pointers.",
    "Hope you are having a good week.",
    "Thanks for the great product.",
    "Let me know if you need more details.",
)
MILD_PHRASES = (
    "This is a bit frustrating.",
    "Hoping to get this sorted soon.",
    "We would appreciate a quick fix.",
    "A little disappointed that this keeps happening.",
)
STRONG_PHRASES = (
    "This is the third time I am contacting you about this.",
    "This is completely unacceptable.",
    "I am extremely frustrated with the lack of response.",
    "Nobody has replied to my previous emails.",
    "Please escalate this to a manager.",
    "Your previous answer did not solve anything.",
)
THREAT_PHRASES = (
    "We are seriously considering cancelling our contract.",
    "If this is not fixed I will dispute the charge with my bank.",
    "Our legal team is asking about this.",
    "I will be sharing this experience publicly.",
)
FOLLOW_UP_PHRASES = (
    "I already contacted support about this {n} time(s) this month.",
    "Following up on my earlier ticket about this.",
    "This is a follow-up to my previous conversation with your team.",
)
CHANNEL_OPENERS = {
    "phone": "Call summary:",
    "chat": "Chat transcript summary:",
    "social": "Customer posted about this publicly on social media.",
}
SIGN_OFFS = ("Regards,", "Thanks,", "Best,", "Cheers,", "Sent from my phone")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _pick(rng: np.random.Generator, options: tuple[str, ...]) -> str:
    return options[int(rng.integers(len(options)))]


def _pick_weighted(rng: np.random.Generator, options: tuple[str, ...], probs: tuple) -> str:
    return options[int(rng.choice(len(options), p=probs))]


def _sample(rng: np.random.Generator, options: tuple[str, ...], count: int) -> list[str]:
    chosen = rng.choice(len(options), size=count, replace=False)
    return [options[int(i)] for i in chosen]


def _impact_sentences(rng: np.random.Generator, severity: float) -> list[str]:
    """Impact statements: none/low for minor issues up to two blocking-impact statements."""
    observed = severity + float(rng.normal(0.0, IMPACT_NOISE_SD))
    if observed >= 0.63:
        return _sample(rng, HIGH_IMPACT_PHRASES, 2)
    if observed >= 0.51:
        return _sample(rng, HIGH_IMPACT_PHRASES, 1)
    if observed >= 0.39:
        return _sample(rng, MODERATE_IMPACT_PHRASES, 2)
    if observed >= 0.27:
        return _sample(rng, MODERATE_IMPACT_PHRASES, 1)
    if observed >= 0.15:
        return []
    return _sample(rng, LOW_IMPACT_PHRASES, 1) if rng.random() < 0.85 else []


def _tone_sentences(rng: np.random.Generator, frustration: float) -> list[str]:
    """Tone: calm or neutral, then mild, then strong complaints, then threats."""
    observed = frustration + float(rng.normal(0.0, FRUSTRATION_NOISE_SD))
    if observed < 0.35:
        return _sample(rng, CALM_PHRASES, 1) if rng.random() < 0.85 else []
    if observed < 0.7:
        return []
    if observed < 1.0:
        return _sample(rng, MILD_PHRASES, 1)
    if observed < 1.3:
        return _sample(rng, MILD_PHRASES, 2)
    if observed < 1.6:
        return _sample(rng, STRONG_PHRASES, 1)
    if observed < 1.9:
        return _sample(rng, STRONG_PHRASES, 1) + _sample(rng, THREAT_PHRASES, 1)
    return _sample(rng, STRONG_PHRASES, 2) + _sample(rng, THREAT_PHRASES, 1)


def _make_ticket(index: int, rng: np.random.Generator, include_latent: bool) -> dict[str, Any]:
    tier = _pick_weighted(rng, TIERS, TIER_PROBS)
    area = _pick_weighted(rng, PRODUCT_AREAS, AREA_PROBS)
    channel = _pick_weighted(rng, CHANNELS, CHANNEL_PROBS)
    key_account = tier in KEY_ACCOUNT_TIERS

    severity = float(rng.beta(1.6, 3.6))
    prior_contacts = int(min(rng.poisson(0.25 + 2.6 * severity), 12))
    sla_hours = SLA_HOURS[tier]
    hours_open = max(0.1, round(sla_hours * math.exp(rng.normal(-0.55 + 0.9 * severity, 0.8)), 1))
    breached = hours_open > sla_hours
    frustration = (
        1.1 * severity
        + 0.3 * min(prior_contacts, 6)
        + 0.65 * breached
        + (0.3 if channel == "social" else 0.0)
        + float(rng.normal(0.0, FRUSTRATION_LATENT_SD))
    )

    slots = {
        "plan": _pick(rng, PLANS),
        "integration": _pick(rng, INTEGRATIONS),
        "report": _pick(rng, REPORTS),
        "device": _pick(rng, DEVICES),
        "region": _pick(rng, REGIONS),
        "invoice": f"INV-{int(rng.integers(10_000, 99_999))}",
        "amount": f"{int(rng.integers(12, 2_400))}",
    }
    subject_template, issue_template, issue_risk = ISSUES[area][int(rng.integers(5))]
    subject = subject_template.format(**slots)
    if prior_contacts > 0 and rng.random() < 0.5:
        subject = f"Re: {subject}"
    if rng.random() < 0.06 + 0.12 * (severity > 0.5):
        subject = f"URGENT: {subject}"

    sentences: list[str] = []
    if channel in CHANNEL_OPENERS:
        sentences.append(CHANNEL_OPENERS[channel])
    sentences.append(issue_template.format(**slots))
    sentences.extend(_impact_sentences(rng, severity))
    if prior_contacts > 0 and rng.random() < 0.6:
        sentences.append(_pick(rng, FOLLOW_UP_PHRASES).format(n=prior_contacts))
    sentences.extend(_tone_sentences(rng, frustration))
    sentences.append(_pick(rng, SIGN_OFFS))

    logit = (
        INTERCEPT
        + SEVERITY_WEIGHT * severity
        + FRUSTRATION_WEIGHT * frustration
        + issue_risk
        + TIER_EFFECT[tier]
        + AREA_EFFECT[area]
        + CHANNEL_EFFECT[channel]
        + (KEY_ACCOUNT_BREACH_BONUS if breached and key_account else 0.0)
        + (KEY_ACCOUNT_FRUSTRATION_WEIGHT * frustration if key_account else 0.0)
        + (REPEAT_CONTACT_BONUS if prior_contacts >= 3 else 0.0)
    )
    noisy_logit = logit + float(rng.normal(0.0, LOGIT_NOISE_SD))
    escalated = rng.random() < _sigmoid(SHARPNESS * noisy_logit)
    if rng.random() < LABEL_FLIP_RATE:
        escalated = not escalated

    ticket: dict[str, Any] = {
        "ticket_id": f"T{index:05d}",
        "subject": subject,
        "body": " ".join(sentences),
        "customer_tier": tier,
        "product_area": area,
        "channel": channel,
        "prior_contacts_30d": prior_contacts,
        "hours_open": hours_open,
        "sla_hours": sla_hours,
        "sla_breached": breached,
        "escalated": int(escalated),
    }
    if include_latent:
        ticket["latent"] = {
            "severity": round(severity, 4),
            "frustration": round(frustration, 4),
            "p_escalation": round(_sigmoid(SHARPNESS * logit), 4),
        }
    return ticket


def generate_tickets(
    n: int, seed: int = DEFAULT_SEED, *, include_latent: bool = False
) -> list[dict[str, Any]]:
    """Generate ``n`` synthetic tickets; the same ``(n, seed)`` always yields the same data.

    ``include_latent`` adds the hidden severity / frustration / noise-free escalation
    probability under ``"latent"`` (for analysis only; never use it as a model input).
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    return [_make_ticket(index, rng, include_latent) for index in range(1, n + 1)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic support tickets (JSONL).")
    parser.add_argument("--n", type=int, default=1000, help="number of tickets")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument("--out", type=Path, default=None, help="output file (default: stdout)")
    args = parser.parse_args(argv)

    lines = "\n".join(json.dumps(t) for t in generate_tickets(args.n, args.seed)) + "\n"
    if args.out is None:
        sys.stdout.write(lines)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(lines, encoding="utf-8")
        print(f"wrote {args.n} tickets to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
