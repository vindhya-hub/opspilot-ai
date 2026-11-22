"""Rule-based risk signals: short, human-readable reasons a ticket may be sensitive.

These are *not* model explanations (the risk score comes from the model). They surface the
escalation triggers from ``kb/sla-escalation-policy.md`` and obvious cues in the text so a
reviewer can see at a glance what to look at, and they steer the tone of the draft reply.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from opspilot.domain import KEY_ACCOUNT_TIERS, REPEAT_CONTACT_TRIGGER, sla_hours_for

#: Fraction of the SLA window after which a ticket counts as "at risk".
SLA_AT_RISK_FRACTION = 0.75


@dataclass(frozen=True)
class Signal:
    code: str
    label: str


_TEXT_CUES: tuple[tuple[str, str, re.Pattern[str]], ...] = tuple(
    (code, label, re.compile(pattern, re.IGNORECASE))
    for code, label, pattern in (
        (
            "cancellation_or_legal",
            "Mentions cancelling, legal action or a payment dispute",
            r"\bcancel\w*|\blegal\b|\blawyer|\bdispute\b|\bchargeback",
        ),
        (
            "public_complaint",
            "Mentions a public complaint",
            r"\bpublicly\b|\bsocial media\b|\bposted about this\b",
        ),
        (
            "manager_request",
            "Asks for a manager or an escalation",
            r"\bmanager\b|\bescalate\b",
        ),
        (
            "strong_frustration",
            "Strong frustration or repeated failed contact",
            r"unacceptable|extremely frustrated|lack of response|nobody has replied"
            r"|did not solve|third time",
        ),
        (
            "business_impact",
            "Describes business-critical impact",
            r"whole team is blocked|all of our users|production is|losing data|payroll"
            r"|cannot complete orders|board meeting",
        ),
        (
            "security",
            "Possible account compromise",
            r"compromised|suspicious login|unauthori[sz]ed",
        ),
    )
)


def detect_signals(ticket: Mapping[str, Any]) -> list[Signal]:
    """Signals for one ticket (same field names as the API / generator)."""
    tier = str(ticket["customer_tier"])
    hours_open = float(ticket.get("hours_open", 0.0))
    prior_contacts = int(ticket.get("prior_contacts_30d", 0))
    sla_hours = sla_hours_for(tier)
    signals: list[Signal] = []

    if hours_open > sla_hours:
        signals.append(
            Signal(
                "sla_breached",
                f"SLA breached: open {hours_open:.1f}h against the {sla_hours}h "
                f"{tier.title()} window",
            )
        )
    elif hours_open >= SLA_AT_RISK_FRACTION * sla_hours:
        used = round(100 * hours_open / sla_hours)
        signals.append(
            Signal("sla_at_risk", f"SLA at risk: {used}% of the {sla_hours}h window used")
        )
    if prior_contacts >= REPEAT_CONTACT_TRIGGER:
        signals.append(
            Signal("repeat_contact", f"{prior_contacts} prior contacts in the last 30 days")
        )
    if tier in KEY_ACCOUNT_TIERS:
        signals.append(Signal("key_account", f"Key account ({tier.title()} tier)"))
    if ticket.get("channel") == "social":
        signals.append(Signal("public_channel", "Raised on a public channel (social media)"))

    text = f"{ticket.get('subject', '')} {ticket.get('body', '')}"
    signals.extend(Signal(code, label) for code, label, cue in _TEXT_CUES if cue.search(text))
    return signals
