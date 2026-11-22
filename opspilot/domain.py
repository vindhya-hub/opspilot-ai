"""Shared ticket vocabulary used by the data generator, the model and the API."""

from __future__ import annotations

from typing import Literal, get_args

CustomerTier = Literal["free", "pro", "business", "enterprise"]
ProductArea = Literal[
    "billing", "account_access", "integrations", "data_export", "performance", "mobile_app"
]
Channel = Literal["email", "chat", "web_form", "phone", "social"]
Route = Literal["auto_draft", "human_review"]
RiskBand = Literal["low", "medium", "high"]

TIERS: tuple[str, ...] = get_args(CustomerTier)
PRODUCT_AREAS: tuple[str, ...] = get_args(ProductArea)
CHANNELS: tuple[str, ...] = get_args(Channel)

AUTO_DRAFT: Route = "auto_draft"
HUMAN_REVIEW: Route = "human_review"

#: SLA window in hours per tier: the longest a ticket may stay open without a
#: resolution or an agreed action plan. Mirrors ``kb/sla-escalation-policy.md``.
SLA_HOURS: dict[str, int] = {"free": 72, "pro": 24, "business": 8, "enterprise": 4}


def sla_hours_for(tier: str) -> int:
    """SLA window for ``tier`` (unknown tiers fall back to the Pro window)."""
    return SLA_HOURS.get(tier, SLA_HOURS["pro"])
