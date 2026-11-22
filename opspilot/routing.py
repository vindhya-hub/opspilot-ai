"""Risk bands and routing: scores at or above the review threshold go to a human."""

from __future__ import annotations

import math
from dataclasses import dataclass

from opspilot.domain import AUTO_DRAFT, HUMAN_REVIEW, RiskBand, Route

#: Without an explicit medium threshold, "medium" starts at this fraction of the review one.
DEFAULT_MEDIUM_FRACTION = 0.6


@dataclass(frozen=True)
class RoutingDecision:
    risk_score: float
    risk_band: RiskBand
    route: Route


@dataclass(frozen=True)
class RiskRouter:
    """``score >= review_threshold`` -> ``human_review`` (band "high"), else ``auto_draft``.

    The band below that is "medium" from ``medium_threshold`` and "low" underneath.
    """

    review_threshold: float
    medium_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 < self.review_threshold <= 1.0:
            raise ValueError("review_threshold must be in (0, 1]")
        if not 0.0 <= self.medium_threshold <= self.review_threshold:
            raise ValueError("medium_threshold must be in [0, review_threshold]")

    @classmethod
    def from_thresholds(
        cls, review_threshold: float, medium_threshold: float | None = None
    ) -> RiskRouter:
        if medium_threshold is None:
            medium_threshold = round(DEFAULT_MEDIUM_FRACTION * review_threshold, 4)
        return cls(review_threshold, medium_threshold)

    def decide(self, risk_score: float) -> RoutingDecision:
        if math.isnan(risk_score) or not 0.0 <= risk_score <= 1.0:
            raise ValueError(f"risk_score must be a probability, got {risk_score!r}")
        if risk_score >= self.review_threshold:
            return RoutingDecision(risk_score, "high", HUMAN_REVIEW)
        band: RiskBand = "medium" if risk_score >= self.medium_threshold else "low"
        return RoutingDecision(risk_score, band, AUTO_DRAFT)
