"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from opspilot.domain import (
    Channel,
    CustomerTier,
    ProductArea,
    Resolution,
    ReviewStatus,
    RiskBand,
    Route,
)
from opspilot.review_queue import ReviewItem

MAX_SUBJECT_CHARS = 200
MAX_BODY_CHARS = 5_000
MAX_BATCH_SIZE = 100

TicketId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$"
    ),
]
Subject = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_SUBJECT_CHARS)
]
Body = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_BODY_CHARS)
]

EXAMPLE_TICKET: dict[str, Any] = {
    "ticket_id": "T-1042",
    "subject": "Charged twice this month",
    "body": (
        "I was charged twice for our Team subscription and invoice INV-20431 shows a duplicate "
        "charge. This is the third time I am contacting you about this. If this is not fixed I "
        "will dispute the charge with my bank."
    ),
    "customer_tier": "enterprise",
    "product_area": "billing",
    "channel": "email",
    "prior_contacts_30d": 3,
    "hours_open": 9.5,
}


class TicketIn(BaseModel):
    """A support ticket to score. Unknown fields are rejected to catch client typos."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_TICKET]})

    ticket_id: TicketId | None = None
    subject: Subject
    body: Body
    customer_tier: CustomerTier
    product_area: ProductArea
    channel: Channel
    prior_contacts_30d: int = Field(
        default=0, ge=0, le=1_000, description="Contacts from this customer in the last 30 days"
    )
    hours_open: float = Field(
        default=0.0, ge=0.0, le=8_760.0, description="Hours since the ticket was opened"
    )


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickets: list[TicketIn] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class PredictionOut(BaseModel):
    ticket_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_band: RiskBand
    route: Route


class PredictResponse(BaseModel):
    model_version: str
    review_threshold: float
    predictions: list[PredictionOut]


class CitationOut(BaseModel):
    ref: int = Field(description="Number used in the draft, e.g. 1 for [1]")
    doc_id: str
    title: str
    section: str
    audience: str = Field(description="'customer' (quoted in the reply) or 'internal'")
    score: float = Field(description="Cosine similarity between the ticket and the passage")
    excerpt: str


class TimingsOut(BaseModel):
    inference_ms: float
    drafting_ms: float
    total_ms: float


class TriageResponse(BaseModel):
    ticket_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_band: RiskBand
    route: Route
    review_threshold: float
    model_version: str
    signals: list[str] = Field(description="Rule-based policy triggers and text cues")
    draft: str = Field(description="Customer-facing reply with [n] citations and sources")
    guidance: list[str] = Field(description="Internal notes for the agent, with citations")
    citations: list[CitationOut]
    composer: str
    review_item_id: str | None = Field(description="Set when the ticket was queued for review")
    timings: TimingsOut


class ReviewItemOut(BaseModel):
    id: str
    ticket_id: str
    status: ReviewStatus
    subject: str
    body: str
    customer_tier: CustomerTier
    product_area: ProductArea
    channel: Channel
    risk_score: float
    risk_band: RiskBand
    signals: list[str]
    draft: str
    guidance: list[str]
    citations: list[CitationOut]
    created_at: str
    resolved_at: str | None
    resolution: Resolution | None
    reviewer: str | None
    note: str | None

    @classmethod
    def from_item(cls, item: ReviewItem) -> ReviewItemOut:
        return cls.model_validate(item, from_attributes=True)


class QueueResponse(BaseModel):
    items: list[ReviewItemOut]
    pending_count: int


class ResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution: Resolution
    reviewer: Annotated[str, StringConstraints(strip_whitespace=True, max_length=80)] | None = None
    note: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2_000)] | None = None


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    model_version: str | None = None
    kb_documents: int | None = None
    reason: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    trained_at: str
    input_dim: int
    tuned_threshold: float = Field(description="F1-optimal threshold found on validation data")
    review_threshold: float = Field(description="Threshold in use (env override or tuned)")
    medium_threshold: float
    synthetic_data: bool
    test_metrics: dict[str, Any] | None = Field(
        description="Held-out test metrics from training (synthetic data)"
    )
