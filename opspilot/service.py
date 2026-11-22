"""Triage orchestration: score -> route -> retrieve evidence -> draft -> enqueue for review.

Framework-free on purpose: the FastAPI layer only validates input, calls this service and
records metrics, so the same logic can run from a worker, a CLI or a notebook.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from opspilot.config import Settings
from opspilot.domain import HUMAN_REVIEW
from opspilot.guidance import Draft, DraftComposer, DraftRequest
from opspilot.model import EscalationModel
from opspilot.retrieval import INTERNAL, Evidence, Retriever
from opspilot.review_queue import NewReviewItem, ReviewItem, ReviewQueue
from opspilot.routing import RiskRouter
from opspilot.signals import Signal, detect_signals


class ModelNotLoadedError(RuntimeError):
    """Scoring was requested but no model artifacts are loaded."""


@dataclass(frozen=True)
class Prediction:
    ticket_id: str
    risk_score: float
    risk_band: str
    route: str


@dataclass(frozen=True)
class TriageResult:
    prediction: Prediction
    signals: tuple[Signal, ...]
    draft: Draft
    review_item: ReviewItem | None
    inference_seconds: float
    drafting_seconds: float


class TriageService:
    def __init__(
        self,
        *,
        model: EscalationModel | None,
        retriever: Retriever,
        queue: ReviewQueue,
        composer: DraftComposer,
        settings: Settings,
    ) -> None:
        self._model = model
        self.retriever = retriever
        self.queue = queue
        self.composer = composer
        self.settings = settings
        self._router: RiskRouter | None = None
        if model is not None:
            review_threshold = (
                settings.risk_threshold
                if settings.risk_threshold is not None
                else model.decision_threshold  # F1-optimal threshold tuned on validation
            )
            self._router = RiskRouter.from_thresholds(review_threshold, settings.medium_threshold)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _not_loaded(self) -> ModelNotLoadedError:
        return ModelNotLoadedError(
            f"model artifacts are not loaded from {self.settings.artifacts_dir}; run "
            "`python -m opspilot.train` or point OPSPILOT_ARTIFACTS_DIR at trained artifacts"
        )

    @property
    def model(self) -> EscalationModel:
        if self._model is None:
            raise self._not_loaded()
        return self._model

    @property
    def router(self) -> RiskRouter:
        if self._router is None:
            raise self._not_loaded()
        return self._router

    def predict(self, tickets: Sequence[Mapping[str, Any]]) -> tuple[list[Prediction], float]:
        """Scores and routes for a batch; also returns the model inference time (seconds)."""
        model, router = self.model, self.router
        started = time.perf_counter()
        scores = model.predict_proba(tickets)
        elapsed = time.perf_counter() - started
        predictions = []
        for ticket, score in zip(tickets, scores, strict=True):
            decision = router.decide(float(score))
            predictions.append(
                Prediction(
                    ticket_id=str(ticket.get("ticket_id") or _new_ticket_id()),
                    risk_score=decision.risk_score,
                    risk_band=decision.risk_band,
                    route=decision.route,
                )
            )
        return predictions, elapsed

    def triage(self, ticket: Mapping[str, Any]) -> TriageResult:
        ticket = {**ticket, "ticket_id": ticket.get("ticket_id") or _new_ticket_id()}
        (prediction,), inference_seconds = self.predict([ticket])
        signals = tuple(detect_signals(ticket))

        started = time.perf_counter()
        evidence = self._evidence(ticket, prediction.route, signals)
        draft = self.composer.compose(
            DraftRequest(
                subject=ticket["subject"],
                body=ticket["body"],
                customer_tier=ticket["customer_tier"],
                route=prediction.route,
                risk_band=prediction.risk_band,
                signals=signals,
                evidence=evidence,
            )
        )
        drafting_seconds = time.perf_counter() - started

        review_item = None
        if prediction.route == HUMAN_REVIEW:
            review_item = self.queue.add(
                NewReviewItem(
                    ticket_id=prediction.ticket_id,
                    subject=ticket["subject"],
                    body=ticket["body"],
                    customer_tier=ticket["customer_tier"],
                    product_area=ticket["product_area"],
                    channel=ticket["channel"],
                    risk_score=round(prediction.risk_score, 4),
                    risk_band=prediction.risk_band,
                    signals=tuple(s.label for s in signals),
                    draft=draft.reply,
                    guidance=draft.guidance,
                    citations=tuple(asdict(c) for c in draft.citations),
                )
            )
        return TriageResult(
            prediction, signals, draft, review_item, inference_seconds, drafting_seconds
        )

    def _evidence(
        self, ticket: Mapping[str, Any], route: str, signals: Sequence[Signal]
    ) -> tuple[Evidence, ...]:
        """Top-k runbook passages; human-review cases always get an internal policy passage."""
        query = f"{ticket['subject']}. {ticket['body']} {ticket['product_area'].replace('_', ' ')}"
        evidence = self.retriever.retrieve(query, self.settings.retrieval_top_k)
        if route == HUMAN_REVIEW and not any(e.passage.audience == INTERNAL for e in evidence):
            policy_query = " ".join([query, "escalation", *(s.label for s in signals)])
            evidence += self.retriever.retrieve(policy_query, top_k=1, audience=INTERNAL)
        return tuple(evidence)


def _new_ticket_id() -> str:
    return f"T-{uuid.uuid4().hex[:8]}"
