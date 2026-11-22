"""FastAPI application: model inference, cited drafts, the review queue and ops endpoints.

Serve it with ``python -m opspilot.api`` (host, port and log level come from ``OPSPILOT_*``
settings) or ``uvicorn opspilot.api.app:create_app --factory``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import torch
from fastapi import APIRouter, Depends, FastAPI, Path, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from opspilot import __version__
from opspilot.api.metrics import ApiMetrics
from opspilot.api.schemas import (
    CitationOut,
    HealthResponse,
    ModelInfoResponse,
    PredictionOut,
    PredictRequest,
    PredictResponse,
    QueueResponse,
    ReadinessResponse,
    ResolveRequest,
    ReviewItemOut,
    TicketIn,
    TimingsOut,
    TriageResponse,
)
from opspilot.config import Settings
from opspilot.guidance import DraftComposer, build_composer
from opspilot.model import METRICS_FILE, EscalationModel
from opspilot.retrieval import Retriever
from opspilot.review_queue import (
    ReviewItemAlreadyResolvedError,
    ReviewItemNotFoundError,
    ReviewQueue,
    create_review_queue,
)
from opspilot.service import ModelNotLoadedError, TriageService

log = logging.getLogger("opspilot.api")
router = APIRouter()


def get_service(request: Request) -> TriageService:
    return request.app.state.service


def get_metrics(request: Request) -> ApiMetrics:
    return request.app.state.metrics


ServiceDep = Annotated[TriageService, Depends(get_service)]
MetricsDep = Annotated[ApiMetrics, Depends(get_metrics)]


@router.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz() -> HealthResponse:
    """Liveness: the process is up (it may still be waiting for model artifacts)."""
    return HealthResponse(status="ok")


@router.get(
    "/readyz",
    response_model=ReadinessResponse,
    tags=["ops"],
    responses={503: {"model": ReadinessResponse, "description": "Model not loaded"}},
)
def readyz(service: ServiceDep, response: Response) -> ReadinessResponse:
    """Readiness: a model is loaded and the knowledge base is indexed."""
    kb_documents = len(service.retriever.documents)
    if not service.ready:
        response.status_code = 503
        return ReadinessResponse(
            status="not_ready", kb_documents=kb_documents, reason="model artifacts not loaded"
        )
    return ReadinessResponse(
        status="ready", model_version=service.model.version, kb_documents=kb_documents
    )


@router.get("/metrics", tags=["ops"], response_class=Response)
def prometheus_metrics(metrics: MetricsDep) -> Response:
    """Prometheus exposition format."""
    return Response(metrics.render(), media_type=metrics.content_type)


@router.get("/v1/model", response_model=ModelInfoResponse, tags=["model"])
def model_info(service: ServiceDep, request: Request) -> ModelInfoResponse:
    model, risk_router = service.model, service.router
    training = request.app.state.training_metrics or {}
    return ModelInfoResponse(
        model_version=model.version,
        trained_at=model.metadata.trained_at,
        input_dim=model.metadata.input_dim,
        tuned_threshold=model.decision_threshold,
        review_threshold=risk_router.review_threshold,
        medium_threshold=risk_router.medium_threshold,
        synthetic_data=bool(training.get("data", {}).get("synthetic", True)),
        test_metrics=training.get("test"),
    )


@router.post("/v1/predict", response_model=PredictResponse, tags=["model"])
def predict(body: PredictRequest, service: ServiceDep, metrics: MetricsDep) -> PredictResponse:
    """Score a batch of tickets (no drafting, nothing is queued)."""
    predictions, seconds = service.predict([ticket.model_dump() for ticket in body.tickets])
    metrics.inference_latency.labels("predict").observe(seconds)
    for prediction in predictions:
        metrics.risk_scores.observe(prediction.risk_score)
    return PredictResponse(
        model_version=service.model.version,
        review_threshold=service.router.review_threshold,
        predictions=[
            PredictionOut(
                ticket_id=p.ticket_id,
                risk_score=round(p.risk_score, 4),
                risk_band=p.risk_band,
                route=p.route,
            )
            for p in predictions
        ],
    )


@router.post("/v1/triage", response_model=TriageResponse, tags=["triage"])
def triage(ticket: TicketIn, service: ServiceDep, metrics: MetricsDep) -> TriageResponse:
    """Score one ticket, draft a cited reply and queue it for review if the risk is high."""
    started = time.perf_counter()
    result = service.triage(ticket.model_dump())
    prediction, draft = result.prediction, result.draft
    metrics.inference_latency.labels("triage").observe(result.inference_seconds)
    metrics.risk_scores.observe(prediction.risk_score)
    metrics.decisions.labels(prediction.route, prediction.risk_band).inc()
    if result.review_item is not None:
        metrics.queue_pending.set(service.queue.count("pending"))
    return TriageResponse(
        ticket_id=prediction.ticket_id,
        risk_score=round(prediction.risk_score, 4),
        risk_band=prediction.risk_band,
        route=prediction.route,
        review_threshold=service.router.review_threshold,
        model_version=service.model.version,
        signals=[signal.label for signal in result.signals],
        draft=draft.reply,
        guidance=list(draft.guidance),
        citations=[CitationOut.model_validate(c, from_attributes=True) for c in draft.citations],
        composer=draft.composer,
        review_item_id=result.review_item.id if result.review_item else None,
        timings=TimingsOut(
            inference_ms=round(result.inference_seconds * 1000, 2),
            drafting_ms=round(result.drafting_seconds * 1000, 2),
            total_ms=round((time.perf_counter() - started) * 1000, 2),
        ),
    )


@router.get("/v1/queue", response_model=QueueResponse, tags=["review"])
def list_queue(
    service: ServiceDep,
    status: Literal["pending", "resolved", "all"] = "pending",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> QueueResponse:
    """Review items: pending ones highest risk first, resolved or all newest first."""
    items = service.queue.list_items(None if status == "all" else status, limit)
    return QueueResponse(
        items=[ReviewItemOut.from_item(item) for item in items],
        pending_count=service.queue.count("pending"),
    )


@router.post(
    "/v1/queue/{item_id}/resolve",
    response_model=ReviewItemOut,
    tags=["review"],
    responses={404: {"description": "Unknown review item"}, 409: {"description": "Resolved"}},
)
def resolve_item(
    item_id: Annotated[str, Path(min_length=1, max_length=64)],
    body: ResolveRequest,
    service: ServiceDep,
    metrics: MetricsDep,
) -> ReviewItemOut:
    item = service.queue.resolve(
        item_id, body.resolution, reviewer=body.reviewer or None, note=body.note or None
    )
    metrics.resolutions.labels(body.resolution).inc()
    metrics.queue_pending.set(service.queue.count("pending"))
    return ReviewItemOut.from_item(item)


def _error_handler(status_code: int) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"detail": str(exc).strip("'\"")})

    return handle


def _route_template(request: Request) -> str:
    """Route path template (``/v1/queue/{item_id}/resolve``), keeping label cardinality low.

    Requests that never reached a route (unknown paths, bodies rejected before routing)
    share the ``<unmatched>`` label.
    """
    path = getattr(request.scope.get("route"), "path", None)
    return path if isinstance(path, str) else "<unmatched>"


def _load_model(settings: Settings) -> EscalationModel | None:
    try:
        return EscalationModel.load(settings.artifacts_dir)
    except FileNotFoundError as exc:
        log.warning("starting without a model, /readyz will report not ready: %s", exc)
        return None


def _load_training_metrics(settings: Settings) -> dict[str, Any] | None:
    path = settings.artifacts_dir / METRICS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def create_app(
    settings: Settings | None = None,
    *,
    model: EscalationModel | None = None,
    queue: ReviewQueue | None = None,
    composer: DraftComposer | None = None,
) -> FastAPI:
    """Build the application.

    Anything not injected is created from ``settings`` at startup: the model from
    ``OPSPILOT_ARTIFACTS_DIR`` (missing artifacts leave the service "not ready" instead of
    crashing), the queue from ``OPSPILOT_QUEUE_BACKEND`` and the composer from
    ``OPSPILOT_COMPOSER``. The knowledge base must load, or startup fails.
    """
    settings = settings or Settings()
    metrics = ApiMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        torch.set_num_threads(settings.torch_threads)
        loaded = model if model is not None else _load_model(settings)
        review_queue = queue if queue is not None else create_review_queue(settings)
        service = TriageService(
            model=loaded,
            retriever=Retriever.from_directory(settings.kb_dir),
            queue=review_queue,
            composer=composer or build_composer(settings),
            settings=settings,
        )
        app.state.service = service
        app.state.training_metrics = _load_training_metrics(settings) if loaded else None
        metrics.model_loaded.set(1 if service.ready else 0)
        metrics.queue_pending.set(review_queue.count("pending"))
        if loaded is not None:
            threshold = service.router.review_threshold
            metrics.model_info.info({"version": loaded.version, "review_threshold": f"{threshold}"})
            log.info(
                "model %s loaded (review threshold %.2f, composer %s, %d runbooks indexed)",
                loaded.version,
                threshold,
                service.composer.name,
                len(service.retriever.documents),
            )
        try:
            yield
        finally:
            review_queue.close()

    app = FastAPI(
        title="OpsPilot AI",
        version=__version__,
        summary="Escalation-risk triage for support tickets (model trained on synthetic data).",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.metrics = metrics
    app.include_router(router)
    app.add_exception_handler(ModelNotLoadedError, _error_handler(503))
    app.add_exception_handler(ReviewItemNotFoundError, _error_handler(404))
    app.add_exception_handler(ReviewItemAlreadyResolvedError, _error_handler(409))

    @app.middleware("http")
    async def observe_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started, status = time.perf_counter(), 500
        try:
            declared = request.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > settings.max_request_bytes:
                status = 413
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"request body exceeds {settings.max_request_bytes} bytes"},
                )
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = _route_template(request)
            metrics.requests.labels(request.method, route, str(status)).inc()
            metrics.request_latency.labels(request.method, route).observe(
                time.perf_counter() - started
            )

    # Added last, so it is the outermost middleware and also covers error responses.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origin_list),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=600,
    )
    return app
