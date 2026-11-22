"""Prometheus metrics for the API, kept in a per-app registry (no global state).

Histogram buckets include 0.3 s so the p95 > 300 ms alert in ``prometheus/alerts.yml``
sits on a bucket boundary, where ``histogram_quantile`` is exact rather than interpolated.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0)
INFERENCE_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class ApiMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "opspilot_http_requests_total",
            "HTTP requests by method, route template and status code.",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "opspilot_http_request_duration_seconds",
            "HTTP request latency by method and route template.",
            ["method", "route"],
            buckets=HTTP_BUCKETS,
            registry=self.registry,
        )
        self.inference_latency = Histogram(
            "opspilot_model_inference_seconds",
            "Model inference latency per call (feature pipeline + MLP forward pass).",
            ["endpoint"],
            buckets=INFERENCE_BUCKETS,
            registry=self.registry,
        )
        self.decisions = Counter(
            "opspilot_triage_decisions_total",
            "Routing decisions made by /v1/triage.",
            ["route", "risk_band"],
            registry=self.registry,
        )
        self.risk_scores = Histogram(
            "opspilot_risk_score",
            "Distribution of predicted escalation risk (watch for drift).",
            buckets=SCORE_BUCKETS,
            registry=self.registry,
        )
        self.resolutions = Counter(
            "opspilot_review_resolutions_total",
            "Review items resolved, by resolution.",
            ["resolution"],
            registry=self.registry,
        )
        self.queue_pending = Gauge(
            "opspilot_review_queue_pending",
            "Tickets waiting for human review.",
            registry=self.registry,
        )
        self.model_loaded = Gauge(
            "opspilot_model_loaded",
            "1 when model artifacts are loaded and the service is ready, else 0.",
            registry=self.registry,
        )
        self.model_info = Info(
            "opspilot_model", "Metadata of the loaded model.", registry=self.registry
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)

    content_type = CONTENT_TYPE_LATEST
