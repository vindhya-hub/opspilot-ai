"""Integration tests: the full HTTP flow through FastAPI's TestClient (2 tests)."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from tests.conftest import HIGH_RISK_TICKET, LOW_RISK_TICKET

pytestmark = pytest.mark.integration


def test_triage_routes_high_risk_to_review_queue_and_resolves(client: TestClient) -> None:
    low = client.post("/v1/triage", json=LOW_RISK_TICKET)
    assert low.status_code == 200
    low_body = low.json()
    assert (low_body["route"], low_body["risk_band"]) == ("auto_draft", "low")
    assert low_body["review_item_id"] is None
    assert low_body["citations"][0]["doc_id"] == "mobile-app-support"
    assert "[1]" in low_body["draft"]
    assert "Sources:" in low_body["draft"]
    assert client.get("/v1/queue").json()["pending_count"] == 0  # nothing queued

    high = client.post("/v1/triage", json=HIGH_RISK_TICKET)
    assert high.status_code == 200
    body = high.json()
    assert (body["ticket_id"], body["route"], body["risk_band"]) == (
        "T-HIGH-1",
        "human_review",
        "high",
    )
    assert body["risk_score"] >= body["review_threshold"]
    assert any(signal.startswith("SLA breached") for signal in body["signals"])
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body["draft"] + " ".join(body["guidance"]))}
    assert cited == {c["ref"] for c in body["citations"]}  # every citation is used
    assert any(c["audience"] == "internal" for c in body["citations"])  # policy guidance
    assert body["timings"]["total_ms"] > 0

    queue = client.get("/v1/queue").json()
    assert queue["pending_count"] == 1
    (item,) = queue["items"]
    assert (item["id"], item["ticket_id"], item["status"]) == (
        body["review_item_id"],
        "T-HIGH-1",
        "pending",
    )
    assert item["draft"] == body["draft"]
    assert item["citations"] == body["citations"]

    resolved = client.post(
        f"/v1/queue/{item['id']}/resolve",
        json={"resolution": "edited", "reviewer": "  sam  ", "note": "Offered a call."},
    )
    assert resolved.status_code == 200
    assert (resolved.json()["status"], resolved.json()["reviewer"]) == ("resolved", "sam")
    assert client.get("/v1/queue").json() == {"items": [], "pending_count": 0}
    history = client.get("/v1/queue", params={"status": "resolved"}).json()["items"]
    assert [i["id"] for i in history] == [item["id"]]

    again = client.post(f"/v1/queue/{item['id']}/resolve", json={"resolution": "approved"})
    assert again.status_code == 409


def test_predict_batch_readiness_and_prometheus_metrics(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    info = client.get("/v1/model").json()
    assert info["review_threshold"] == info["tuned_threshold"]
    assert info["synthetic_data"] is True
    assert 0 <= info["test_metrics"]["f1"] <= 1

    response = client.post("/v1/predict", json={"tickets": [HIGH_RISK_TICKET, LOW_RISK_TICKET]})
    assert response.status_code == 200
    high, low = response.json()["predictions"]
    assert (high["ticket_id"], high["route"]) == ("T-HIGH-1", "human_review")
    assert (low["ticket_id"], low["route"]) == ("T-LOW-1", "auto_draft")
    assert high["risk_score"] > low["risk_score"]
    assert client.get("/v1/queue").json()["pending_count"] == 0  # predict never enqueues
    client.post("/v1/triage", json=HIGH_RISK_TICKET)

    allowed = client.options(
        "/v1/triage",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    blocked = client.get("/healthz", headers={"Origin": "http://evil.test"})
    assert "access-control-allow-origin" not in blocked.headers

    metrics = client.get("/metrics")
    assert metrics.headers["content-type"].startswith("text/plain")
    text = metrics.text
    for expected in (
        'opspilot_http_request_duration_seconds_bucket{le="0.3",method="POST",route="/v1/triage"}',
        'opspilot_http_requests_total{method="POST",route="/v1/predict",status="200"} 1.0',
        'opspilot_model_inference_seconds_count{endpoint="predict"} 1.0',
        'opspilot_model_inference_seconds_count{endpoint="triage"} 1.0',
        'opspilot_triage_decisions_total{risk_band="high",route="human_review"} 1.0',
        "opspilot_review_queue_pending 1.0",
        "opspilot_model_loaded 1.0",
        "opspilot_risk_score_count 3.0",
    ):
        assert expected in text, expected
