"""Failure paths: bad input, missing model, oversized payloads, unknown ids (4 tests)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opspilot.api.app import create_app
from opspilot.api.schemas import MAX_BATCH_SIZE, MAX_BODY_CHARS, MAX_SUBJECT_CHARS
from opspilot.review_queue import InMemoryReviewQueue
from tests.conftest import HIGH_RISK_TICKET, LOW_RISK_TICKET

pytestmark = pytest.mark.failure


def _error_fields(response) -> set[str]:
    return {str(error["loc"][-1]) for error in response.json()["detail"]}


def test_invalid_payloads_return_422(client: TestClient) -> None:
    missing = client.post("/v1/triage", json={"subject": "Help"})
    assert missing.status_code == 422
    assert {"body", "customer_tier", "product_area", "channel"} <= _error_fields(missing)

    invalid = {
        "customer_tier": "gold",  # not a known tier
        "channel": "fax",
        "prior_contacts_30d": -1,
        "hours_open": "soon",
        "ticket_id": "bad id with spaces",
    }
    for field, value in invalid.items():
        response = client.post("/v1/triage", json={**LOW_RISK_TICKET, field: value})
        assert response.status_code == 422, field
        assert field in _error_fields(response)

    typo = client.post("/v1/triage", json={**LOW_RISK_TICKET, "prior_contacts": 2})
    assert typo.status_code == 422  # unknown fields are rejected, not silently ignored
    malformed = client.post(
        "/v1/triage", content=b'{"subject": ', headers={"content-type": "application/json"}
    )
    assert malformed.status_code == 422

    assert client.post("/v1/predict", json={"tickets": []}).status_code == 422
    too_many = {"tickets": [LOW_RISK_TICKET] * (MAX_BATCH_SIZE + 1)}
    assert client.post("/v1/predict", json=too_many).status_code == 422
    assert client.get("/v1/queue", params={"status": "archived"}).status_code == 422
    assert client.get("/v1/queue", params={"limit": 0}).status_code == 422

    queued = client.post("/v1/triage", json=HIGH_RISK_TICKET).json()["review_item_id"]
    bad_resolution = client.post(f"/v1/queue/{queued}/resolve", json={"resolution": "maybe"})
    assert bad_resolution.status_code == 422
    assert client.get("/v1/queue").json()["pending_count"] == 1  # nothing was changed


def test_model_not_loaded_returns_503(make_settings, tmp_path: Path) -> None:
    settings = make_settings(artifacts_dir=tmp_path / "no-artifacts-here")
    with TestClient(create_app(settings, queue=InMemoryReviewQueue())) as client:
        assert client.get("/healthz").status_code == 200  # alive, just not ready
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["status"] == "not_ready"

        for method, path, payload in (
            ("post", "/v1/triage", LOW_RISK_TICKET),
            ("post", "/v1/predict", {"tickets": [LOW_RISK_TICKET]}),
            ("get", "/v1/model", None),
        ):
            response = client.post(path, json=payload) if method == "post" else client.get(path)
            assert response.status_code == 503, path
            assert "python -m opspilot.train" in response.json()["detail"]

        assert client.get("/v1/queue").status_code == 200  # reviewers can still work
        metrics = client.get("/metrics").text
        assert "opspilot_model_loaded 0.0" in metrics
        assert 'route="/v1/triage",status="503"} 1.0' in metrics


def test_empty_and_oversized_text_is_rejected(
    client: TestClient, make_settings, tiny_model
) -> None:
    for field, value in (
        ("subject", ""),
        ("subject", "   \n\t "),  # whitespace only is empty after stripping
        ("body", ""),
        ("subject", "s" * (MAX_SUBJECT_CHARS + 1)),
        ("body", "b" * (MAX_BODY_CHARS + 1)),
    ):
        response = client.post("/v1/triage", json={**LOW_RISK_TICKET, field: value})
        assert response.status_code == 422, (field, len(value))
        assert field in _error_fields(response)

    at_limit = client.post("/v1/triage", json={**LOW_RISK_TICKET, "body": "b" * MAX_BODY_CHARS})
    assert at_limit.status_code == 200

    small = make_settings(max_request_bytes=4096)
    with TestClient(create_app(small, model=tiny_model, queue=InMemoryReviewQueue())) as capped:
        payload = json.dumps({**LOW_RISK_TICKET, "body": "b" * MAX_BODY_CHARS})
        response = capped.post(
            "/v1/triage", content=payload, headers={"content-type": "application/json"}
        )
        assert response.status_code == 413  # rejected before the body is parsed
        assert "4096" in response.json()["detail"]


def test_unknown_queue_item_returns_404(client: TestClient) -> None:
    response = client.post("/v1/queue/rv_does_not_exist/resolve", json={"resolution": "approved"})
    assert response.status_code == 404
    assert "rv_does_not_exist" in response.json()["detail"]

    too_long = client.post(f"/v1/queue/{'x' * 65}/resolve", json={"resolution": "approved"})
    assert too_long.status_code == 422
    assert client.get("/v1/nope").status_code == 404

    metrics = client.get("/metrics").text
    assert 'route="/v1/queue/{item_id}/resolve",status="404"} 1.0' in metrics
    assert 'route="<unmatched>",status="404"} 1.0' in metrics
