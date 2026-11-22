"""Shared fixtures: a tiny model trained once per session through the real training CLI."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from opspilot import train
from opspilot.api.app import create_app
from opspilot.config import Settings
from opspilot.model import EscalationModel
from opspilot.review_queue import InMemoryReviewQueue

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "kb"

#: Enterprise customer, SLA long breached, repeat contacts, threats: clearly escalating.
HIGH_RISK_TICKET: dict[str, Any] = {
    "ticket_id": "T-HIGH-1",
    "subject": "Re: Charged twice this month",
    "body": (
        "Customer posted about this publicly on social media. I was charged twice for our "
        "Scale subscription this month and invoice INV-55120 shows a duplicate charge of $2,100. "
        "Our payroll run is blocked until this is fixed. This is the third time I am contacting "
        "you about this. This is completely unacceptable. If this is not fixed I will dispute "
        "the charge with my bank."
    ),
    "customer_tier": "enterprise",
    "product_area": "billing",
    "channel": "social",
    "prior_contacts_30d": 5,
    "hours_open": 30.0,
}

#: Free tier, cosmetic issue, calm tone, freshly opened: clearly routine.
LOW_RISK_TICKET: dict[str, Any] = {
    "ticket_id": "T-LOW-1",
    "subject": "Dark mode text unreadable",
    "body": (
        "In dark mode some text in the mobile app is unreadable on my Pixel 8. Not urgent, just "
        "wanted to flag it. Thanks for the great product."
    ),
    "customer_tier": "free",
    "product_area": "mobile_app",
    "channel": "email",
    "prior_contacts_30d": 0,
    "hours_open": 1.0,
}


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never see the developer's OPSPILOT_* variables (settings also skip .env)."""
    for name in list(os.environ):
        if name.startswith("OPSPILOT_"):
            monkeypatch.delenv(name)


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a small model through the CLI (a few seconds) and return its artifact dir."""
    output_dir = tmp_path_factory.mktemp("artifacts")
    args = ["--output-dir", str(output_dir), "--n-samples", "2000", "--max-epochs", "8"]
    exit_code = train.main([*args, "--max-features", "600"])
    assert exit_code == 0
    return output_dir


@pytest.fixture(scope="session")
def tiny_model(trained_artifacts: Path) -> EscalationModel:
    return EscalationModel.load(trained_artifacts)


@pytest.fixture
def make_settings(trained_artifacts: Path) -> Callable[..., Settings]:
    """Settings with test defaults; keyword arguments override individual fields."""

    def factory(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "artifacts_dir": trained_artifacts,
            "kb_dir": KB_DIR,
            "queue_backend": "memory",
            "cors_origins": "http://localhost:5173",
        }
        return Settings(_env_file=None, **{**values, **overrides})

    return factory


@pytest.fixture
def client(
    make_settings: Callable[..., Settings], tiny_model: EscalationModel
) -> Iterator[TestClient]:
    app = create_app(make_settings(), model=tiny_model, queue=InMemoryReviewQueue())
    with TestClient(app) as test_client:
        yield test_client
