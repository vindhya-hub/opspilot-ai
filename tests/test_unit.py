"""Unit tests: one component at a time (8 tests)."""

from __future__ import annotations

import json
import math
import re
import secrets
from pathlib import Path

import pytest
from pydantic import ValidationError
from scipy import sparse

from opspilot.config import Settings
from opspilot.data import generator
from opspilot.domain import CHANNELS, PRODUCT_AREAS, SLA_HOURS, TIERS
from opspilot.features import build_feature_pipeline, tickets_to_frame
from opspilot.guidance import DraftRequest, LLMComposer, TemplateComposer, build_composer
from opspilot.model import METRICS_FILE, EscalationModel
from opspilot.retrieval import INTERNAL, Retriever, parse_runbook
from opspilot.review_queue import (
    InMemoryReviewQueue,
    NewReviewItem,
    ReviewItemAlreadyResolvedError,
    ReviewItemNotFoundError,
    SqliteReviewQueue,
    create_review_queue,
)
from opspilot.routing import RiskRouter
from opspilot.service import TriageService
from opspilot.signals import detect_signals
from tests.conftest import HIGH_RISK_TICKET, KB_DIR, LOW_RISK_TICKET

pytestmark = pytest.mark.unit


def _draft_request(retriever: Retriever, ticket: dict, route: str, band: str) -> DraftRequest:
    query = f"{ticket['subject']}. {ticket['body']}"
    evidence = retriever.retrieve(query) + retriever.retrieve(query, top_k=1, audience=INTERNAL)
    return DraftRequest(
        subject=ticket["subject"],
        body=ticket["body"],
        customer_tier=ticket["customer_tier"],
        route=route,
        risk_band=band,
        signals=tuple(detect_signals(ticket)),
        evidence=tuple(evidence),
    )


def test_generator_is_deterministic_and_seeded(tmp_path: Path) -> None:
    first = generator.generate_tickets(300, seed=7)
    assert first == generator.generate_tickets(300, seed=7)
    assert first != generator.generate_tickets(300, seed=8)

    labels = [t["escalated"] for t in first]
    assert set(labels) == {0, 1}
    assert 0.1 < sum(labels) / len(labels) < 0.5  # imbalanced, but not degenerate
    for ticket in first:
        assert ticket["customer_tier"] in TIERS
        assert ticket["product_area"] in PRODUCT_AREAS
        assert ticket["channel"] in CHANNELS
        assert ticket["sla_breached"] == (ticket["hours_open"] > SLA_HOURS[ticket["customer_tier"]])
        assert "latent" not in ticket  # hidden generative state is opt-in only

    out = tmp_path / "tickets.jsonl"
    assert generator.main(["--n", "5", "--seed", "7", "--out", str(out)]) == 0
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert written == generator.generate_tickets(5, seed=7)
    with pytest.raises(ValueError, match="positive"):
        generator.generate_tickets(0)


def test_feature_pipeline_derives_sla_features_and_ignores_unseen_categories() -> None:
    tickets = generator.generate_tickets(400, seed=3)
    pipeline = build_feature_pipeline(max_features=250)
    matrix = pipeline.fit_transform(tickets)
    assert sparse.issparse(matrix)
    vocabulary = pipeline.named_steps["columns"].named_transformers_["text"].vocabulary_
    assert matrix.shape == (400, len(vocabulary) + len(TIERS + PRODUCT_AREAS + CHANNELS) + 4)
    assert not any(re.search(r"\d", term) for term in vocabulary)  # no invoice ids/amounts

    frame = tickets_to_frame([{**LOW_RISK_TICKET, "customer_tier": "enterprise", "hours_open": 6}])
    row = frame.iloc[0]
    assert row["sla_ratio"] == pytest.approx(6 / SLA_HOURS["enterprise"])
    assert row["sla_breached"] == 1.0
    assert row["log_hours_open"] == pytest.approx(math.log1p(6))

    unseen = {**LOW_RISK_TICKET, "channel": "carrier_pigeon", "product_area": "teleportation"}
    assert pipeline.transform([unseen]).shape == (1, matrix.shape[1])  # ignored, not an error


def test_model_artifacts_roundtrip_and_report_metrics(
    trained_artifacts: Path, tiny_model: EscalationModel, tmp_path: Path
) -> None:
    metrics = json.loads((trained_artifacts / METRICS_FILE).read_text(encoding="utf-8"))
    assert metrics["data"]["synthetic"] is True
    assert sum(metrics["data"]["splits"].values()) == metrics["data"]["n_samples"] == 2000
    for split in ("validation", "test"):
        assert 0.0 <= metrics[split]["f1"] <= 1.0
        assert set(metrics[split]) >= {"precision", "recall", "roc_auc", "confusion_matrix"}
    assert metrics["test"]["threshold"] == tiny_model.decision_threshold
    assert metrics["model"]["version"] == tiny_model.version

    reloaded = EscalationModel.load(trained_artifacts)
    scores = tiny_model.predict_proba([HIGH_RISK_TICKET, LOW_RISK_TICKET])
    assert reloaded.predict_proba([HIGH_RISK_TICKET, LOW_RISK_TICKET]) == pytest.approx(scores)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > tiny_model.decision_threshold > scores[1]

    with pytest.raises(FileNotFoundError, match=re.escape("python -m opspilot.train")):
        EscalationModel.load(tmp_path)


def test_retriever_ranks_the_matching_runbook_first() -> None:
    retriever = Retriever.from_directory(KB_DIR)
    assert len(retriever.documents) == 10
    internal = {p.doc_id for p in retriever.passages if p.audience == INTERNAL}
    assert internal == {"sla-escalation-policy", "customer-communication"}

    cases = {
        "I was charged twice and my invoice shows a duplicate charge": (
            "refunds-and-plan-changes",
            "Duplicate charges",
        ),
        "We got a suspicious login alert and think the account is compromised": (
            "account-security",
            "Suspicious login alerts",
        ),
        "Our webhook endpoint keeps receiving 401 errors": (
            "integrations-troubleshooting",
            "Webhook delivery failures",
        ),
        "The export of our audit log has been stuck at 0% for hours": (
            "data-exports-and-privacy",
            "Slow or stuck exports",
        ),
    }
    for query, expected in cases.items():
        best, score = retriever.search(query)[0]
        assert (best.doc_id, best.section) == expected
        assert score >= retriever.min_score

    escalation = retriever.search("customer threatens to cancel the contract", audience=INTERNAL)
    assert escalation
    assert all(p.audience == INTERNAL for p, _ in escalation)
    assert retriever.search("zzzz qqqq xxxx") == []  # nothing relevant -> no citations
    with pytest.raises(ValueError, match="Title"):
        parse_runbook("broken", "## Section\nText without a title.")
    with pytest.raises(ValueError, match="audience"):
        parse_runbook("broken", "---\naudience: everyone\n---\n# T\n## S\nText.")


def test_router_threshold_boundaries_and_configuration(
    tiny_model: EscalationModel, make_settings
) -> None:
    router = RiskRouter.from_thresholds(0.5)
    assert router.medium_threshold == pytest.approx(0.3)
    expectations = {
        1.0: ("high", "human_review"),
        0.5: ("high", "human_review"),  # the threshold itself goes to a human
        0.4999: ("medium", "auto_draft"),
        0.3: ("medium", "auto_draft"),
        0.2999: ("low", "auto_draft"),
        0.0: ("low", "auto_draft"),
    }
    for score, (band, route) in expectations.items():
        decision = router.decide(score)
        assert (decision.risk_band, decision.route) == (band, route), score

    for bad_score in (-0.1, 1.2, float("nan")):
        with pytest.raises(ValueError, match="probability"):
            router.decide(bad_score)
    with pytest.raises(ValueError, match="review_threshold"):
        RiskRouter(0.0, 0.0)
    with pytest.raises(ValueError, match="medium_threshold"):
        RiskRouter(0.5, 0.6)

    def service_router(**overrides) -> RiskRouter:
        service = TriageService(
            model=tiny_model,
            retriever=Retriever.from_directory(KB_DIR),
            queue=InMemoryReviewQueue(),
            composer=TemplateComposer(),
            settings=make_settings(**overrides),
        )
        return service.router

    assert service_router().review_threshold == tiny_model.decision_threshold
    configured = service_router(risk_threshold=0.9, medium_threshold=0.5)
    assert (configured.review_threshold, configured.medium_threshold) == (0.9, 0.5)


def test_template_draft_cites_retrieved_sources() -> None:
    retriever = Retriever.from_directory(KB_DIR)
    request = _draft_request(retriever, HIGH_RISK_TICKET, "human_review", "high")
    draft = TemplateComposer().compose(request)

    refs = [c.ref for c in draft.citations]
    assert refs == list(range(1, len(refs) + 1))
    customer = [c for c in draft.citations if c.audience == "customer"]
    internal = [c for c in draft.citations if c.audience == INTERNAL]
    assert customer
    assert internal
    assert max(c.ref for c in customer) < min(c.ref for c in internal)  # reply sources first

    cited_in_reply = {int(n) for n in re.findall(r"\[(\d+)\]", draft.reply)}
    assert cited_in_reply == {c.ref for c in customer}
    assert customer[0].doc_id == "refunds-and-plan-changes"
    assert "[1] Refunds and Plan Changes - Duplicate charges" in draft.reply
    assert '"Charged twice this month"' in draft.reply  # "Re:" prefix dropped
    assert "senior support specialist" in draft.reply
    assert all(
        note.endswith(f"[{c.ref}]") for note, c in zip(draft.guidance, internal, strict=True)
    )

    calm = TemplateComposer().compose(
        _draft_request(retriever, LOW_RISK_TICKET, "auto_draft", "low")
    )
    assert "within 72 hours" in calm.reply  # free-tier SLA window
    empty = TemplateComposer().compose(
        DraftRequest("Hello", "Hi", "pro", "auto_draft", "low", (), ())
    )
    assert empty.citations == ()
    assert "[1]" not in empty.reply


def test_settings_from_env_and_optional_llm_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPSPILOT_RISK_THRESHOLD", "0.7")
    monkeypatch.setenv("OPSPILOT_CORS_ORIGINS", "http://a.test, http://b.test")
    monkeypatch.setenv("OPSPILOT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OPSPILOT_PORT", "9001")
    monkeypatch.setenv("OPSPILOT_MEDIUM_THRESHOLD", "")  # empty -> default
    settings = Settings(_env_file=None)
    assert settings.risk_threshold == 0.7
    assert settings.medium_threshold is None
    assert settings.cors_origin_list == ("http://a.test", "http://b.test")
    assert (settings.log_level, settings.port, settings.composer) == ("debug", 9001, "template")
    assert isinstance(build_composer(settings), TemplateComposer)
    monkeypatch.setenv("OPSPILOT_RISK_THRESHOLD", "1.5")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

    # The LLM hook is off by default and refuses to start without its key.
    with pytest.raises(ValidationError, match="OPSPILOT_LLM_API_KEY"):
        Settings(_env_file=None, risk_threshold=0.5, composer="llm")
    key = f"test-{secrets.token_hex(16)}"  # ephemeral, never a real credential
    llm_settings = Settings(
        _env_file=None,
        risk_threshold=0.5,
        composer="llm",
        llm_api_key=key,
        llm_base_url="https://llm.invalid/v1",
        llm_model="test-model",
    )
    assert key not in repr(llm_settings)
    assert key not in str(llm_settings.model_dump())
    assert isinstance(build_composer(llm_settings), LLMComposer)

    calls = []

    def fake_transport(url, payload, headers, timeout):
        calls.append((url, payload, headers))
        return {"choices": [{"message": {"content": "Hi there, the duplicate is refunded [1]."}}]}

    request = _draft_request(
        Retriever.from_directory(KB_DIR), HIGH_RISK_TICKET, "human_review", "high"
    )
    composer = LLMComposer(
        base_url=str(llm_settings.llm_base_url),
        model="test-model",
        api_key=llm_settings.llm_api_key,
        transport=fake_transport,
    )
    draft = composer.compose(request)
    assert draft.composer == "llm"
    assert draft.reply.startswith("Hi there, the duplicate")
    assert "[1] Refunds and Plan Changes - Duplicate charges" in draft.reply
    url, payload, headers = calls[0]
    assert url == "https://llm.invalid/v1/chat/completions"
    assert headers["Authorization"] == f"Bearer {key}"
    assert payload["model"] == "test-model"
    assert "Duplicate charges" in payload["messages"][1]["content"]  # grounded in evidence

    def failing_transport(*args):
        raise OSError("network down")

    def inventing_transport(*args):
        return {"choices": [{"message": {"content": "Refunded, see policy [9]."}}]}

    for transport in (failing_transport, inventing_transport):
        fallback = LLMComposer(
            base_url="https://llm.invalid/v1",
            model="m",
            api_key=llm_settings.llm_api_key,
            transport=transport,
        )
        assert fallback.compose(request).composer == "template"


def test_review_queue_backends_share_one_contract(tmp_path: Path, make_settings) -> None:
    def new_item(subject: str, risk: float) -> NewReviewItem:
        return NewReviewItem(
            "T-1",
            subject,
            "body",
            "pro",
            "billing",
            "email",
            risk,
            "high",
            ("signal",),
            "draft [1]",
            ("note [2]",),
            ({"ref": 1},),
        )

    db_path = tmp_path / "queue.db"
    for queue in (InMemoryReviewQueue(), SqliteReviewQueue(db_path)):
        high = queue.add(new_item("high", 0.97))
        low = queue.add(new_item("low", 0.61))
        queue.add(new_item("medium", 0.80))
        assert [i.subject for i in queue.list_items()] == ["high", "medium", "low"]  # riskiest
        assert [i.subject for i in queue.list_items(limit=1)] == ["high"]
        assert (queue.count(), queue.get(low.id)) == (3, low)

        resolved = queue.resolve(high.id, "edited", reviewer="sam", note="softened tone")
        assert (resolved.status, resolved.resolution, resolved.reviewer) == (
            "resolved",
            "edited",
            "sam",
        )
        assert resolved.resolved_at is not None
        with pytest.raises(ReviewItemAlreadyResolvedError):
            queue.resolve(high.id, "approved")
        with pytest.raises(ReviewItemNotFoundError):
            queue.resolve("rv_missing", "approved")
        with pytest.raises(ReviewItemNotFoundError):
            queue.get("rv_missing")

        assert [i.subject for i in queue.list_items("resolved")] == ["high"]
        assert [i.subject for i in queue.list_items(None)] == ["medium", "low", "high"]  # newest
        assert (queue.count("pending"), queue.count("resolved"), queue.count(None)) == (2, 1, 3)
        queue.close()

    reopened = SqliteReviewQueue(db_path)  # SQLite persists across restarts
    assert reopened.get(high.id) == resolved
    assert reopened.count(None) == 3
    reopened.close()
    assert isinstance(create_review_queue(make_settings()), InMemoryReviewQueue)
    sqlite_queue = create_review_queue(
        make_settings(queue_backend="sqlite", queue_db_path=tmp_path / "x.db")
    )
    assert isinstance(sqlite_queue, SqliteReviewQueue)
    sqlite_queue.close()
