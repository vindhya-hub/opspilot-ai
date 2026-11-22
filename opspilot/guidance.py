"""Draft replies and agent guidance that cite the runbooks they rely on.

:class:`TemplateComposer` (the default) is fully offline. It quotes the most relevant
sentence of each retrieved *customer-facing* passage in the reply, marks each one with a
numbered citation ([1], [2], ...) and lists the sources (runbook title and section) under
the reply. Passages from *internal* runbooks become cited guidance notes for the agent.

:class:`LLMComposer` is an optional hook, disabled by default. With
``OPSPILOT_COMPOSER=llm`` it sends the same retrieved evidence to an OpenAI-compatible
``/chat/completions`` endpoint, accepts the reply only if it cites known sources, and falls
back to the template draft on any error. Its key is read from ``OPSPILOT_LLM_API_KEY`` only
(see :mod:`opspilot.config`); nothing calls out to a network unless that is configured.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import SecretStr

from opspilot.config import Settings
from opspilot.domain import HUMAN_REVIEW, sla_hours_for
from opspilot.retrieval import CUSTOMER, INTERNAL, Evidence
from opspilot.signals import Signal

log = logging.getLogger(__name__)

_SUBJECT_PREFIX = re.compile(r"^(?:(?:re|fwd?|urgent)\s*:\s*)+", re.IGNORECASE)
_CITATION_MARK = re.compile(r"\[(\d+)\]")

LLM_SYSTEM_PROMPT = (
    "You draft replies for a customer support team. Use only the numbered sources you are "
    "given. Cite each factual statement with its source number in square brackets, e.g. [1]. "
    "Do not invent policies, prices, timelines or promises. Write at most 180 words, start "
    "with 'Hi there,' and do not add a list of sources."
)


@dataclass(frozen=True)
class Citation:
    ref: int
    doc_id: str
    title: str
    section: str
    audience: str
    score: float
    excerpt: str


@dataclass(frozen=True)
class DraftRequest:
    subject: str
    body: str
    customer_tier: str
    route: str
    risk_band: str
    signals: tuple[Signal, ...]
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Draft:
    #: Customer-facing reply with [n] markers and a "Sources:" footer.
    reply: str
    #: Internal notes for the agent or reviewer, each ending with its [n] marker.
    guidance: tuple[str, ...]
    citations: tuple[Citation, ...]
    composer: str


class DraftComposer(Protocol):
    """Anything that turns a :class:`DraftRequest` into a :class:`Draft`."""

    name: str

    def compose(self, request: DraftRequest) -> Draft: ...


def clean_subject(subject: str) -> str:
    """Drop reply/forward/urgent prefixes: ``"Re: URGENT: X"`` -> ``"X"``."""
    return _SUBJECT_PREFIX.sub("", subject).strip() or subject.strip()


def number_citations(evidence: Sequence[Evidence]) -> tuple[Citation, ...]:
    """Customer-facing sources first (they appear in the reply), then internal ones."""
    ordered = [e for e in evidence if e.passage.audience == CUSTOMER]
    ordered += [e for e in evidence if e.passage.audience == INTERNAL]
    return tuple(
        Citation(
            ref=ref,
            doc_id=e.passage.doc_id,
            title=e.passage.title,
            section=e.passage.section,
            audience=e.passage.audience,
            score=e.score,
            excerpt=" ".join(e.key_sentences),
        )
        for ref, e in enumerate(ordered, start=1)
    )


def sources_footer(citations: Sequence[Citation]) -> list[str]:
    return ["Sources:", *(f"[{c.ref}] {c.title} - {c.section}" for c in citations)]


def _opening(subject: str, risk_band: str, codes: set[str]) -> str:
    if codes & {"repeat_contact", "strong_frustration"}:
        return (
            f'Thank you for following up about "{subject}". I am sorry this is still not '
            "resolved and that you have had to contact us again."
        )
    if risk_band == "high" or codes & {"business_impact", "security"}:
        return (
            f'Thank you for contacting us about "{subject}". I am sorry for the disruption '
            "this is causing."
        )
    return f'Thanks for reaching out about "{subject}".'


def _closing(route: str, customer_tier: str) -> str:
    if route == HUMAN_REVIEW:
        return (
            "I have passed your ticket to a senior support specialist, who will take "
            "ownership of it and confirm the next steps with you within one hour."
        )
    return (
        "If this does not fully solve the problem, reply to this message and we will follow "
        f"up within {sla_hours_for(customer_tier)} hours."
    )


class TemplateComposer:
    """Deterministic, offline composer (the default)."""

    name = "template"

    def compose(self, request: DraftRequest) -> Draft:
        citations = number_citations(request.evidence)
        sources = [c for c in citations if c.audience == CUSTOMER]
        guidance = tuple(f"{c.excerpt} [{c.ref}]" for c in citations if c.audience == INTERNAL)
        codes = {signal.code for signal in request.signals}

        lines = [
            "Hi there,",
            "",
            _opening(clean_subject(request.subject), request.risk_band, codes),
        ]
        lines.append("")
        if sources:
            lines.append("Here is what applies to your request:")
            lines.extend(f"- {c.excerpt} [{c.ref}]" for c in sources)
        else:
            lines.append(
                "A member of our team is looking into this and will reply with the next steps."
            )
        lines += ["", _closing(request.route, request.customer_tier), "", "Kind regards,"]
        lines.append("Customer Support")
        if sources:
            lines += ["", *sources_footer(sources)]
        return Draft("\n".join(lines), guidance, citations, self.name)


#: ``(url, json_payload, headers, timeout_seconds) -> parsed JSON response``
Transport = Callable[[str, Mapping[str, Any], Mapping[str, str], float], Mapping[str, Any]]


def post_json(
    url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout: float
) -> Mapping[str, Any]:
    """Minimal JSON POST (stdlib only). The URL scheme is validated as http(s) by Settings."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class LLMComposer:
    """Optional composer backed by an OpenAI-compatible chat-completions endpoint."""

    name = "llm"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: SecretStr,
        timeout: float = 15.0,
        transport: Transport = post_json,
        fallback: DraftComposer | None = None,
    ) -> None:
        if not api_key.get_secret_value():
            raise ValueError("LLMComposer needs an API key (OPSPILOT_LLM_API_KEY)")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport
        self._fallback = fallback or TemplateComposer()

    def compose(self, request: DraftRequest) -> Draft:
        draft = self._fallback.compose(request)
        sources = [c for c in draft.citations if c.audience == CUSTOMER]
        if not sources:
            return draft
        try:
            response = self._transport(
                self._url,
                self._payload(request, sources),
                {"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
                self._timeout,
            )
            reply = str(response["choices"][0]["message"]["content"]).strip()
        except Exception as exc:  # any transport or format problem -> keep the template draft
            log.warning("LLM composer failed (%s); using the template draft", type(exc).__name__)
            return draft

        cited = {int(ref) for ref in _CITATION_MARK.findall(reply)}
        if not reply or not cited or not cited <= {c.ref for c in sources}:
            log.warning("LLM reply cited no or unknown sources; using the template draft")
            return draft
        footer = sources_footer([c for c in sources if c.ref in cited])
        return Draft("\n".join([reply, "", *footer]), draft.guidance, draft.citations, self.name)

    def _payload(self, request: DraftRequest, sources: Sequence[Citation]) -> dict[str, Any]:
        passages = {(e.passage.doc_id, e.passage.section): e.passage.text for e in request.evidence}
        context = "\n".join(
            f"[{c.ref}] {c.title} - {c.section}: {passages[(c.doc_id, c.section)]}" for c in sources
        )
        prompt = (
            f"Customer tier: {request.customer_tier}\nRoute: {request.route}\n"
            f"Subject: {request.subject}\nMessage: {request.body}\n\nSources:\n{context}"
        )
        return {
            "model": self._model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }


def build_composer(settings: Settings) -> DraftComposer:
    """The composer selected by ``OPSPILOT_COMPOSER`` (validated by :class:`Settings`)."""
    if settings.composer == "llm":
        if not (settings.llm_api_key and settings.llm_base_url and settings.llm_model):
            # Settings validation already enforces this; guard against hand-built settings.
            raise ValueError(
                "OPSPILOT_COMPOSER=llm requires OPSPILOT_LLM_API_KEY, OPSPILOT_LLM_BASE_URL "
                "and OPSPILOT_LLM_MODEL"
            )
        return LLMComposer(
            base_url=str(settings.llm_base_url),
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_seconds,
        )
    return TemplateComposer()
