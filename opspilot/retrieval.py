"""Knowledge-base retrieval over the markdown runbooks in ``kb/``.

Every ``## Section`` of a runbook is one passage. Passages are indexed with TF-IDF over
word unigrams/bigrams plus character n-grams (so "charged" still matches "charges"
without a stemmer) and ranked by cosine similarity to the ticket text. The same index
scores individual sentences, which lets the draft composer quote the most relevant
sentence of each passage instead of a whole section.

A runbook may start with a small front-matter block; ``audience: internal`` marks
agent-only guidance (escalation policy, de-escalation) that is cited to the agent but never
quoted to the customer. Everything else defaults to ``audience: customer``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer

CUSTOMER = "customer"
INTERNAL = "internal"
AUDIENCES = (CUSTOMER, INTERNAL)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_LIST_MARKER = re.compile(r"^(?:[-*+]|\d+[.)])\s+")


@dataclass(frozen=True)
class Passage:
    """One ``## Section`` of a runbook."""

    doc_id: str
    title: str
    section: str
    text: str
    audience: str = CUSTOMER

    @property
    def sentences(self) -> tuple[str, ...]:
        return split_sentences(self.text)


@dataclass(frozen=True)
class Evidence:
    """A retrieved passage, its similarity to the query and its best-matching sentences."""

    passage: Passage
    score: float
    key_sentences: tuple[str, ...]


def split_sentences(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip())


def _split_front_matter(doc_id: str, markdown: str) -> tuple[dict[str, str], list[str]]:
    """``---``-delimited ``key: value`` lines at the top of a runbook, and the remaining lines."""
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, lines
    for end in range(1, len(lines)):
        if lines[end].strip() == "---":
            meta = {}
            for entry in lines[1:end]:
                key, _, value = entry.partition(":")
                meta[key.strip().lower()] = value.strip()
            return meta, lines[end + 1 :]
    raise ValueError(f"runbook {doc_id!r} has an unterminated front-matter block")


def parse_runbook(doc_id: str, markdown: str) -> list[Passage]:
    """Split a runbook into passages: one ``# Title``, then one passage per ``## Section``."""
    meta, body = _split_front_matter(doc_id, markdown)
    audience = meta.get("audience", CUSTOMER)
    if audience not in AUDIENCES:
        raise ValueError(f"runbook {doc_id!r}: audience must be one of {AUDIENCES}")

    title: str | None = None
    section: str | None = None
    lines: list[str] = []
    passages: list[Passage] = []

    def flush() -> None:
        if section is not None and lines:
            text = " ".join(lines)
            passages.append(Passage(doc_id, title or doc_id, section, text, audience))
        lines.clear()

    for raw in body:
        line = raw.strip()
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            section = line[3:].strip()
        elif line and section is not None:
            lines.append(_LIST_MARKER.sub("", line))
    flush()

    if title is None:
        raise ValueError(f"runbook {doc_id!r} has no '# Title' heading")
    if not passages:
        raise ValueError(f"runbook {doc_id!r} has no '## Section' with text")
    return passages


def load_knowledge_base(kb_dir: Path) -> list[Passage]:
    """Parse every ``*.md`` file in ``kb_dir`` (sorted, so the index is deterministic)."""
    paths = sorted(Path(kb_dir).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no markdown runbooks found in {kb_dir}")
    passages: list[Passage] = []
    for path in paths:
        passages.extend(parse_runbook(path.stem, path.read_text(encoding="utf-8")))
    return passages


def build_text_index() -> Pipeline:
    """Unfitted word + character TF-IDF whose output rows have unit L2 norm."""
    words = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english", sublinear_tf=True, strip_accents="unicode"
    )
    chars = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, strip_accents="unicode"
    )
    # Each branch is L2-normalised on its own; normalising the concatenation again makes
    # the dot product of two rows their cosine similarity.
    union = FeatureUnion([("words", words), ("chars", chars)])
    return Pipeline([("union", union), ("l2", Normalizer())])


class Retriever:
    """TF-IDF index over runbook passages and their sentences.

    A hit must score at least ``min_score`` (absolute cosine similarity) and at least
    ``min_relative_score`` times the best hit, which drops weak "also-ran" passages.
    """

    def __init__(
        self,
        passages: Sequence[Passage],
        *,
        min_score: float = 0.08,
        min_relative_score: float = 0.5,
    ) -> None:
        if not passages:
            raise ValueError("cannot build a retriever over an empty knowledge base")
        self.passages = tuple(passages)
        self.min_score = min_score
        self.min_relative_score = min_relative_score
        self._index = build_text_index()
        self._passage_vectors = self._index.fit_transform(
            [f"{p.title}. {p.section}. {p.text}" for p in self.passages]
        )
        # Sentence vectors are precomputed so a query is vectorised exactly once.
        self._sentence_rows: list[range] = []
        sentences: list[str] = []
        for passage in self.passages:
            start = len(sentences)
            sentences.extend(passage.sentences)
            self._sentence_rows.append(range(start, len(sentences)))
        self._sentences = tuple(sentences)
        self._sentence_vectors = self._index.transform(sentences)

    @classmethod
    def from_directory(cls, kb_dir: Path, **options: float) -> Retriever:
        return cls(load_knowledge_base(kb_dir), **options)

    @property
    def documents(self) -> dict[str, str]:
        """``doc_id -> title`` for every runbook in the index."""
        return {p.doc_id: p.title for p in self.passages}

    def _vectorise(self, query: str) -> sparse.spmatrix:
        return self._index.transform([query])

    def _ranked(
        self, query_vector: sparse.spmatrix, top_k: int, audience: str | None
    ) -> list[tuple[int, float]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        scores = (self._passage_vectors @ query_vector.T).toarray().ravel()
        candidates = [
            (int(i), float(scores[i]))
            for i in np.argsort(-scores, kind="stable")
            if audience is None or self.passages[i].audience == audience
        ]
        if not candidates:
            return []
        floor = max(self.min_score, self.min_relative_score * candidates[0][1])
        return [(i, score) for i, score in candidates[:top_k] if score >= floor]

    def _key_sentences(
        self, query_vector: sparse.spmatrix, passage_index: int, limit: int
    ) -> tuple[str, ...]:
        rows = self._sentence_rows[passage_index]
        if len(rows) <= limit:
            return self._sentences[rows.start : rows.stop]
        scores = (self._sentence_vectors[rows.start : rows.stop] @ query_vector.T).toarray().ravel()
        best = sorted(np.argsort(-scores, kind="stable")[:limit])  # keep reading order
        return tuple(self._sentences[rows.start + int(i)] for i in best)

    def search(
        self, query: str, top_k: int = 3, *, audience: str | None = None
    ) -> list[tuple[Passage, float]]:
        """Best passages first; ``audience`` restricts to customer-facing or internal runbooks."""
        ranked = self._ranked(self._vectorise(query), top_k, audience)
        return [(self.passages[i], score) for i, score in ranked]

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        *,
        lead_sentences: int = 2,
        sentences_per_passage: int = 1,
        audience: str | None = None,
    ) -> list[Evidence]:
        """:meth:`search` plus the most query-relevant sentences of each hit.

        The best hit contributes ``lead_sentences`` sentences, the others
        ``sentences_per_passage``.
        """
        query_vector = self._vectorise(query)
        evidence = []
        for rank, (i, score) in enumerate(self._ranked(query_vector, top_k, audience)):
            limit = lead_sentences if rank == 0 else sentences_per_passage
            sentences = self._key_sentences(query_vector, i, limit)
            evidence.append(Evidence(self.passages[i], round(score, 4), sentences))
        return evidence
