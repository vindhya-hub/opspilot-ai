"""Human-review queue behind a small repository interface.

Two interchangeable backends implement :class:`ReviewQueue`: :class:`SqliteReviewQueue`
(the default, persists across restarts) and :class:`InMemoryReviewQueue` (tests, demos).
Pending items are listed highest risk first; resolved or all items newest first.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opspilot.config import Settings
from opspilot.domain import Resolution, ReviewStatus


class ReviewItemNotFoundError(LookupError):
    """No review item with that id."""


class ReviewItemAlreadyResolvedError(RuntimeError):
    """The review item was resolved before."""


@dataclass(frozen=True)
class NewReviewItem:
    ticket_id: str
    subject: str
    body: str
    customer_tier: str
    product_area: str
    channel: str
    risk_score: float
    risk_band: str
    signals: tuple[str, ...]
    draft: str
    guidance: tuple[str, ...]
    citations: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ReviewItem(NewReviewItem):
    id: str
    status: ReviewStatus
    created_at: str
    resolved_at: str | None = None
    resolution: Resolution | None = None
    reviewer: str | None = None
    note: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _create(item: NewReviewItem) -> ReviewItem:
    return ReviewItem(
        **asdict(item), id=f"rv_{uuid.uuid4().hex[:12]}", status="pending", created_at=_now()
    )


def _resolved(
    item: ReviewItem, resolution: Resolution, reviewer: str | None, note: str | None
) -> ReviewItem:
    if item.status == "resolved":
        raise ReviewItemAlreadyResolvedError(f"review item {item.id!r} is already resolved")
    return replace(
        item,
        status="resolved",
        resolved_at=_now(),
        resolution=resolution,
        reviewer=reviewer,
        note=note,
    )


class ReviewQueue(ABC):
    """Repository interface for review items."""

    @abstractmethod
    def add(self, item: NewReviewItem) -> ReviewItem: ...

    @abstractmethod
    def get(self, item_id: str) -> ReviewItem:
        """Raises :class:`ReviewItemNotFoundError` for unknown ids."""

    @abstractmethod
    def list_items(
        self, status: ReviewStatus | None = "pending", limit: int = 50
    ) -> list[ReviewItem]:
        """``status=None`` lists every item."""

    @abstractmethod
    def resolve(
        self,
        item_id: str,
        resolution: Resolution,
        *,
        reviewer: str | None = None,
        note: str | None = None,
    ) -> ReviewItem:
        """Raises :class:`ReviewItemNotFoundError` / :class:`ReviewItemAlreadyResolvedError`."""

    @abstractmethod
    def count(self, status: ReviewStatus | None = "pending") -> int: ...

    def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release resources (no-op unless the backend holds any)."""


class InMemoryReviewQueue(ReviewQueue):
    def __init__(self) -> None:
        self._items: dict[str, ReviewItem] = {}  # insertion-ordered
        self._lock = threading.Lock()

    def add(self, item: NewReviewItem) -> ReviewItem:
        created = _create(item)
        with self._lock:
            self._items[created.id] = created
        return created

    def get(self, item_id: str) -> ReviewItem:
        with self._lock:
            if item_id not in self._items:
                raise ReviewItemNotFoundError(f"review item {item_id!r} not found")
            return self._items[item_id]

    def list_items(
        self, status: ReviewStatus | None = "pending", limit: int = 50
    ) -> list[ReviewItem]:
        with self._lock:
            items = [i for i in self._items.values() if status is None or i.status == status]
        if status == "pending":
            items.sort(key=lambda i: -i.risk_score)  # stable: ties keep FIFO order
        else:
            items.reverse()
        return items[:limit]

    def resolve(
        self,
        item_id: str,
        resolution: Resolution,
        *,
        reviewer: str | None = None,
        note: str | None = None,
    ) -> ReviewItem:
        with self._lock:
            if item_id not in self._items:
                raise ReviewItemNotFoundError(f"review item {item_id!r} not found")
            updated = _resolved(self._items[item_id], resolution, reviewer, note)
            self._items[item_id] = updated
        return updated

    def count(self, status: ReviewStatus | None = "pending") -> int:
        with self._lock:
            return sum(1 for i in self._items.values() if status is None or i.status == status)


class SqliteReviewQueue(ReviewQueue):
    """SQLite-backed queue: indexed columns for filtering, the full item as JSON."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS review_items (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL CHECK (status IN ('pending', 'resolved')),
            risk_score REAL NOT NULL,
            created_at TEXT NOT NULL,
            item TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS review_items_by_status
            ON review_items (status, risk_score DESC);
    """

    def __init__(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        with self._lock, self._conn:
            self._conn.executescript(self._SCHEMA)

    @staticmethod
    def _load(raw: str) -> ReviewItem:
        data = json.loads(raw)
        for key in ("signals", "guidance", "citations"):
            data[key] = tuple(data[key])
        return ReviewItem(**data)

    def add(self, item: NewReviewItem) -> ReviewItem:
        created = _create(item)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO review_items (id, status, risk_score, created_at, item) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    created.id,
                    created.status,
                    created.risk_score,
                    created.created_at,
                    json.dumps(asdict(created)),
                ),
            )
        return created

    def get(self, item_id: str) -> ReviewItem:
        with self._lock:
            row = self._conn.execute(
                "SELECT item FROM review_items WHERE id = ?", (item_id,)
            ).fetchone()
        if row is None:
            raise ReviewItemNotFoundError(f"review item {item_id!r} not found")
        return self._load(row[0])

    def list_items(
        self, status: ReviewStatus | None = "pending", limit: int = 50
    ) -> list[ReviewItem]:
        if status == "pending":
            query = (
                "SELECT item FROM review_items WHERE status = 'pending' "
                "ORDER BY risk_score DESC, seq ASC LIMIT ?"
            )
            params: tuple[Any, ...] = (limit,)
        elif status is None:
            query, params = "SELECT item FROM review_items ORDER BY seq DESC LIMIT ?", (limit,)
        else:
            query = "SELECT item FROM review_items WHERE status = ? ORDER BY seq DESC LIMIT ?"
            params = (status, limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._load(raw) for (raw,) in rows]

    def resolve(
        self,
        item_id: str,
        resolution: Resolution,
        *,
        reviewer: str | None = None,
        note: str | None = None,
    ) -> ReviewItem:
        with self._lock, self._conn:  # one transaction: read, check, write
            row = self._conn.execute(
                "SELECT item FROM review_items WHERE id = ?", (item_id,)
            ).fetchone()
            if row is None:
                raise ReviewItemNotFoundError(f"review item {item_id!r} not found")
            updated = _resolved(self._load(row[0]), resolution, reviewer, note)
            self._conn.execute(
                "UPDATE review_items SET status = ?, item = ? WHERE id = ?",
                (updated.status, json.dumps(asdict(updated)), item_id),
            )
        return updated

    def count(self, status: ReviewStatus | None = "pending") -> int:
        with self._lock:
            if status is None:
                (total,) = self._conn.execute("SELECT COUNT(*) FROM review_items").fetchone()
            else:
                (total,) = self._conn.execute(
                    "SELECT COUNT(*) FROM review_items WHERE status = ?", (status,)
                ).fetchone()
        return int(total)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def create_review_queue(settings: Settings) -> ReviewQueue:
    if settings.queue_backend == "memory":
        return InMemoryReviewQueue()
    return SqliteReviewQueue(settings.queue_db_path)
