"""Runtime settings, read from ``OPSPILOT_*`` environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_KB_DIR = PROJECT_ROOT / "kb"
DEFAULT_QUEUE_DB = PROJECT_ROOT / "data" / "review_queue.db"
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://localhost:8080",
)
QUEUE_BACKENDS = ("sqlite", "memory")


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    kb_dir: Path = DEFAULT_KB_DIR
    #: Route to human review when risk >= this. ``None`` uses the threshold tuned on
    #: the validation set during training (stored with the model).
    risk_threshold: float | None = None
    #: Lower edge of the "medium" band. ``None`` means 60% of the review threshold.
    medium_threshold: float | None = None
    queue_backend: str = "sqlite"
    queue_db_path: Path = DEFAULT_QUEUE_DB
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    retrieval_top_k: int = 3
    composer: str = "template"
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if self.queue_backend not in QUEUE_BACKENDS:
            raise ValueError(f"queue_backend must be one of {QUEUE_BACKENDS}")
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be >= 1")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ

        def optional_float(name: str) -> float | None:
            raw = env.get(name, "").strip()
            return float(raw) if raw else None

        origins = env.get("OPSPILOT_CORS_ORIGINS")
        return cls(
            artifacts_dir=Path(env.get("OPSPILOT_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR))),
            kb_dir=Path(env.get("OPSPILOT_KB_DIR", str(DEFAULT_KB_DIR))),
            risk_threshold=optional_float("OPSPILOT_RISK_THRESHOLD"),
            medium_threshold=optional_float("OPSPILOT_MEDIUM_THRESHOLD"),
            queue_backend=env.get("OPSPILOT_QUEUE_BACKEND", "sqlite").strip().lower(),
            queue_db_path=Path(env.get("OPSPILOT_QUEUE_DB", str(DEFAULT_QUEUE_DB))),
            cors_origins=(
                tuple(o.strip() for o in origins.split(",") if o.strip())
                if origins is not None
                else DEFAULT_CORS_ORIGINS
            ),
            retrieval_top_k=int(env.get("OPSPILOT_TOP_K", "3")),
            composer=env.get("OPSPILOT_COMPOSER", "template").strip().lower(),
            torch_threads=int(env.get("OPSPILOT_TORCH_THREADS", "1")),
        )
