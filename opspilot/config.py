"""Runtime settings, read from ``OPSPILOT_*`` environment variables (and an optional ``.env``).

Every value has a safe, non-secret default (local paths, localhost, the template composer),
so the service runs with no configuration at all. The only secret the project knows about
is ``OPSPILOT_LLM_API_KEY`` for the optional LLM composer: it has no default, is held as a
``SecretStr`` (masked in logs and reprs) and is required only when that composer is enabled.
See ``.env.example`` for the full list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["critical", "error", "warning", "info", "debug"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPSPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `OPSPILOT_X=` (e.g. from compose substitution) -> default
        extra="ignore",  # the shared .env also holds compose/dashboard variables
        frozen=True,
        hide_input_in_errors=True,  # validation errors must never echo a secret value
    )

    # --- model, knowledge base and storage (paths are relative to the working directory)
    artifacts_dir: Path = Path("artifacts")
    kb_dir: Path = Path("kb")
    queue_backend: Literal["sqlite", "memory"] = "sqlite"
    queue_db_path: Path = Path("data/review_queue.db")

    # --- routing. Unset review threshold = the F1-optimal threshold tuned during training.
    risk_threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    #: Lower edge of the "medium" band; unset = 60% of the review threshold.
    medium_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    # --- guidance drafting
    retrieval_top_k: int = Field(default=3, ge=1, le=10)
    composer: Literal["template", "llm"] = "template"
    llm_base_url: HttpUrl | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)

    # --- HTTP server
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: LogLevel = "info"
    #: Comma-separated list of origins allowed to call the API from a browser.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_request_bytes: int = Field(default=1_048_576, ge=1024)
    torch_threads: int = Field(default=1, ge=1, le=64)

    @field_validator("log_level", "composer", "queue_backend", mode="before")
    @classmethod
    def _lowercase(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if (
            self.risk_threshold is not None
            and self.medium_threshold is not None
            and self.medium_threshold > self.risk_threshold
        ):
            raise ValueError("OPSPILOT_MEDIUM_THRESHOLD must not exceed OPSPILOT_RISK_THRESHOLD")
        if self.composer == "llm":
            required = {
                "OPSPILOT_LLM_API_KEY": self.llm_api_key and self.llm_api_key.get_secret_value(),
                "OPSPILOT_LLM_BASE_URL": self.llm_base_url,
                "OPSPILOT_LLM_MODEL": self.llm_model,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(
                    "OPSPILOT_COMPOSER=llm requires " + ", ".join(missing) + " to be set in the "
                    "environment (never commit the key); unset OPSPILOT_COMPOSER to use the "
                    "built-in template composer"
                )
        return self

    @property
    def cors_origin_list(self) -> tuple[str, ...]:
        return tuple(origin.strip() for origin in self.cors_origins.split(",") if origin.strip())
