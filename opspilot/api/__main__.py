"""Serve the API: ``python -m opspilot.api``.

Host, port and log level come from ``OPSPILOT_HOST``, ``OPSPILOT_PORT`` and
``OPSPILOT_LOG_LEVEL`` (see ``.env.example``). Invalid configuration, such as enabling the
LLM composer without ``OPSPILOT_LLM_API_KEY``, stops startup with a readable message.
"""

from __future__ import annotations

import logging

import uvicorn
from pydantic import ValidationError

from opspilot.config import Settings


def main() -> None:
    try:
        settings = Settings()
    except ValidationError as exc:
        problems = "; ".join(error["msg"] for error in exc.errors())
        raise SystemExit(f"invalid OpsPilot configuration: {problems}") from None
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "opspilot.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
