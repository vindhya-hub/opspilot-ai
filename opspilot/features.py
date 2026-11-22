"""scikit-learn feature pipeline: raw ticket dicts -> sparse feature matrix.

* text (subject + body) -> TF-IDF over word unigrams and bigrams
* customer tier, product area, channel -> one-hot (unseen categories are ignored)
* prior contacts, log hours open, hours-open / SLA ratio, SLA breach flag -> standardised
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from opspilot.domain import sla_hours_for

TEXT_COLUMN = "text"
CATEGORICAL_COLUMNS = ("customer_tier", "product_area", "channel")
NUMERIC_COLUMNS = ("prior_contacts_30d", "log_hours_open", "sla_ratio", "sla_breached")
FRAME_COLUMNS = (TEXT_COLUMN, *CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS)

MAX_PRIOR_CONTACTS = 20
MAX_SLA_RATIO = 10.0


def tickets_to_frame(tickets: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten tickets into model columns and derive the SLA features from tier + age."""
    rows = []
    for ticket in tickets:
        tier = str(ticket["customer_tier"])
        hours_open = max(float(ticket["hours_open"]), 0.0)
        sla_hours = sla_hours_for(tier)
        rows.append(
            {
                TEXT_COLUMN: f"{ticket['subject']}. {ticket['body']}",
                "customer_tier": tier,
                "product_area": str(ticket["product_area"]),
                "channel": str(ticket["channel"]),
                "prior_contacts_30d": float(
                    min(int(ticket["prior_contacts_30d"]), MAX_PRIOR_CONTACTS)
                ),
                "log_hours_open": math.log1p(hours_open),
                "sla_ratio": min(hours_open / sla_hours, MAX_SLA_RATIO),
                "sla_breached": float(hours_open > sla_hours),
            }
        )
    return pd.DataFrame(rows, columns=list(FRAME_COLUMNS))


def build_feature_pipeline(max_features: int = 3000, min_df: int = 2) -> Pipeline:
    """Unfitted pipeline mapping a list of ticket dicts to a sparse CSR feature matrix."""
    columns = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    # words only: drop invoice numbers, amounts and other IDs that a
                    # network would otherwise memorise
                    token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
                ),
                TEXT_COLUMN,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_COLUMNS)),
            ("numeric", StandardScaler(), list(NUMERIC_COLUMNS)),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline(steps=[("frame", FunctionTransformer(tickets_to_frame)), ("columns", columns)])
