"""PyTorch MLP escalation classifier and the inference wrapper that ships with it."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from scipy import sparse
from sklearn.pipeline import Pipeline
from torch import nn

PIPELINE_FILE = "feature_pipeline.joblib"
WEIGHTS_FILE = "classifier.pt"
METADATA_FILE = "model_metadata.json"
ARTIFACT_FILES = (PIPELINE_FILE, WEIGHTS_FILE, METADATA_FILE)


class EscalationMLP(nn.Module):
    """Feed-forward network over the sparse ticket features; outputs one logit per row."""

    def __init__(
        self, input_dim: int, hidden_dims: Sequence[int] = (64, 16), dropout: float = 0.3
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


@dataclass(frozen=True)
class ModelMetadata:
    version: str
    decision_threshold: float
    input_dim: int
    hidden_dims: tuple[int, ...]
    dropout: float
    trained_at: str
    training_seed: int


def to_tensor(matrix: Any) -> torch.Tensor:
    """Densify a (sparse) feature matrix into a float32 tensor."""
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    return torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32))


def fingerprint(network: nn.Module) -> str:
    """Content hash of the weights, used as a reproducible model version."""
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return f"mlp-{digest.hexdigest()[:10]}"


class EscalationModel:
    """Fitted feature pipeline + MLP. ``predict_proba`` returns P(escalation) per ticket."""

    def __init__(self, features: Pipeline, network: EscalationMLP, metadata: ModelMetadata):
        self.features = features
        self.network = network.eval()
        self.metadata = metadata

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def decision_threshold(self) -> float:
        return self.metadata.decision_threshold

    def predict_proba(self, tickets: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = self.features.transform(list(tickets))
        with torch.inference_mode():
            logits = self.network(to_tensor(matrix))
        return torch.sigmoid(logits).numpy().astype(np.float64)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.features, directory / PIPELINE_FILE)
        torch.save(self.network.state_dict(), directory / WEIGHTS_FILE)
        (directory / METADATA_FILE).write_text(
            json.dumps(asdict(self.metadata), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> EscalationModel:
        """Load artifacts written by :meth:`save` (only load artifacts you produced)."""
        directory = Path(directory)
        missing = [name for name in ARTIFACT_FILES if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"model artifacts missing in {directory}: {', '.join(missing)} "
                "(run `python -m opspilot.train` first)"
            )
        raw = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        metadata = ModelMetadata(**{**raw, "hidden_dims": tuple(raw["hidden_dims"])})
        network = EscalationMLP(metadata.input_dim, metadata.hidden_dims, metadata.dropout)
        state = torch.load(directory / WEIGHTS_FILE, map_location="cpu", weights_only=True)
        network.load_state_dict(state)
        features = joblib.load(directory / PIPELINE_FILE)
        return cls(features, network, metadata)
