"""Train the escalation-risk model on synthetic tickets.

    python -m opspilot.train --output-dir artifacts

1. generate seeded synthetic tickets and make a stratified 70/15/15 train/val/test split
2. fit the scikit-learn feature pipeline on the training split only
3. train the PyTorch MLP with early stopping on validation loss
4. pick the F1-optimal decision threshold on the validation split
5. report held-out test metrics, plus two references on the same split: a logistic-regression
   baseline on the same features and an "oracle" that thresholds the generator's hidden
   noise-free probability (the best any model could roughly hope for)
6. write the model artifacts and ``metrics.json``
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn

from opspilot.config import Settings
from opspilot.data.generator import DEFAULT_SEED, generate_tickets
from opspilot.features import build_feature_pipeline
from opspilot.model import (
    METRICS_FILE,
    EscalationMLP,
    EscalationModel,
    ModelMetadata,
    fingerprint,
    to_tensor,
)

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


@dataclass(frozen=True)
class TrainingConfig:
    n_samples: int = 20_000
    seed: int = DEFAULT_SEED
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    max_features: int = 3000
    hidden_dims: tuple[int, ...] = (64, 16)
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 40
    patience: int = 4
    run_baseline: bool = True


@dataclass
class TrainingResult:
    model: EscalationModel
    metrics: dict[str, Any]


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Decision threshold that maximises F1 (first maximum on a 0.01 grid)."""
    f1s = [f1_score(y_true, scores >= t, zero_division=0) for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmax(f1s))])


def classification_report(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": len(y_true),
    }


def _split(labels: np.ndarray, config: TrainingConfig) -> tuple[np.ndarray, ...]:
    indices = np.arange(len(labels))
    holdout = config.val_fraction + config.test_fraction
    train_idx, rest_idx = train_test_split(
        indices, test_size=holdout, stratify=labels, random_state=config.seed
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=config.test_fraction / holdout,
        stratify=labels[rest_idx],
        random_state=config.seed,
    )
    return train_idx, val_idx, test_idx


def _scores(network: nn.Module, matrix: sparse.csr_matrix, batch_size: int = 2048) -> np.ndarray:
    network.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, matrix.shape[0], batch_size):
            logits = network(to_tensor(matrix[start : start + batch_size]))
            chunks.append(torch.sigmoid(logits).numpy())
    return np.concatenate(chunks).astype(np.float64)


def _fit_mlp(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_val: sparse.csr_matrix,
    y_val: np.ndarray,
    config: TrainingConfig,
) -> tuple[EscalationMLP, list[dict[str, float]], int]:
    torch.manual_seed(config.seed)
    shuffle = torch.Generator().manual_seed(config.seed)
    network = EscalationMLP(x_train.shape[1], config.hidden_dims, config.dropout)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    x_val_t, y_val_t = to_tensor(x_val), torch.from_numpy(y_val.astype(np.float32))

    history: list[dict[str, float]] = []
    best_loss, best_epoch, stale = math.inf, 0, 0
    best_state = copy.deepcopy(network.state_dict())
    for epoch in range(1, config.max_epochs + 1):
        network.train()
        order = torch.randperm(x_train.shape[0], generator=shuffle)
        running = 0.0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(network(to_tensor(x_train[batch.numpy()])), y_train_t[batch])
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch)

        network.eval()
        with torch.inference_mode():
            val_loss = loss_fn(network(x_val_t), y_val_t).item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(running / len(order), 4),
                "val_loss": round(val_loss, 4),
            }
        )
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(network.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break

    network.load_state_dict(best_state)
    network.eval()
    return network, history, best_epoch


def _logistic_baseline(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_val: sparse.csr_matrix,
    y_val: np.ndarray,
    x_test: sparse.csr_matrix,
    y_test: np.ndarray,
) -> dict[str, Any]:
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(x_train, y_train)
    threshold = select_threshold(y_val, clf.predict_proba(x_val)[:, 1])
    test = classification_report(y_test, clf.predict_proba(x_test)[:, 1], threshold)
    return {"model": "logistic_regression (same features)", "test": test}


def run_training(config: TrainingConfig, output_dir: Path | None = None) -> TrainingResult:
    """Train, evaluate and (optionally) save a model. Used by the CLI and the test suite."""
    started = time.perf_counter()
    tickets = generate_tickets(config.n_samples, seed=config.seed, include_latent=True)
    # The generator's hidden, noise-free escalation probability is stripped from the
    # tickets before any feature work; it is only used as an evaluation reference.
    oracle_scores = np.array([t.pop("latent")["p_escalation"] for t in tickets])
    labels = np.array([t["escalated"] for t in tickets], dtype=np.int64)
    train_idx, val_idx, test_idx = _split(labels, config)

    def subset(idx: np.ndarray) -> list[dict[str, Any]]:
        return [tickets[i] for i in idx]

    pipeline = build_feature_pipeline(max_features=config.max_features)
    x_train = sparse.csr_matrix(pipeline.fit_transform(subset(train_idx)))
    x_val = sparse.csr_matrix(pipeline.transform(subset(val_idx)))
    x_test = sparse.csr_matrix(pipeline.transform(subset(test_idx)))
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

    network, history, best_epoch = _fit_mlp(x_train, y_train, x_val, y_val, config)
    p_val, p_test = _scores(network, x_val), _scores(network, x_test)
    threshold = select_threshold(y_val, p_val)

    metadata = ModelMetadata(
        version=fingerprint(network),
        decision_threshold=threshold,
        input_dim=int(x_train.shape[1]),
        hidden_dims=tuple(config.hidden_dims),
        dropout=config.dropout,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        training_seed=config.seed,
    )
    model = EscalationModel(pipeline, network, metadata)
    n_terms = len(pipeline.named_steps["columns"].named_transformers_["text"].vocabulary_)

    metrics: dict[str, Any] = {
        "data": {
            "synthetic": True,
            "generator_seed": config.seed,
            "n_samples": config.n_samples,
            "positive_rate": round(float(labels.mean()), 4),
            "splits": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        },
        "model": {
            "type": "pytorch_mlp",
            "version": metadata.version,
            "input_dim": metadata.input_dim,
            "tfidf_terms": n_terms,
            "hidden_dims": list(config.hidden_dims),
            "dropout": config.dropout,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
        },
        "decision_threshold": threshold,
        "validation": classification_report(y_val, p_val, threshold),
        "test": classification_report(y_test, p_test, threshold),
        "test_at_threshold_0_5": classification_report(y_test, p_test, 0.5),
        "baseline": (
            _logistic_baseline(x_train, y_train, x_val, y_val, x_test, y_test)
            if config.run_baseline
            else None
        ),
        "reference_oracle": {
            "description": (
                "generator's hidden noise-free escalation probability (never a model input); "
                "an approximate ceiling given the label noise"
            ),
            "test": classification_report(
                y_test,
                oracle_scores[test_idx],
                select_threshold(y_val, oracle_scores[val_idx]),
            ),
        },
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 2),
        "trained_at": metadata.trained_at,
        "versions": {"torch": torch.__version__, "scikit-learn": sklearn.__version__},
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        model.save(output_dir)
        (output_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2) + "\n", "utf-8")
    return TrainingResult(model=model, metrics=metrics)


def format_summary(metrics: dict[str, Any]) -> str:
    data, test = metrics["data"], metrics["test"]
    splits = data["splits"]
    lines = [
        "OpsPilot AI - training summary (SYNTHETIC data)",
        f"  samples          {data['n_samples']} (train {splits['train']} / "
        f"val {splits['validation']} / test {splits['test']}), "
        f"positive rate {data['positive_rate']:.3f}",
        f"  model            {metrics['model']['version']}, "
        f"{metrics['model']['input_dim']} input features, "
        f"epochs {metrics['model']['epochs_run']} (best {metrics['model']['best_epoch']})",
        f"  threshold        {metrics['decision_threshold']:.2f} (tuned on validation)",
        f"  test F1          {test['f1']:.4f}  precision {test['precision']:.4f}  "
        f"recall {test['recall']:.4f}",
        f"  test ROC-AUC     {test['roc_auc']:.4f}  PR-AUC {test['pr_auc']:.4f}  "
        f"accuracy {test['accuracy']:.4f}",
    ]
    if metrics.get("baseline"):
        base = metrics["baseline"]["test"]
        lines.append(
            f"  baseline (LR)    test F1 {base['f1']:.4f}  precision {base['precision']:.4f}  "
            f"recall {base['recall']:.4f}"
        )
    oracle = metrics["reference_oracle"]["test"]
    lines.append(
        f"  oracle (hidden)  test F1 {oracle['f1']:.4f}  precision {oracle['precision']:.4f}  "
        f"recall {oracle['recall']:.4f}"
    )
    lines.append(f"  training time    {metrics['training_seconds']:.1f}s")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the OpsPilot escalation-risk model on synthetic tickets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write artifacts (default: OPSPILOT_ARTIFACTS_DIR, else ./artifacts)",
    )
    parser.add_argument("--n-samples", type=int, default=TrainingConfig.n_samples)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-epochs", type=int, default=TrainingConfig.max_epochs)
    parser.add_argument("--max-features", type=int, default=TrainingConfig.max_features)
    parser.add_argument("--no-baseline", action="store_true", help="skip the LR baseline")
    parser.add_argument(
        "--min-test-f1",
        type=float,
        default=None,
        help="exit non-zero when held-out test F1 is below this value (CI quality gate)",
    )
    args = parser.parse_args(argv)

    config = TrainingConfig(
        n_samples=args.n_samples,
        seed=args.seed,
        max_epochs=args.max_epochs,
        max_features=args.max_features,
        run_baseline=not args.no_baseline,
    )
    output_dir = args.output_dir or Settings().artifacts_dir
    result = run_training(config, output_dir=output_dir)
    print(format_summary(result.metrics))
    print(f"  artifacts        {output_dir}")

    test_f1 = result.metrics["test"]["f1"]
    if args.min_test_f1 is not None and test_f1 < args.min_test_f1:
        print(
            f"QUALITY GATE FAILED: test F1 {test_f1:.4f} < required {args.min_test_f1:.4f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
