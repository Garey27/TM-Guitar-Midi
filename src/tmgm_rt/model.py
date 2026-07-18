from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import logging
import pickle

import numpy as np

# TMU imports its optional CUDA module even for an explicit CPU backend. Avoid
# printing a misleading traceback when PyCUDA is intentionally absent.
logging.getLogger("tmu.util.cuda_profiler").setLevel(logging.CRITICAL)
logging.getLogger("tmu.clause_bank.clause_bank_cuda").setLevel(logging.CRITICAL)
from tmu.experimental.models.multioutput_classifier import (
    TMCoalesceMultiOuputClassifier,
)

from .binarize import QuantileThermometer
from .config import ContextConfig, FrontendConfig, TargetConfig
from .metrics import tolerant_event_metrics


@dataclass(frozen=True)
class TMConfig:
    clauses: int = 256
    threshold: int = 128
    specificity: float = 5.0
    negative_samples: float = 8.0
    weighted_clauses: bool = True
    max_included_literals: int = 32
    clause_drop: float = 0.1
    literal_drop: float = 0.05
    seed: int = 42
    platform: str = "CPU"

    def __post_init__(self) -> None:
        if self.platform not in {"CPU", "CUDA"}:
            raise ValueError("TM platform must be CPU or CUDA")


@dataclass
class ModelBundle:
    frontend: FrontendConfig
    context: ContextConfig
    targets: TargetConfig
    tm_config: TMConfig
    binarizer: QuantileThermometer
    model: Any
    output_thresholds: np.ndarray
    metadata: dict[str, Any]

    def predict_scores(self, continuous_features: np.ndarray) -> np.ndarray:
        binary = self.binarizer.transform(continuous_features)
        _, scores = self.model.predict(binary, return_class_sums=True)
        return np.asarray(scores, dtype=np.float32)

    def predict(self, continuous_features: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(continuous_features)
        return (scores >= self.output_thresholds[None, :]).astype(np.uint32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        with Path(path).open("rb") as stream:
            value = pickle.load(stream)
        if not isinstance(value, ModelBundle):
            raise TypeError("artifact does not contain a ModelBundle")
        return value


@dataclass
class HeadEnsembleMember:
    """One TM specialist plus validation-derived score normalization."""

    model: Any
    tm_config: TMConfig
    score_center: float
    score_scale: float
    validation_f1: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.score_center):
            raise ValueError("member score center must be finite")
        if not np.isfinite(self.score_scale) or self.score_scale <= 0.0:
            raise ValueError("member score scale must be finite and positive")

    def predict_scores(self, binary_features: np.ndarray) -> np.ndarray:
        _, raw = self.model.predict(binary_features, return_class_sums=True)
        margin = (np.asarray(raw, dtype=np.float32) - self.score_center) / max(
            self.score_scale, 1.0e-6
        )
        return np.clip(margin, -4.0, 4.0)


@dataclass
class TMHeadEnsemble:
    """Fuse normalized raw class sums before any binary/MIDI decision."""

    name: str
    members: list[HeadEnsembleMember]
    weights: np.ndarray
    reducer: str = "mean"

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("an ensemble needs at least one member")
        self.weights = np.asarray(self.weights, dtype=np.float32)
        if self.weights.shape != (len(self.members),):
            raise ValueError("ensemble weight count does not match members")
        if np.any(self.weights < 0.0) or float(self.weights.sum()) <= 0.0:
            raise ValueError("ensemble weights must be non-negative and non-zero")
        self.weights = self.weights / self.weights.sum()
        if self.reducer not in {"mean", "max", "top2_mean"}:
            raise ValueError(f"unsupported ensemble reducer: {self.reducer}")

    def predict_scores(self, binary_features: np.ndarray) -> np.ndarray:
        member_scores = self.predict_member_scores(binary_features)
        return self.reduce_member_scores(member_scores)

    def predict_member_scores(self, binary_features: np.ndarray) -> np.ndarray:
        return np.stack(
            [member.predict_scores(binary_features) for member in self.members],
            axis=0,
        )

    def reduce_member_scores(self, member_scores: np.ndarray) -> np.ndarray:
        # Old schema-2 artifacts were pickled before ``reducer`` existed.
        strategy = getattr(self, "reducer", "mean")
        if strategy == "mean":
            return np.tensordot(self.weights, member_scores, axes=(0, 0))
        if strategy == "max":
            return member_scores.max(axis=0)
        if strategy == "top2_mean":
            count = min(2, member_scores.shape[0])
            return np.partition(member_scores, -count, axis=0)[-count:].mean(axis=0)
        raise ValueError(f"unsupported ensemble reducer: {strategy}")


@dataclass
class EnsembleBundle:
    frontend: FrontendConfig
    context: ContextConfig
    targets: TargetConfig
    binarizer: QuantileThermometer
    activity: TMHeadEnsemble
    onset: TMHeadEnsemble | None
    output_thresholds: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.output_thresholds = np.asarray(
            self.output_thresholds, dtype=np.float32
        )
        expected = self.frontend.note_count * (2 if self.onset is not None else 1)
        if self.output_thresholds.shape != (expected,):
            raise ValueError("ensemble threshold count does not match output heads")

    def predict_scores(self, continuous_features: np.ndarray) -> np.ndarray:
        binary = self.binarizer.transform(continuous_features)
        heads = [self.activity.predict_scores(binary)]
        if self.onset is not None:
            heads.append(self.onset.predict_scores(binary))
        return np.concatenate(heads, axis=1)

    def predict(self, continuous_features: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(continuous_features)
        return (scores >= self.output_thresholds[None, :]).astype(np.uint32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(path: str | Path) -> ModelBundle | EnsembleBundle:
    """Load either the original single-TM artifact or an ensemble artifact."""
    with Path(path).open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, (ModelBundle, EnsembleBundle)):
        raise TypeError("artifact does not contain a supported model bundle")
    return value


def create_model(config: TMConfig) -> TMCoalesceMultiOuputClassifier:
    return TMCoalesceMultiOuputClassifier(
        number_of_clauses=config.clauses,
        T=config.threshold,
        s=config.specificity,
        q=config.negative_samples,
        platform=config.platform,
        feature_negation=True,
        weighted_clauses=config.weighted_clauses,
        max_included_literals=config.max_included_literals,
        clause_drop_p=config.clause_drop,
        literal_drop_p=config.literal_drop,
        seed=config.seed,
    )


def _fbeta(truth: np.ndarray, prediction: np.ndarray, beta: float) -> float:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    tp = np.logical_and(truth, prediction).sum()
    fp = np.logical_and(np.logical_not(truth), prediction).sum()
    fn = np.logical_and(truth, np.logical_not(prediction)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    beta2 = beta * beta
    return float(
        (1.0 + beta2) * precision * recall
        / max(beta2 * precision + recall, 1.0e-12)
    )


def calibrate_head_threshold(
    scores: np.ndarray,
    truth: np.ndarray,
    maximum_ratio: float,
) -> float:
    """Choose one conservative global threshold for a complete output head."""
    true_mean = float(truth.sum(axis=1).mean())
    if true_mean <= 0.0:
        return float(scores.max() + 1.0)
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.995, 100)))
    candidates = np.unique(np.concatenate((candidates, np.asarray([0.0]))))
    rows: list[tuple[float, float, float]] = []
    for value in candidates:
        prediction = scores >= value
        f1 = _fbeta(truth, prediction, 1.0)
        predicted_mean = float(prediction.sum(axis=1).mean())
        rows.append((f1, float(value), predicted_mean))
    allowed = [
        row
        for row in rows
        if 0.5 * true_mean <= row[2] <= maximum_ratio * true_mean
    ]
    if allowed:
        _, selected, _ = max(allowed, key=lambda row: (row[0], row[1]))
    else:
        _, selected, _ = min(rows, key=lambda row: abs(row[2] - true_mean))
    return selected


def calibrate_event_threshold(
    scores: np.ndarray,
    truth: np.ndarray,
    maximum_ratio: float,
    radius: int = 4,
    minimum_ratio: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Calibrate sparse event decisions for pitch-aware timing tolerance."""
    true_mean = float(truth.sum(axis=1).mean())
    if true_mean <= 0.0:
        threshold = float(scores.max() + 1.0)
        return threshold, tolerant_event_metrics(
            truth, scores >= threshold, radius
        )
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.999, 160)))
    candidates = np.unique(np.concatenate((candidates, np.asarray([0.0]))))
    rows: list[tuple[float, float, float, dict[str, float]]] = []
    for value in candidates:
        prediction = scores >= value
        metrics = tolerant_event_metrics(truth, prediction, radius)
        predicted_mean = float(prediction.sum(axis=1).mean())
        rows.append((metrics["f1"], float(value), predicted_mean, metrics))
    allowed = [
        row
        for row in rows
        if minimum_ratio * true_mean <= row[2] <= maximum_ratio * true_mean
    ]
    selected = max(allowed or rows, key=lambda row: (row[0], row[1]))
    return selected[1], selected[3]


def make_head_member(
    model: Any,
    tm_config: TMConfig,
    validation_scores: np.ndarray,
    validation_truth: np.ndarray,
    maximum_ratio: float,
) -> HeadEnsembleMember:
    center = calibrate_head_threshold(
        validation_scores, validation_truth, maximum_ratio
    )
    # A scalar robust scale keeps members comparable without independently
    # warping pitches or learning dozens of fragile validation parameters.
    scale = float(np.quantile(np.abs(validation_scores - center), 0.95))
    prediction = validation_scores >= center
    validation_f1 = _fbeta(validation_truth, prediction, 1.0)
    return HeadEnsembleMember(
        model=model,
        tm_config=tm_config,
        score_center=center,
        score_scale=max(scale, 1.0),
        validation_f1=validation_f1,
    )


def calibrate_thresholds(
    scores: np.ndarray, truth: np.ndarray, note_count: int
) -> np.ndarray:
    thresholds = np.zeros(scores.shape[1], dtype=np.float32)
    heads = [(0, note_count, 1.25)]
    if scores.shape[1] >= 2 * note_count:
        heads.append((note_count, 2 * note_count, 1.5))
    for begin, end, maximum_ratio in heads:
        head_scores = scores[:, begin:end]
        head_truth = truth[:, begin:end]
        thresholds[begin:end] = calibrate_head_threshold(
            head_scores, head_truth, maximum_ratio
        )
    return thresholds


def bundle_metadata(
    frontend: FrontendConfig,
    context: ContextConfig,
    targets: TargetConfig,
    tm_config: TMConfig,
) -> dict[str, Any]:
    return {
        "frontend": asdict(frontend),
        "context": asdict(context),
        "targets": asdict(targets),
        "tm": asdict(tm_config),
        "backend": "TMU.experimental.TMCoalesceMultiOuputClassifier",
        "causal": True,
    }
