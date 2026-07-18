from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuantileThermometer:
    quantiles: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95)
    thresholds: np.ndarray | None = None
    keep_columns: np.ndarray | None = None

    def fit(
        self, features: np.ndarray, *, constant_scan_batch_rows: int | None = None
    ) -> "QuantileThermometer":
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError("binarizer needs at least two feature rows")
        if constant_scan_batch_rows is not None and constant_scan_batch_rows <= 0:
            raise ValueError("constant scan batch size must be positive")
        self.thresholds = np.quantile(
            features, self.quantiles, axis=0
        ).T.astype(np.float32)
        if constant_scan_batch_rows is None:
            raw = self._raw_transform(features)
            self.keep_columns = np.logical_and(
                raw.any(axis=0), np.logical_not(raw.all(axis=0))
            )
        else:
            raw_column_count = features.shape[1] * len(self.quantiles)
            seen_true = np.zeros(raw_column_count, dtype=np.bool_)
            seen_false = np.zeros(raw_column_count, dtype=np.bool_)
            for first in range(0, features.shape[0], constant_scan_batch_rows):
                raw = self._raw_transform(
                    features[first : first + constant_scan_batch_rows]
                )
                seen_true |= raw.any(axis=0)
                seen_false |= np.logical_not(raw).any(axis=0)
            self.keep_columns = seen_true & seen_false
        if not self.keep_columns.any():
            raise ValueError("all thermometer literals are constant")
        return self

    def _raw_transform(self, features: np.ndarray) -> np.ndarray:
        if self.thresholds is None:
            raise RuntimeError("binarizer has not been fitted")
        values = np.asarray(features, dtype=np.float32)
        encoded = values[:, :, None] >= self.thresholds[None, :, :]
        return encoded.reshape(values.shape[0], -1)

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.keep_columns is None:
            raise RuntimeError("binarizer has not been fitted")
        return np.ascontiguousarray(
            self._raw_transform(features)[:, self.keep_columns], dtype=np.uint32
        )

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)
