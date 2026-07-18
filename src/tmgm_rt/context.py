from __future__ import annotations

from collections import deque

import numpy as np

from .config import ContextConfig


def stack_causal_context(
    features: np.ndarray, config: ContextConfig
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("features must have shape [frames, features]")
    frames, width = features.shape
    output = np.zeros((frames, width * len(config.delays)), dtype=np.float32)
    for slot, delay in enumerate(config.delays):
        if delay == 0:
            output[:, slot * width : (slot + 1) * width] = features
        elif delay < frames:
            output[delay:, slot * width : (slot + 1) * width] = features[:-delay]
    return output


class StreamingContext:
    def __init__(self, feature_count: int, config: ContextConfig):
        self.feature_count = feature_count
        self.config = config
        self.history: deque[np.ndarray] = deque(maxlen=max(config.delays) + 1)

    def push(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size != self.feature_count:
            raise ValueError("streaming context feature width changed")
        self.history.appendleft(frame.copy())
        zero = np.zeros(self.feature_count, dtype=np.float32)
        return np.concatenate(
            [self.history[d] if d < len(self.history) else zero for d in self.config.delays]
        )
