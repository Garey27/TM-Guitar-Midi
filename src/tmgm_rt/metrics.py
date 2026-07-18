from __future__ import annotations

import numpy as np


def _binary_counts(truth: np.ndarray, prediction: np.ndarray) -> tuple[int, int, int]:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    tp = int(np.logical_and(truth, prediction).sum())
    fp = int(np.logical_and(np.logical_not(truth), prediction).sum())
    fn = int(np.logical_and(truth, np.logical_not(prediction)).sum())
    return tp, fp, fn


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    tp, fp, fn = _binary_counts(truth, prediction)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def _dilate_frames(values: np.ndarray, radius: int) -> np.ndarray:
    """Dilate binary events along time without mixing pitch columns."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    source = values.astype(bool)
    result = source.copy()
    for offset in range(1, radius + 1):
        result[offset:] |= source[:-offset]
        result[:-offset] |= source[offset:]
    return result


def tolerant_event_metrics(
    truth: np.ndarray, prediction: np.ndarray, radius: int
) -> dict[str, float]:
    """Pitch-aware event metrics allowing a symmetric frame-time tolerance.

    Precision asks whether each predicted pitch/frame is near a matching truth
    pitch. Recall performs the inverse query. This is intentionally diagnostic;
    it does not alter labels or decoder timing during training/inference.
    """
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    truth_nearby = _dilate_frames(truth, radius)
    prediction_nearby = _dilate_frames(prediction, radius)
    precision = float(np.logical_and(prediction, truth_nearby).sum()) / max(
        int(prediction.sum()), 1
    )
    recall = float(np.logical_and(truth, prediction_nearby).sum()) / max(
        int(truth.sum()), 1
    )
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def polyphonic_metrics(
    truth: np.ndarray, prediction: np.ndarray, note_count: int
) -> dict[str, float]:
    activity_truth = truth[:, :note_count]
    activity_prediction = prediction[:, :note_count]
    activity = binary_metrics(activity_truth, activity_prediction)
    result = {f"activity_{name}": value for name, value in activity.items()}
    if truth.shape[1] >= 2 * note_count:
        onset_truth = truth[:, note_count : 2 * note_count]
        onset_prediction = prediction[:, note_count : 2 * note_count]
        onset = binary_metrics(onset_truth, onset_prediction)
        result.update({f"onset_{name}": value for name, value in onset.items()})
        for radius in (2, 4, 8):
            tolerant = tolerant_event_metrics(
                onset_truth, onset_prediction, radius=radius
            )
            result.update(
                {
                    f"onset_tolerance_{radius}f_{name}": value
                    for name, value in tolerant.items()
                }
            )

    true_polyphony = activity_truth.sum(axis=1)
    chord_rows = true_polyphony >= 2
    if chord_rows.any():
        intersection = np.logical_and(
            activity_truth[chord_rows], activity_prediction[chord_rows]
        ).sum(axis=1)
        recall = intersection / true_polyphony[chord_rows]
        result["chord_frame_pitch_recall"] = float(recall.mean())
        result["chord_frame_complete_rate"] = float((recall == 1.0).mean())
        result["chord_frame_count"] = float(chord_rows.sum())
    else:
        result["chord_frame_pitch_recall"] = 0.0
        result["chord_frame_complete_rate"] = 0.0
        result["chord_frame_count"] = 0.0
    result["predicted_mean_polyphony"] = float(activity_prediction.sum(axis=1).mean())
    result["teacher_mean_polyphony"] = float(true_polyphony.mean())
    return result
