import numpy as np

from tmgm_rt.metrics import tolerant_event_metrics


def test_tolerant_event_metrics_preserve_pitch_and_allow_time_shift():
    truth = np.zeros((8, 2), dtype=np.uint32)
    prediction = np.zeros_like(truth)
    truth[3, 0] = 1
    prediction[5, 0] = 1

    assert tolerant_event_metrics(truth, prediction, radius=1)["f1"] == 0.0
    assert tolerant_event_metrics(truth, prediction, radius=2)["f1"] == 1.0

    prediction[:] = 0
    prediction[3, 1] = 1
    assert tolerant_event_metrics(truth, prediction, radius=2)["f1"] == 0.0
