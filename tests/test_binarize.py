import numpy as np

from tmgm_rt.binarize import QuantileThermometer


def test_quantile_binarizer_removes_constant_literals():
    values = np.asarray(
        [[0.0, 1.0, 5.0], [0.2, 1.0, 5.0], [0.8, 1.0, 5.0], [1.0, 1.0, 5.0]],
        dtype=np.float32,
    )
    encoder = QuantileThermometer((0.25, 0.5, 0.75))
    encoded = encoder.fit_transform(values)
    assert encoded.dtype == np.uint32
    assert encoded.shape[0] == values.shape[0]
    assert 0 < encoded.shape[1] < values.shape[1] * 3
    np.testing.assert_array_equal(encoded, encoder.transform(values))


def test_quantile_binarizer_batched_constant_scan_matches_full_scan():
    rng = np.random.default_rng(2026)
    values = rng.normal(size=(37, 11)).astype(np.float32)
    values[:, 3] = 1.0
    full = QuantileThermometer((0.25, 0.5, 0.75)).fit(values)
    batched = QuantileThermometer((0.25, 0.5, 0.75)).fit(
        values, constant_scan_batch_rows=5
    )

    np.testing.assert_array_equal(batched.thresholds, full.thresholds)
    np.testing.assert_array_equal(batched.keep_columns, full.keep_columns)
    np.testing.assert_array_equal(batched.transform(values), full.transform(values))
