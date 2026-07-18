import numpy as np

from tmgm_rt.model import TMConfig, create_model


def test_tmu_multioutput_backend_smoke():
    rng = np.random.default_rng(7)
    x = rng.integers(0, 2, size=(96, 12), dtype=np.uint32)
    y = np.stack(
        (
            np.logical_xor(x[:, 0], x[:, 1]),
            np.logical_and(x[:, 2], x[:, 3]),
            np.logical_or(x[:, 4], x[:, 5]),
        ),
        axis=1,
    ).astype(np.uint32)
    model = create_model(
        TMConfig(
            clauses=32,
            threshold=16,
            specificity=4.0,
            negative_samples=2.0,
            max_included_literals=8,
            clause_drop=0.0,
            literal_drop=0.0,
            seed=9,
        )
    )
    for _ in range(2):
        model.fit(x, y)
    prediction, scores = model.predict(x, return_class_sums=True)
    assert prediction.shape == y.shape
    assert scores.shape == y.shape
