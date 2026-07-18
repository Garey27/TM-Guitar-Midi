from pathlib import Path

import numpy as np

from tmgm_rt.binarize import QuantileThermometer
from tmgm_rt.config import ContextConfig, FrontendConfig, TargetConfig
from tmgm_rt.model import (
    EnsembleBundle,
    HeadEnsembleMember,
    ModelBundle,
    TMConfig,
    TMHeadEnsemble,
    load_bundle,
)


class FakeScoreModel:
    def __init__(self, scores: np.ndarray):
        self.scores = np.asarray(scores, dtype=np.float32)

    def predict(self, features, return_class_sums=False):
        scores = self.scores[: features.shape[0]]
        prediction = (scores >= 0.0).astype(np.uint32)
        return (prediction, scores) if return_class_sums else prediction


def _member(scores, center, scale):
    return HeadEnsembleMember(
        model=FakeScoreModel(np.asarray(scores, dtype=np.float32)),
        tm_config=TMConfig(),
        score_center=center,
        score_scale=scale,
        validation_f1=0.5,
    )


def test_fusion_uses_normalized_raw_margins():
    first = _member([[10.0, 20.0], [-50.0, 50.0]], 0.0, 10.0)
    second = _member([[100.0, 200.0], [-500.0, 500.0]], 0.0, 100.0)
    ensemble = TMHeadEnsemble(
        name="activity",
        members=[first, second],
        weights=np.asarray([1.0, 1.0]),
    )
    result = ensemble.predict_scores(np.zeros((2, 1), dtype=np.uint32))
    np.testing.assert_allclose(result[0], [1.0, 2.0])
    # Both members are clipped before fusion, preventing one outlier vote.
    np.testing.assert_allclose(result[1], [-4.0, 4.0])


def test_onset_reducers_keep_rare_positive_members():
    scores = np.asarray(
        [
            [[2.0, -2.0]],
            [[-1.0, 3.0]],
            [[-2.0, -1.0]],
        ],
        dtype=np.float32,
    )
    ensemble = TMHeadEnsemble(
        name="onset",
        members=[_member([[0.0, 0.0]], 0.0, 1.0) for _ in range(3)],
        weights=np.ones(3, dtype=np.float32),
        reducer="max",
    )
    np.testing.assert_array_equal(
        ensemble.reduce_member_scores(scores), [[2.0, 3.0]]
    )
    ensemble.reducer = "top2_mean"
    np.testing.assert_array_equal(
        ensemble.reduce_member_scores(scores), [[0.5, 1.0]]
    )


def test_generic_loader_accepts_legacy_model_bundle(tmp_path: Path):
    legacy = ModelBundle(
        frontend=FrontendConfig(),
        context=ContextConfig(),
        targets=TargetConfig(),
        tm_config=TMConfig(),
        binarizer=QuantileThermometer(),
        model=FakeScoreModel([[0.0] * 98]),
        output_thresholds=np.zeros(98, dtype=np.float32),
        metadata={"legacy": True},
    )
    path = tmp_path / "legacy.pkl"
    legacy.save(path)
    loaded = load_bundle(path)
    assert isinstance(loaded, ModelBundle)
    assert loaded.metadata == {"legacy": True}


def test_ensemble_bundle_save_load_preserves_fusion(tmp_path: Path):
    frontend = FrontendConfig(midi_min=40, midi_max=41)
    head = TMHeadEnsemble(
        name="activity",
        members=[_member([[10.0, -10.0]], 0.0, 10.0)],
        weights=np.ones(1, dtype=np.float32),
    )
    binarizer = QuantileThermometer(
        thresholds=np.asarray([[0.5]], dtype=np.float32),
        keep_columns=np.asarray([True]),
    )
    bundle = EnsembleBundle(
        frontend=frontend,
        context=ContextConfig(),
        targets=TargetConfig(activity_outputs=True, onset_outputs=False),
        binarizer=binarizer,
        activity=head,
        onset=None,
        output_thresholds=np.zeros(2, dtype=np.float32),
        metadata={"artifact_schema": 2},
    )
    path = tmp_path / "ensemble.pkl"
    bundle.save(path)
    loaded = load_bundle(path)
    before = bundle.predict_scores(np.asarray([[1.0]], dtype=np.float32))
    after = loaded.predict_scores(np.asarray([[1.0]], dtype=np.float32))
    np.testing.assert_array_equal(before, after)
