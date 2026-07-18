from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tmgm_rt.config import ContextConfig, FrontendConfig, TargetConfig
from tmgm_rt.dataset import CorpusEntry
from tmgm_rt.native_dataset import read_native_dataset, unpack_binary_rows
from tmgm_rt.native_export import (
    cache_split_tracks,
    export_cached_split,
    fit_or_load_global_binarizer,
)
from tmgm_rt.stft_plus import CausalSTFTPlus


def _entry(root: Path, split: str, source: str, identifier: str) -> CorpusEntry:
    audio = root / f"{identifier}.wav"
    audio.write_bytes(b"synthetic audio identity")
    (root / f"{identifier}.nnpg").write_bytes(b"synthetic teacher")
    (root / f"{identifier}.events.tsv").write_text(
        "onset\toffset\tnote\n", encoding="utf-8"
    )
    return CorpusEntry(
        split=split,
        source=source,
        identifier=identifier,
        input_path=audio,
        output_relative=Path(identifier),
        group=f"group-{source}",
    )


def test_cached_full_export_uses_one_train_fitted_binarizer(tmp_path: Path):
    frontend = FrontendConfig(midi_min=40, midi_max=41)
    context = ContextConfig(delays=(0,))
    targets = TargetConfig()
    train_entries = [
        _entry(tmp_path, "train", "goat", "train-a"),
        _entry(tmp_path, "train", "guitarset", "train-b"),
    ]
    validation_entries = [
        _entry(tmp_path, "validation", "guitar-techs", "validation-a")
    ]
    feature_values = {
        "train-a": np.asarray(
            [[0.0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4]],
            dtype=np.float32,
        ),
        "train-b": np.asarray(
            [[0.8, 0.2], [1.0, 0.0], [0.7, 0.3], [0.5, 0.5]],
            dtype=np.float32,
        ),
        "validation-a": np.asarray(
            [[0.1, 0.9], [0.9, 0.1], [0.55, 0.45]], dtype=np.float32
        ),
    }
    feature_width = CausalSTFTPlus(frontend).feature_count
    feature_values = {
        name: np.resize(values, (values.shape[0], feature_width)).astype(
            np.float32
        )
        for name, values in feature_values.items()
    }
    build_calls: list[str] = []

    def builder(entry, *_args):
        build_calls.append(entry.identifier)
        features = feature_values[entry.identifier]
        truth = np.zeros((features.shape[0], 4), dtype=np.uint32)
        truth[:, 0] = 1
        truth[::2, 1] = 1
        truth[[0, -1], 2] = 1
        truth[0, 3] = 1
        return (
            features,
            truth,
            truth.astype(np.float32),
            truth[:, :2].sum(axis=1).astype(np.uint8),
            {"synthetic": features.shape[0]},
        )

    cache_root = tmp_path / "cache"
    train = cache_split_tracks(
        train_entries,
        tmp_path,
        cache_root,
        "train",
        4,
        17,
        frontend,
        context,
        targets,
        builder=builder,
    )
    validation = cache_split_tracks(
        validation_entries,
        tmp_path,
        cache_root,
        "validation",
        4,
        17,
        frontend,
        context,
        targets,
        builder=builder,
    )
    assert build_calls == ["train-a", "train-b", "validation-a"]

    # A resumed invocation uses the per-track cache instead of calling the
    # expensive audio/frontend builder again.
    cache_split_tracks(
        train_entries,
        tmp_path,
        cache_root,
        "train",
        4,
        17,
        frontend,
        context,
        targets,
        builder=builder,
    )
    assert build_calls == ["train-a", "train-b", "validation-a"]

    encoder_path = tmp_path / "output" / "global-quantile-thermometer.npz"
    binarizer, signature, digest = fit_or_load_global_binarizer(
        train,
        cache_root,
        encoder_path,
        quantiles=(0.5,),
        constant_scan_batch_rows=3,
    )
    expected_train = np.concatenate(
        [feature_values["train-a"], feature_values["train-b"]]
    )
    np.testing.assert_allclose(
        binarizer.thresholds[:, 0], np.quantile(expected_train, 0.5, axis=0)
    )

    train_path = tmp_path / "output" / "train.tmgd"
    validation_path = tmp_path / "output" / "validation.tmgd"
    export_cached_split(
        train_path,
        train,
        "train",
        binarizer,
        signature,
        digest,
        frontend,
        context,
        targets,
        seed=17,
        batch_rows=3,
        onset_row_multiplier=1.0,
    )
    export_cached_split(
        validation_path,
        validation,
        "validation",
        binarizer,
        signature,
        digest,
        frontend,
        context,
        targets,
        seed=17,
        batch_rows=2,
    )

    train_dataset = read_native_dataset(train_path)
    validation_dataset = read_native_dataset(validation_path)
    assert train_dataset.header.frame_count == 8
    assert validation_dataset.header.frame_count == 3
    assert train_dataset.header.feature_count == validation_dataset.header.feature_count
    np.testing.assert_array_equal(
        unpack_binary_rows(
            train_dataset.feature_words, train_dataset.header.feature_count
        ),
        binarizer.transform(expected_train),
    )
    np.testing.assert_array_equal(
        unpack_binary_rows(
            validation_dataset.feature_words,
            validation_dataset.header.feature_count,
        ),
        binarizer.transform(feature_values["validation-a"]),
    )
    sidecar = json.loads(
        train_path.with_suffix(".tmgd.json").read_text(encoding="utf-8")
    )
    assert sidecar["source_track_counts"] == {"goat": 1, "guitarset": 1}
    assert sidecar["binarizer"]["signature"] == signature
    assert sidecar["sampling"]["frame_sampling_policy"] == "balanced"
    assert sidecar["sampling"]["train_balanced_categories"] is True
    assert sidecar["sampling"]["train_natural_uniform"] is False

    # Completed files are cache hits too: a second export leaves the file alone.
    first_mtime = train_path.stat().st_mtime_ns
    export_cached_split(
        train_path,
        train,
        "train",
        binarizer,
        signature,
        digest,
        frontend,
        context,
        targets,
        seed=17,
        batch_rows=1,
        onset_row_multiplier=1.0,
    )
    assert train_path.stat().st_mtime_ns == first_mtime


def test_train_sampling_policy_changes_cache_and_reaches_builder(tmp_path: Path):
    frontend = FrontendConfig(midi_min=40, midi_max=41)
    context = ContextConfig(delays=(0,))
    targets = TargetConfig()
    train_entry = _entry(tmp_path, "train", "goat", "train-policy")
    validation_entry = _entry(
        tmp_path, "validation", "guitarset", "validation-policy"
    )
    build_calls: list[tuple[str, bool]] = []

    def builder(entry, _teacher_root, _frontend, _context, _targets, count, _seed,
                balanced_sampling):
        build_calls.append((entry.identifier, balanced_sampling))
        rows = min(count, 4)
        feature_width = CausalSTFTPlus(frontend).feature_count
        features = np.arange(
            rows * feature_width, dtype=np.float32
        ).reshape(rows, feature_width)
        truth = np.zeros((rows, 4), dtype=np.uint32)
        truth[:, 0] = 1
        truth[0, 2] = 1
        category = "balanced_synthetic" if balanced_sampling else "natural_uniform"
        return (
            features,
            truth,
            truth.astype(np.float32),
            truth[:, :2].sum(axis=1).astype(np.uint8),
            {category: rows},
        )

    cache_root = tmp_path / "cache"
    balanced = cache_split_tracks(
        [train_entry],
        tmp_path,
        cache_root,
        "train",
        4,
        23,
        frontend,
        context,
        targets,
        train_sampling="balanced",
        builder=builder,
    )
    natural = cache_split_tracks(
        [train_entry],
        tmp_path,
        cache_root,
        "train",
        4,
        23,
        frontend,
        context,
        targets,
        train_sampling="natural",
        builder=builder,
    )
    assert balanced[0].signature != natural[0].signature
    assert balanced[0].sampling_policy == "balanced"
    assert natural[0].sampling_policy == "natural"
    assert build_calls == [("train-policy", True), ("train-policy", False)]

    # Natural train is now independently resumable.
    cached_natural = cache_split_tracks(
        [train_entry],
        tmp_path,
        cache_root,
        "train",
        4,
        23,
        frontend,
        context,
        targets,
        train_sampling="natural",
        builder=builder,
    )
    assert cached_natural[0].signature == natural[0].signature
    assert build_calls == [("train-policy", True), ("train-policy", False)]

    # Validation ignores the train policy and is always natural/uniform. The
    # second call must hit exactly the same cache entry.
    validation_balanced_arg = cache_split_tracks(
        [validation_entry],
        tmp_path,
        cache_root,
        "validation",
        4,
        23,
        frontend,
        context,
        targets,
        train_sampling="balanced",
        builder=builder,
    )
    validation_natural_arg = cache_split_tracks(
        [validation_entry],
        tmp_path,
        cache_root,
        "validation",
        4,
        23,
        frontend,
        context,
        targets,
        train_sampling="natural",
        builder=builder,
    )
    assert validation_balanced_arg[0].signature == validation_natural_arg[0].signature
    assert validation_natural_arg[0].sampling_policy == "natural"
    assert build_calls[-1] == ("validation-policy", False)
    assert build_calls.count(("validation-policy", False)) == 1

    natural_metadata = json.loads(
        natural[0].path.with_suffix(".npz.json").read_text(encoding="utf-8")
    )
    assert natural_metadata["sampling_policy"] == "natural"
    assert natural_metadata["balanced_sampling"] is False

    binarizer, signature, digest = fit_or_load_global_binarizer(
        natural,
        cache_root,
        tmp_path / "natural-output" / "global-quantile-thermometer.npz",
        quantiles=(0.5,),
    )
    train_path = tmp_path / "natural-output" / "train.tmgd"
    export_cached_split(
        train_path,
        natural,
        "train",
        binarizer,
        signature,
        digest,
        frontend,
        context,
        targets,
        seed=23,
        onset_row_multiplier=1.0,
    )
    sidecar = json.loads(
        train_path.with_suffix(".tmgd.json").read_text(encoding="utf-8")
    )
    natural_dataset = read_native_dataset(train_path)
    assert sidecar["sampling"]["frame_sampling_policy"] == "natural"
    assert sidecar["sampling"]["train_balanced_categories"] is False
    assert sidecar["sampling"]["train_natural_uniform"] is True
    assert sidecar["sampling"]["onset_training_indices"] == {
        "name": "natural_order",
        "rows": natural_dataset.header.frame_count,
    }
    np.testing.assert_array_equal(
        natural_dataset.onset_indices,
        np.arange(natural_dataset.header.frame_count, dtype=np.uint32),
    )
