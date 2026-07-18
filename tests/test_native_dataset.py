from pathlib import Path

import numpy as np
import pytest

from tmgm_rt.native_dataset import (
    HEADER_BYTES,
    onset_training_indices,
    pack_binary_rows,
    read_native_dataset,
    unpack_binary_rows,
    write_native_dataset,
    write_native_dataset_batches,
)


def test_pack_binary_rows_round_trip_across_word_boundary():
    values = np.zeros((3, 70), dtype=np.uint32)
    values[0, [0, 1, 63, 64, 69]] = 1
    values[1, 2::3] = 1
    words = pack_binary_rows(values)
    assert words.dtype == np.dtype("<u8")
    assert words.shape == (3, 2)
    np.testing.assert_array_equal(unpack_binary_rows(words, 70), values)


def test_native_dataset_header_and_payload_round_trip(tmp_path: Path):
    rng = np.random.default_rng(17)
    features = rng.integers(0, 2, size=(11, 131), dtype=np.uint32)
    activity = rng.integers(0, 2, size=(11, 49), dtype=np.uint32)
    onset = rng.integers(0, 2, size=(11, 49), dtype=np.uint32)
    indices = np.asarray([3, 3, 8, 0, 10], dtype=np.uint32)
    path = tmp_path / "one-track.tmgd"

    written = write_native_dataset(
        path,
        features,
        activity,
        onset,
        indices,
        midi_min=40,
        sample_rate=22_050,
        hop_size=256,
        seed=20260718,
    )
    loaded = read_native_dataset(path)

    assert written == loaded.header
    assert loaded.header.features_offset == HEADER_BYTES
    assert loaded.header.midi_max == 88
    np.testing.assert_array_equal(
        unpack_binary_rows(loaded.feature_words, 131), features
    )
    np.testing.assert_array_equal(
        unpack_binary_rows(loaded.activity_words, 49), activity
    )
    np.testing.assert_array_equal(unpack_binary_rows(loaded.onset_words, 49), onset)
    np.testing.assert_array_equal(loaded.onset_indices, indices)


def test_native_dataset_checksum_rejects_damage(tmp_path: Path):
    path = tmp_path / "damage.tmgd"
    write_native_dataset(
        path,
        np.asarray([[0, 1, 0]], dtype=np.uint32),
        np.asarray([[1]], dtype=np.uint32),
        np.asarray([[0]], dtype=np.uint32),
        np.asarray([0], dtype=np.uint32),
        midi_min=40,
        sample_rate=22_050,
        hop_size=256,
        seed=1,
    )
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        read_native_dataset(path)


def test_onset_training_indices_are_deterministic_and_in_range():
    targets = np.zeros((12, 6), dtype=np.uint32)
    targets[1:8, 0] = 1
    targets[2:4, 1] = 1
    targets[[1, 7], 3] = 1
    targets[2, 4] = 1
    first = onset_training_indices(targets, note_count=3, rows=100, seed=91)
    second = onset_training_indices(targets, note_count=3, rows=100, seed=91)
    assert first.dtype == np.dtype("<u4")
    assert first.shape == (100,)
    assert int(first.max()) < targets.shape[0]
    np.testing.assert_array_equal(first, second)


def test_batched_native_writer_matches_monolithic_writer(tmp_path: Path):
    rng = np.random.default_rng(31)
    features = rng.integers(0, 2, size=(13, 137), dtype=np.uint32)
    activity = rng.integers(0, 2, size=(13, 49), dtype=np.uint32)
    onset = rng.integers(0, 2, size=(13, 49), dtype=np.uint32)
    indices = np.asarray([12, 0, 4, 4, 9], dtype=np.uint32)
    monolithic_path = tmp_path / "monolithic.tmgd"
    batched_path = tmp_path / "batched.tmgd"

    write_native_dataset(
        monolithic_path,
        features,
        activity,
        onset,
        indices,
        midi_min=40,
        sample_rate=22_050,
        hop_size=256,
        seed=123,
    )
    write_native_dataset_batches(
        batched_path,
        (
            (features[:3], activity[:3], onset[:3]),
            (features[3:11], activity[3:11], onset[3:11]),
            (features[11:], activity[11:], onset[11:]),
        ),
        indices,
        feature_count=features.shape[1],
        note_count=activity.shape[1],
        midi_min=40,
        sample_rate=22_050,
        hop_size=256,
        seed=123,
    )

    assert batched_path.read_bytes() == monolithic_path.read_bytes()
    loaded = read_native_dataset(batched_path)
    np.testing.assert_array_equal(
        unpack_binary_rows(loaded.feature_words, features.shape[1]), features
    )


def test_native_dataset_v2_feature_fingerprint_round_trip_and_corruption(
    tmp_path: Path,
):
    path = tmp_path / "fingerprinted.tmgd"
    fingerprint = bytes(range(32))
    write_native_dataset(
        path,
        np.asarray([[0, 1, 0]], dtype=np.uint32),
        np.asarray([[1]], dtype=np.uint32),
        np.asarray([[0]], dtype=np.uint32),
        np.asarray([0], dtype=np.uint32),
        midi_min=40,
        sample_rate=22_050,
        hop_size=256,
        seed=1,
        feature_fingerprint_sha256=fingerprint,
    )
    raw = bytearray(path.read_bytes())
    assert int.from_bytes(raw[8:12], "little") == 2
    assert raw[176:208] == fingerprint
    assert read_native_dataset(path).header.feature_fingerprint_sha256 == fingerprint

    raw[176:208] = bytes(32)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="v2 feature fingerprint is zero"):
        read_native_dataset(path)
