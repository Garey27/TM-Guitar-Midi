from pathlib import Path
import struct

import numpy as np

from tmgm_rt.nnpg import _HEADER, event_targets, guitar_teacher_slice, read_nnpg


def test_nnpg_v1_reader(tmp_path: Path):
    frame_count = 3
    note_bins, onset_bins, contour_bins = 88, 88, 264
    frame_bytes = note_bins + onset_bins + contour_bins
    commit = b"abc123" + bytes(35)
    header = _HEADER.pack(
        b"NNPG",
        1,
        256,
        frame_bytes,
        frame_count,
        note_bins,
        onset_bins,
        contour_bins,
        256,
        22050,
        255,
        0.7,
        0.5,
        125.0,
        48000.0,
        1,
        12345,
        54321,
        99,
        commit,
        bytes(119),
    )
    assert len(header) == 256
    payload = np.arange(frame_count * frame_bytes, dtype=np.uint8)
    path = tmp_path / "sample.nnpg"
    path.write_bytes(header + payload.tobytes())

    value = read_nnpg(path)
    assert value.header.frame_count == frame_count
    assert value.notes.shape == (frame_count, 88)
    assert value.onsets.shape == (frame_count, 88)
    assert value.contours.shape == (frame_count, 264)
    guitar_notes, guitar_onsets = guitar_teacher_slice(value, 40, 88)
    assert guitar_notes.shape == (frame_count, 49)
    assert guitar_onsets.shape == (frame_count, 49)


def test_event_targets_can_delay_causal_onset_without_delaying_activity(
    tmp_path: Path,
):
    events = tmp_path / "events.tsv"
    events.write_text(
        "start_frame\tend_frame\tpitch\tamplitude\n"
        "3\t9\t60\t0.75\n",
        encoding="utf-8",
    )

    targets = event_targets(
        events,
        frame_count=12,
        midi_min=60,
        midi_max=60,
        onset_width_frames=2,
        onset_delay_frames=2,
    )

    np.testing.assert_array_equal(
        targets.activity[:, 0],
        np.asarray([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]),
    )
    np.testing.assert_array_equal(
        targets.onset[:, 0],
        np.asarray([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]),
    )
