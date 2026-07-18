import numpy as np

from tmgm_rt.midi import NoteStateConfig, stabilize_frame_predictions


def test_note_state_suppresses_spikes_and_bridges_short_release_gaps():
    raw = np.zeros((14, 2), dtype=np.uint32)
    raw[0, 0] = 1  # one-frame false attack
    raw[3:7, 0] = 1
    raw[9:12, 0] = 1  # two-frame gap must remain held
    stable = stabilize_frame_predictions(
        raw,
        note_count=1,
        config=NoteStateConfig(attack_frames=2, release_frames=3),
    )
    assert not stable[:4, 0].any()
    assert stable[4:14, 0].all()


def test_note_state_releases_after_confirmed_gap():
    raw = np.zeros((10, 2), dtype=np.uint32)
    raw[:4, 0] = 1
    stable = stabilize_frame_predictions(
        raw,
        note_count=1,
        config=NoteStateConfig(attack_frames=1, release_frames=3),
    )
    assert stable[:6, 0].all()
    assert not stable[6:, 0].any()


def test_note_state_retrigger_is_rising_edge_and_pitch_independent():
    raw = np.zeros((12, 4), dtype=np.uint32)
    raw[:, :2] = 1
    raw[3:6, 2] = 1
    raw[4, 3] = 1
    raw[8, 2] = 1
    stable = stabilize_frame_predictions(
        raw,
        note_count=2,
        config=NoteStateConfig(
            attack_frames=1,
            release_frames=2,
            retrigger_refractory_frames=3,
        ),
    )
    assert stable[:, :2].all()
    assert np.flatnonzero(stable[:, 2]).tolist() == [0, 3, 8]
    assert np.flatnonzero(stable[:, 3]).tolist() == [0, 4]
