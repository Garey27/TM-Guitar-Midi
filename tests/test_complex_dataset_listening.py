from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mido
import pytest

from scripts import prepare_complex_dataset_listening as prepare
from scripts import run_complex_dataset_listening as run


def _save(path: Path, messages: list[mido.Message | mido.MetaMessage]) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack(messages)
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)
    midi.save(path)


def test_committed_selection_is_two_tracks_per_dataset() -> None:
    path = prepare.ROOT / "configs/complex-dataset-selected-tracks-20260718.json"
    tracks = prepare.load_selection(path)
    assert len(tracks) == 6
    assert sorted(track["source"] for track in tracks) == [
        "goat", "goat", "guitar-techs", "guitar-techs", "guitarset", "guitarset"
    ]
    gtech = [track for track in tracks if track["source"] == "guitar-techs"]
    assert all("..events.tsv" in track["teacher_events"] for track in gtech)
    assert all(track["dataset_reference"]["kind"] != "absolute-midi" for track in gtech)


def test_normalize_midi_preserves_wall_clock_and_balances_voices(tmp_path: Path) -> None:
    source = tmp_path / "tempo.mid"
    _save(
        source,
        [
            mido.MetaMessage("set_tempo", tempo=500_000, time=0),
            mido.Message("note_on", note=48, velocity=90, time=0),
            mido.Message("note_off", note=48, velocity=0, time=480),
            mido.MetaMessage("set_tempo", tempo=1_000_000, time=0),
            mido.Message("note_on", note=50, velocity=70, time=0),
            mido.Message("note_off", note=50, velocity=0, time=480),
        ],
    )
    output = tmp_path / "normalized.mid"
    prepare.normalize_midi_absolute_time(source, output)
    normalized = mido.MidiFile(output)
    assert normalized.ticks_per_beat == 480
    assert normalized.length == pytest.approx(1.5, abs=1 / 960)
    audit = run.audit_midi_integrity(output)
    assert audit["note_ons"] == audit["note_offs"] == 2
    assert audit["per_channel_pitch_balanced"] is True


def test_labels_reference_filters_plugin_range(tmp_path: Path) -> None:
    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "start\tend\tmidi\tstring\n"
        "0.0\t0.25\t40\t0\n"
        "0.0\t0.25\t89\t1\n",
        encoding="utf-8",
    )
    output = tmp_path / "reference.mid"
    prepare.write_labels_reference(labels, output)
    audit = run.audit_midi_integrity(output)
    assert audit["note_ons"] == audit["note_offs"] == 1
    assert audit["distinct_pitches"] == 1


def test_labels_reference_preserves_string_channels_for_unisons(tmp_path: Path) -> None:
    labels = tmp_path / "unison.tsv"
    labels.write_text(
        "start\tend\tmidi\tstring\n"
        "0.0\t0.50\t52\t0\n"
        "0.1\t0.40\t52\t1\n",
        encoding="utf-8",
    )
    output = tmp_path / "unison.mid"
    prepare.write_labels_reference(labels, output)
    audit = run.audit_midi_integrity(output)
    assert audit["note_ons"] == audit["note_offs"] == 2
    assert audit["maximum_voice_polyphony"] == 2
    channels = {
        message.channel
        for message in mido.MidiFile(output).tracks[0]
        if message.type == "note_on" and message.velocity > 0
    }
    assert channels == {0, 1}


def test_integrity_audit_rejects_orphan_note_off(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mid"
    _save(
        broken,
        [
            mido.MetaMessage("set_tempo", tempo=500_000, time=0),
            mido.Message("note_off", note=48, velocity=0, time=10),
        ],
    )
    with pytest.raises(ValueError, match="orphan NoteOff"):
        run.audit_midi_integrity(broken)


def test_cached_feature_reuse_checks_path_sha_and_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "source.wav"
    wav.write_bytes(b"wav")
    dataset = tmp_path / "track.tmgd"
    dataset.write_bytes(b"dataset")
    fingerprint = "a" * 64
    sidecar = dataset.with_suffix(".tmgd.json")
    sidecar.write_text(
        json.dumps(
            {
                "feature_semantics": {"fingerprint_sha256": fingerprint},
                "input": {
                    "path": str(wav.resolve()),
                    "sha256": prepare.sha256_file(wav),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        prepare,
        "inspect_dataset_contract",
        lambda _: SimpleNamespace(feature_fingerprint_sha256=fingerprint),
    )
    assert prepare.feature_is_reusable(dataset, wav, fingerprint)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["input"]["sha256"] = "0" * 64
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="source WAV SHA-256"):
        prepare.feature_is_reusable(dataset, wav, fingerprint)
