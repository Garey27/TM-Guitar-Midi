from pathlib import Path

import mido

from tmgm_rt.midi import write_teacher_events


def test_teacher_event_writer_has_one_note_and_nonnegative_deltas(tmp_path: Path):
    events = tmp_path / "events.tsv"
    events.write_text(
        "start_sec\tend_sec\tpitch\tamplitude\tstart_frame\tend_frame\tbends_thirds_of_semitone\n"
        "0.1\t0.3\t52\t0.5\t9\t26\t0\n"
        "0.3\t0.5\t52\t1.0\t26\t43\t0\n",
        encoding="utf-8",
    )
    output = tmp_path / "teacher.mid"
    write_teacher_events(output, events, 40, 88)
    midi = mido.MidiFile(output)
    messages = [message for track in midi.tracks for message in track]
    assert sum(m.type == "note_on" and m.velocity > 0 for m in messages) == 2
    assert all(message.time >= 0 for message in messages)
    assert [
        message.velocity
        for message in messages
        if message.type == "note_on" and message.velocity > 0
    ] == [64, 127]
