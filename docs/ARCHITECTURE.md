# Architecture

## Realtime contract

The host supplies mono or stereo floating-point audio at 44.1 or 48 kHz. The
plugin averages stereo channels, then uses a causal rational resampler to feed
the fixed 22.05 kHz analysis domain. The resampler, feature extractors, TM
predictors, tracker, and MIDI queue are preallocated before audio processing.

All context is past-only. There is no future-frame lookahead.

## STFT+ frontend

The classification frontend uses a 2048-sample FFT and a 256-sample hop at
22.05 kHz. Pitch evidence covers MIDI 40..88 and includes fundamental,
harmonic, contrast, profile, subharmonic, flux, and attack-related channels.
Four authenticated feature variants are evaluated:

- `plain`;
- `hcontrast`;
- `hprofile`;
- `cattack`.

Each bank stacks frames at delays `0, 1, 2, 4, 8, 16, 32`, then applies a
train-only quantile thermometer encoder. The frozen constants, selected binary
columns, thresholds, feature fingerprints, and portable equality policy live
in `strict-cap16-v3.tmgmfront`.

## Tsetlin Machine ensemble

The deployment coordinator loads five native bundle files spanning the four
feature variants. It combines seven activity members and ten onset members.
Every member predicts all 49 pitches; `cap16` limits literals per learned
clause, not musical polyphony.

The bundle loader authenticates the binary format and feature contract before
enabling inference. A mismatched or incomplete package fails closed and emits
no new notes.

## Acoustic tracker and MIDI lifecycle

The classifier supplies evidence rather than MIDI events directly. A separate
causal acoustic frontend uses 512/4096-point analysis windows, harmonic
ownership, local attack energy, and persistent per-pitch state. Each of the 49
pitches has independent attack, hold, retrigger, release, and refractory state.

MIDI velocity comes from acoustic attack energy. It is deliberately not
multiplied by TM confidence. This prevents a correct but moderately confident
note from becoming artificially quiet, and prevents a false harmonic with a
high class score from automatically becoming full velocity.

Reset, bypass, model failure, and host teardown paths queue a NoteOff for every
active pitch. The engine tests assert no duplicate NoteOn, orphan NoteOff, or
stuck note for their fixtures.

## Binary formats

- `TMGMDAT` (`.tmgd`): bit-packed native training/evaluation rows;
- `TMGMMOD` (`.tmgmmod`): learned TA state, clause weights, configuration, and
  threshold;
- `TMGMBUNDLE` (`.tmgmbundle`): authenticated deployment ensemble;
- `TMGMFRT` (`.tmgmfront`): frozen realtime frontend and binarizer.

The detailed low-level contracts are in `native/FORMAT.md`,
`native/STRICT_CAP16_V3.md`, and `docs/native_dataset_v1.md`.
