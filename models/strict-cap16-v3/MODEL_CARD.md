# strict-cap16-v3 model card

## Summary

`strict-cap16-v3` is the first preview deployment ensemble for TM Guitar MIDI.
It predicts activity and onset evidence for MIDI 40..88 from causal STFT+
features and is consumed by the native acoustic note-state tracker.

- analysis sample rate: 22,050 Hz;
- hop size: 256 samples;
- pitch outputs: 49;
- temporal context: delays 0, 1, 2, 4, 8, 16, 32;
- ensemble: seven activity and ten onset members;
- maximum included literals per clause/member: 16;
- feature packages: plain, harmonic contrast, harmonic profile, contrast attack;
- plugin host rates: 44.1 and 48 kHz through the bundled causal resampler.

## Training provenance

The training corpus combined GuitarSet, Guitar-TECHS, and GOAT audio. Offline
NeuralNote/Spotify Basic Pitch outputs supplied teacher posterior/event targets.
TMU was the reference Python Tsetlin Machine implementation; final deployment
members were trained/exported by this repository's native pipeline.

Exact dataset and software citations are in the repository `CREDITS.md`.

## Intended use

- research and subjective testing of realtime polyphonic guitar-to-MIDI;
- comparison of TM architectures, feature banks, and causal note trackers;
- non-commercial preview builds under the accompanying weights license.

## Limitations

- experimental model, not a claim of state-of-the-art transcription accuracy;
- trained primarily on clean guitar recordings and teacher-generated targets;
- dynamics, pickups, input gain, noise, distortion, and alternate tunings can
  shift behaviour;
- output is limited to MIDI 40..88;
- teacher-relative validation can inherit teacher mistakes;
- no pitch bend, MPE, articulation labels, or string identity in this version.

## Files

The five `.tmgmbundle` files contain the selected TM members. The
`.tmgmfront` file contains authenticated frontend/binarizer constants. All are
required; the plugin fails closed if the package is incomplete or mismatched.

See `SHA256SUMS` for file identities and
[`WEIGHTS_LICENSE.md`](../../WEIGHTS_LICENSE.md) for redistribution terms.
