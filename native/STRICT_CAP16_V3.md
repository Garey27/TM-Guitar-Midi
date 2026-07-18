# strict-cap16-v3 native cross-bank coordinator

`StrictCap16V3Coordinator` is the allocation-free inference seam for the
frozen TM listening model in
`artifacts/native-listening-comparison-20260718-strict-cap16-v3`.

## Composition

The deployment is not one ordinary `TMGMBND` bundle. It has five logical
bundle IDs and four unique packed feature rows:

| Logical bundle | Packed row | Selected members |
| --- | --- | --- |
| `plain` | `plain` (7973 bits) | 3 activity |
| `hcontrast-d2` | `hcontrast` (7973 bits) | 2 activity |
| `hcontrast-d3` | the same `hcontrast` row | 8 onset |
| `hprofile-d3` | `hprofile` (16205 bits) | 1 activity + 1 onset |
| `cattack-d3` | `cattack` (10374 bits) | 1 activity + 1 onset |

This produces the exact frozen 7-member activity and 10-member onset
ensembles. Dummy opposite heads required by the old per-bank bundle container
are evaluated by the underlying predictor but are never routed into global
fusion.

The native manifest in `strict_cap16_v3.cpp` pins:

- every bundle checksum and embedded feature fingerprint;
- every semantic frontend/binarizer fingerprint;
- each selected bundle member, head, calibration threshold, and float32 scale;
- the cross-bank member order and its source-artifact SHA-256;
- MIDI 40--88, 22050 Hz, hop 256, quantization 1024, and global thresholds
  `-169` / `-492`.

`load(package_root)` performs file I/O and checksum validation. `prepare(...)`
accepts five host-resolved bundle paths in manifest order and retains native
file/checksum validation. Both must run outside the audio callback.

## Realtime API

Each `predict_frame` call accepts four caller-owned LSB-first `uint32` packed
rows. It writes, for all 49 pitches:

- the unquantized global float32 activity/onset mean;
- the exact quantized integer score used by existing `TMGM_SCORES_V1` files;
- the frozen-threshold activity/onset decision.

All scratch storage and member routing are prepared once. The frame call is
`noexcept` and performs no heap allocation, file I/O, locks, JSON parsing, or
string lookup. A coordinator is single-stream state and must not be called
concurrently; prepare one instance per realtime stream.

## Proven parity

`strict_cap16_v3_parity_test.cpp` runs the complete 935-frame `2222` production
fixture through the four packed rows and compares all 49 outputs against:

- the 17 frozen selected-member TSV files (bit-exact float32 global means);
- `activity-final.tsv` and `onset-primary.tsv` (exact quantized scores);
- decisions at `-169` and `-492`.

The test also replaces global allocation functions and requires zero
allocations across the complete frame loop.

## Streaming frontend

`StrictCap16V3StreamingFrontend` converts already-resampled mono float32
22050-Hz audio into the coordinator's four packed rows. `load(...)` verifies
the compact `TMGMFRT` artifact, creates the pinned NumPy PocketFFT plan, and
allocates all state outside the callback. The streaming path uses one shared
2048-point real FFT, emits frame zero after sample one and later frames every
256 samples, applies causal delays `(0, 1, 2, 4, 8, 16, 32)`, and packs the
four authenticated thermometer banks.

`TMGMFRT` format version 2 stores every kept thermometer entry as
`raw_column:u32`, `threshold:f32`, `equality_ulps:u32`. The normal policy is
exact float32 `value >= threshold` (`equality_ulps == 0`). One narrow,
fail-closed portability record is authenticated by the whole-file checksum:

- bank `cattack`, output column 8748, raw column 9336;
- delay 16, `contrast_fast_slow_attack:midi_56`, quantile 0;
- accept a value up to four positive-float32 ULP below the threshold as an
  equality tie.

This record compensates a measured four-ULP code-generation difference
between NumPy's Windows PocketFFT wheel and the same pinned source compiled by
the native MSVC toolchain. The loader rejects any missing, moved, widened, or
additional equality record. It is not a global threshold relaxation.

`strict_cap16_v3_frontend_parity_test.cpp` processes the complete 935-frame
`2222` fixture both contiguously and through deliberately irregular host block
sizes. It requires all four packed rows to be bit-exact against Python TMGD,
the same output hash for both partitions, and zero callback allocations.

Host integration is synchronous: during the frontend callback, pass the
borrowed `StrictCap16V3FrameInput` directly to
`StrictCap16V3Coordinator::predict_frame`. Do not retain its row pointers.
Use one frontend and coordinator per realtime stream and call `reset()` on
transport/stream reset.

## Current boundary

The native seam begins at already-resampled mono float32 22050-Hz audio. It
does not implement host-rate resampling, the note-state decoder, acoustic
velocity, or JUCE/VST integration. Those remain host responsibilities. Legacy
bundles cannot carry their semantic fingerprint internally, so both the
coordinator and frontend loaders separately pin the frozen package identities
before realtime processing begins.
