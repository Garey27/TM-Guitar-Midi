# TMGMBND v1 inference bundle

`TMGMBND` is a deterministic little-endian, inference-only container for two
independent ordered TM ensembles: activity and onset. It contains no paths,
timestamps, JSON, training-state counter bits, or runtime CUDA dependency.

## Header

The fixed header is 256 bytes. Unsigned integers are little-endian. Signed
thresholds are two's-complement `int32`; scales are IEEE-754 binary32.

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `char[8]` | `TMGMBND\0` |
| 8 | `uint32` | version, `1` |
| 12 | `uint32` | header bytes, `256` |
| 16 | `uint32` | member descriptor bytes, `192` |
| 20 | `uint32` | flags, zero |
| 24 | `uint32` | total member count |
| 28 | `uint32` | activity member count, at least `1` |
| 32 | `uint32` | onset member count, at least `1` |
| 36 | `uint32` | binary feature count |
| 40 | `uint32` | output count |
| 44 | `int32` | minimum MIDI note |
| 48 | `int32` | maximum MIDI note |
| 52 | `uint32` | audio sample rate |
| 56 | `uint32` | analysis hop samples |
| 60 | `uint32` | activity fusion, `1 = mean` |
| 64 | `uint32` | activity quantization |
| 68 | `int32` | activity ensemble threshold |
| 72 | `uint32` | onset fusion, `1 = mean` |
| 76 | `uint32` | onset quantization |
| 80 | `int32` | onset ensemble threshold |
| 84 | `uint32` | reserved, zero |
| 88 | `uint64` | descriptor offset, `256` |
| 96 | `uint64` | descriptor bytes |
| 104 | `uint64` | payload offset, aligned to 8 |
| 112 | `uint64` | payload bytes |
| 120 | `uint64` | complete file bytes |
| 128 | `uint8[32]` | activity ID-order SHA-256 |
| 160 | `uint8[32]` | onset ID-order SHA-256 |
| 192 | `uint8[32]` | companion feature-source SHA-256 |
| 224 | `uint8[32]` | canonical bundle SHA-256 |

Each order hash covers the IDs for that head joined by one NUL byte. The
canonical bundle checksum covers the complete file after zeroing bytes
224..255. The feature fingerprint currently identifies exact companion
binarizer file bytes; a frontend-embedded successor should replace it with a
complete semantic frontend/context fingerprint.

## Member descriptor

Descriptors are stored in inference order: all activity members followed by
all onset members. Within each head this is the exact mean-reduction order.

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `char[64]` | ASCII member ID, zero-padded |
| 64 | `uint32` | head: `1 = activity`, `2 = onset` |
| 68 | `uint32` | flags, zero |
| 72 | `int32` | calibration raw-score threshold |
| 76 | `float32` | robust normalization scale |
| 80 | `uint32` | feature count |
| 84 | `uint32` | output count |
| 88 | `uint32` | clause count |
| 92 | `uint32` | literal count, `2 * features` |
| 96 | `uint32` | included sparse literal count |
| 100 | `uint32` | weight width, `16` |
| 104 | `uint8[32]` | complete source TMGMMOD SHA-256 |
| 136 | `uint64` | clause-offset section offset |
| 144 | `uint64` | clause-offset section bytes |
| 152 | `uint64` | literal-ID section offset |
| 160 | `uint64` | literal-ID section bytes |
| 168 | `uint64` | weight section offset |
| 176 | `uint64` | weight section bytes |
| 184 | `uint64` | reserved, zero |

The calibration threshold can legitimately differ from the checkpoint's
stored score threshold when the ensemble was fitted on an isolated calibration
split. The ensemble artifact is authoritative for runtime normalization.

Every section starts on an 8-byte boundary and sections are canonical,
contiguous, and zero-padded. Per-member payload:

1. `uint32 clause_offsets[clause_count + 1]`;
2. `uint16 literal_ids[included_literal_count]`;
3. clause-major `int16 weights[clause_count][output_count]`.

Literal IDs use TMU's positive-first layout. `literal < feature_count` requires
that feature to be true; otherwise `literal - feature_count` must be false.
IDs inside a clause are strictly increasing. Empty clauses remain representable
but are false at inference, matching native CUDA/TMU. Raw scores accumulate in
`int64` and clamp to `int32` after all firing clauses are added.

## Exact per-head mean fusion

Activity and onset use the same algorithm but independent member ranges,
quantization, and final threshold:

```text
z[m] = (float32(raw[m]) - float32(member_threshold[m])) / float32(scale[m])
mean = sequential_float32_sum(z) / float32(member_count)
score = round_to_nearest_ties_even(float64(mean) * quantization)
score = clip(score, INT32_MIN, INT32_MAX - 1)
prediction = score >= ensemble_threshold
```

This exactly matches the current NumPy production oracles. It must not be
replaced by member voting, raw-score averaging, `std::round`, inverse scales,
fast-math reassociation, or a float comparison before integer quantization.
