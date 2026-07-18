# TMGMDAT v1 native training format

`TMGMDAT` is the deterministic bridge from the proven Python one-track
experiment to a native C++/CUDA trainer. The file can be memory-mapped. Every
integer is little-endian and the fixed header occupies 256 bytes.

## Header

| Offset | C/C++ type | Field |
|---:|---|---|
| 0 | `char[8]` | magic: `TMGMDAT\0` |
| 8 | `uint32_t` | version: `1` |
| 12 | `uint32_t` | header bytes: `256` |
| 16 | `uint32_t` | word bits: `64` |
| 20 | `uint32_t` | flags: `0` |
| 24 | `uint64_t` | frame count |
| 32 | `uint32_t` | binary feature count |
| 36 | `uint32_t` | feature words per row |
| 40 | `uint32_t` | note count |
| 44 | `uint32_t` | label words per row |
| 48 | `int32_t` | minimum MIDI pitch |
| 52 | `int32_t` | maximum MIDI pitch |
| 56 | `uint32_t` | sample rate |
| 60 | `uint32_t` | hop size |
| 64 | `uint64_t` | onset training index count |
| 72 | `uint64_t` | feature payload offset |
| 80 | `uint64_t` | feature payload bytes |
| 88 | `uint64_t` | activity payload offset |
| 96 | `uint64_t` | activity payload bytes |
| 104 | `uint64_t` | onset payload offset |
| 112 | `uint64_t` | onset payload bytes |
| 120 | `uint64_t` | onset-index payload offset |
| 128 | `uint64_t` | onset-index payload bytes |
| 136 | `uint64_t` | sampling seed |
| 144 | `uint8_t[32]` | SHA-256 of all payloads concatenated |
| 176 | `uint8_t[80]` | reserved, zero in v1 |

Do not map this table onto an unpacked compiler struct without either
`#pragma pack(push, 1)` or explicit field reads; compiler padding is otherwise
implementation-defined.

## Payloads

The four payloads are contiguous and ordered as follows:

1. Feature matrix: `[frame_count][feature_words_per_row]` of `uint64_t`.
2. Activity labels: `[frame_count][label_words_per_row]` of `uint64_t`.
3. Onset labels: `[frame_count][label_words_per_row]` of `uint64_t`.
4. Rows used to train the onset head: `onset_index_count` values of `uint32_t`.

For every packed matrix, column `c` is bit `c & 63` of word `c >> 6`. Bits
above the declared column count are zero. Repeated onset indices are intentional:
they preserve the exact oversampling policy that produced the successful Python
overfit.

The adjacent `.tmgd.json` sidecar records source identity, frontend/context
configuration and quantile-encoder dimensions. It is diagnostic metadata and is
not required by the trainer. `TMGMDAT v1` contains already-binarized training
features; exporting quantile thresholds for native audio inference is a separate
model/frontend interchange step.

## Full-corpus pair

`scripts/export_full_native.py` produces a compatible pair named `train.tmgd`
and `validation.tmgd`. The default selection uses every manifest entry in both
splits and 800 sampled frames per track. Training frames use the existing
chord/sustain/silence-balanced policy; validation frames retain deterministic
natural temporal sampling.

There is exactly one quantile encoder for the pair. Its thresholds and
non-constant `keep_columns` mask are fitted on all sampled training rows, saved
as `global-quantile-thermometer.npz`, then reused unchanged for validation. The
two TMGMDAT headers must therefore have the same `feature_count`. Do not fit a
second encoder on validation: even if its width happens to match, its binary
columns would have different meanings.

The exporter caches each extracted track independently and an exact continuous
training memmap. Interrupted runs resume from these artifacts. Encoding and bit
packing are batched; the full `rows x binary_features` uint32 matrix (more than
10 GiB for the default full split) is never materialized. Sidecars record cache
signatures, the shared encoder SHA-256, per-source track/row counts and the
selected track identities.
