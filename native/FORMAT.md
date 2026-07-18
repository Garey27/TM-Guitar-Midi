# Native TMGMDAT v1

This C++ loader exactly mirrors `src/tmgm_rt/native_dataset.py`. All integers
are little-endian. A file contains a fixed 256-byte header followed by four
contiguous payloads: bit-packed features, bit-packed activity labels,
bit-packed onset labels, and `uint32` onset training row indices. Extra bytes
are rejected.

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `char[8]` | Magic `TMGMDAT\0` |
| 8 | `uint32` | Version (`1`) |
| 12 | `uint32` | Header bytes (`256`) |
| 16 | `uint32` | Packed word bits (`64`) |
| 20 | `uint32` | Flags (`0`) |
| 24 | `uint64` | Frame count |
| 32 | `uint32` | Feature count |
| 36 | `uint32` | Feature words per row |
| 40 | `uint32` | Note count |
| 44 | `uint32` | Label words per row |
| 48 | `int32` | Minimum MIDI note |
| 52 | `int32` | Maximum MIDI note |
| 56 | `uint32` | Audio sample rate |
| 60 | `uint32` | Hop size |
| 64 | `uint64` | Onset training index count |
| 72..135 | `uint64[8]` | Offset/byte-size pairs for four payloads |
| 136 | `uint64` | Sampling seed |
| 144 | `uint8[32]` | SHA-256 of all four concatenated payloads |
| 176..255 | `uint8[80]` | Reserved zero bytes |

Binary column `c` is bit `c % 64` of word `c / 64`, LSB-first. Every row is
padded independently and padding bits must be zero. This representation can be
copied directly to CUDA device memory.

## Build and smoke test

```powershell
cmake -S . -B build-vs2019 -G "Visual Studio 16 2019" -A x64 -T cuda=12.9 `
  -DTMGM_ENABLE_CUDA=ON -DTMGM_CUDA_ARCHITECTURES=89
cmake --build build-vs2019 --config Release
ctest --test-dir build-vs2019 -C Release --output-on-failure
.\build-vs2019\Release\tmgm_dataset_info.exe --make-demo demo.tmgm --cuda-smoke
```

Inspect a dataset exported by Python:

```powershell
.\build-vs2019\Release\tmgm_dataset_info.exe path\to\one-track.tmgm --cuda-smoke
```
