# TM Guitar MIDI

Experimental, realtime polyphonic guitar-to-MIDI built with Tsetlin Machines.

TM Guitar MIDI analyses a clean DI guitar signal with a causal STFT+ frontend,
runs an ensemble of multi-output Tsetlin Machines, and turns the resulting
activity/onset evidence into MIDI note events. The inference path is native
C++ and does not require Python, CUDA, ONNX Runtime, TensorFlow, or a network
connection.

> **Preview status:** the current model is useful for testing, but the project
> is not yet a stable production release. Builds are unsigned and the
> repository remains private until the Windows, macOS, and Linux artifacts have
> been tested in real hosts.

## Current features

- realtime, causal audio-to-MIDI processing;
- polyphonic output over MIDI notes 40..88 (49 pitches, standard tuning through
  24 frets);
- separate activity and onset TM ensembles;
- acoustic onset/velocity tracking without multiplying velocity by classifier
  confidence;
- mono or stereo audio input (stereo is analysed as mono);
- 44.1 kHz and 48 kHz host sample rates;
- dry-audio switch so a synth can be placed directly after the converter;
- no heap allocation in the realtime prediction callback;
- VST3 for Windows and macOS; LV2 for Linux.

## Signal path

```text
guitar DI
  -> causal resampler (22.05 kHz analysis rate)
  -> STFT+ feature banks and past-only temporal context
  -> quantile thermometer encoding
  -> Tsetlin Machine activity/onset ensemble
  -> causal acoustic note-state tracker and velocity
  -> MIDI note on/off
```

See [Architecture](docs/ARCHITECTURE.md) for the detailed contracts and model
layout.

## Build from source

The project uses CMake 3.24+ and Ninja, and fetches the pinned JUCE dependency
during configuration. Choose the preset for the current platform:

```bash
cmake --preset windows-vst3   # macos-vst3 or linux-lv2 on those platforms
cmake --build --preset windows-vst3
ctest --preset windows-vst3
```

Platform-specific prerequisites, target names, install locations, and the
offline dependency option are documented in [Building](docs/BUILDING.md).

## Use in a DAW

1. Create an audio track and select the clean guitar input.
2. Enable input monitoring.
3. Insert **TM Guitar MIDI** first in the effects chain.
4. Insert a MIDI instrument immediately after it.
5. Start with `Acoustic Gain` around `+20 dB` for a quiet DI signal and reduce
   it if false attacks appear.
6. Leave `Dry Audio` off to hear only the instrument.

The status parameters must report that the model is loaded and that the host
sample rate is supported.

## Training and model weights

The repository contains the complete TM/STFT+ training and native export code.
The small inference package used by the preview build is versioned under
[`models/strict-cap16-v3`](models/strict-cap16-v3). Tagged preview releases may
also publish it as a separately labelled, non-commercial research-weights
archive.

Start with:

- [Training](docs/TRAINING.md) — teacher generation, feature export, CPU/TMU and
  native CUDA training, validation, and deployment export;
- [Datasets](docs/DATASETS.md) — expected local layout and split rules;
- [Model card](models/strict-cap16-v3/MODEL_CARD.md) — scope, limitations,
  provenance, and checksums.

Training data is intentionally not committed. Local paths, audio, MIDI,
posteriorgrams, sampled matrices, checkpoints, and experiment output are
ignored by git.

## Repository layout

```text
plugin/                    JUCE processor and realtime note tracker
native/                    C++/CUDA TM runtime, trainer, formats, and tests
src/tmgm_rt/               Python STFT+, corpus, TMU, evaluation, and export
scripts/                   Reproducible command-line research tools
models/strict-cap16-v3/    Versioned preview inference package
docs/                      Build, training, dataset, and architecture docs
.github/workflows/         Cross-platform CI and release packaging
```

## Licensing

Project source code is licensed under **AGPL-3.0-only**, matching the
open-source licensing path of the pinned JUCE version. The preview weights have
a separate license because their training corpus includes CC-BY-NC-4.0 data;
see [WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md) before redistribution. Commercial
use of the current GOAT-derived package is not permitted.

Third-party software, datasets, teacher models, papers, and required
attributions are listed in [CREDITS.md](CREDITS.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
