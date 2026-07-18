# Training and deployment export

This document separates the reproducible training stages. A plugin build never
needs Python or CUDA; only creation of new weights does.

## 1. Environment

Python training is pinned to Python 3.12 because that is the tested TMU target.

```bash
python3.12 -m venv .venv
# Windows: .venv\Scripts\python -m pip install -U pip
# Linux/macOS: .venv/bin/python -m pip install -U pip
python -m pip install -e ".[train,test]"
```

The `train` extra installs `cair/tmu` at the exact commit used by the preview
experiments. TMU is the reference Python implementation; deployment inference
uses this repository's portable native runtime.

## 2. Prepare the teacher corpus

1. Download GuitarSet, Guitar-TECHS, and GOAT from their official sources.
2. Build and run the pinned headless NeuralNote exporter documented in
   [`tools/neuralnote-teacher`](../tools/neuralnote-teacher), retaining the full
   note/onset posterior grid and decoded note events.
3. Produce the `corpus.tsv` and directory contract described in
   [Datasets](DATASETS.md).
4. Inspect one teacher file before a long export:

```bash
tmgm-rt inspect-nnpg datasets/teacher/goat/train/item_0.nnpg
```

The repository does not relabel audio at plugin runtime. Teacher generation is
an offline data-preparation step.

## 3. Prove the pipeline with a small TMU run

Always begin with a tiny run and a one-track overfit. This catches broken time
alignment, channel selection, pitch ranges, or tempo conversion before a full
training job.

```bash
tmgm-rt train \
  --corpus datasets/teacher/corpus.tsv \
  --teacher-root datasets/teacher \
  --train-tracks 4 --validation-tracks 2 --frames-per-track 800 \
  --epochs 2 --clauses 128 --max-literals 16 \
  --platform CPU --output runs/smoke/model.pkl

python scripts/train_one_track_overfit.py --help
```

Listen to the overfit transcription and compare it to the teacher. A model that
cannot fit one track should not be scaled up.

## 4. Export the native sampled corpus

The exporter computes causal STFT+, past-only temporal context, train-only
quantile thresholds, and bit-packs the result into `TMGMDAT` files. It writes a
per-track cache so interrupted exports can resume without holding a multi-GiB
binary matrix in RAM.

```bash
python scripts/export_full_native.py \
  --corpus datasets/teacher/corpus.tsv \
  --teacher-root datasets/teacher \
  --output-dir runs/native/plain \
  --frames-per-track 800
```

Run separate exports for the deployment feature banks using the relevant
frontend switches shown by `python scripts/export_full_native.py --help`:

- plain STFT+;
- harmonic local contrast (`d2` and `d3` context experiments);
- harmonic local profile;
- contrast attack features.

The fitted binarizer must come only from train rows and must be applied unchanged
to validation/test rows.

## 5. Build the native trainer

CPU inference and tests:

```bash
cmake --preset native-release
cmake --build --preset native-release
ctest --preset native-release
```

For CUDA training, install a CUDA toolkit supported by your compiler and set:

```bash
cmake --preset native-cuda -DTMGM_CUDA_ARCHITECTURES=89
cmake --build --preset native-cuda
```

Architecture `89` is appropriate for an RTX 4090; choose the architecture for
your own GPU.

## 6. Train activity and onset heads

Activity and onset are trained independently. Always provide held-out
validation data and save the best validation checkpoint rather than the final
epoch.

```bash
build/native-cuda/native/tmgm_train \
  runs/native/plain/train.tmgd \
  --validation runs/native/plain/validation.tmgd \
  --head activity --epochs 100 --validation-patience 8 \
  --max-literals 16 \
  --output runs/models/plain/activity.tsv \
  --model runs/models/plain/activity.tmgmmod

build/native-cuda/native/tmgm_train \
  runs/native/plain/train.tmgd \
  --validation runs/native/plain/validation.tmgd \
  --head onset --epochs 100 --validation-patience 8 \
  --max-literals 16 \
  --output runs/models/plain/onset.tsv \
  --model runs/models/plain/onset.tmgmmod
```

Use `tmgm_train --help` for clause count, threshold, specificity, negative
sampling, seed, and device options. Store every command, seed, feature
fingerprint, dataset manifest hash, and validation result with the checkpoint.
On Windows, invoke the same Ninja-built trainer as
`build/native-cuda/native/tmgm_train.exe`.

## 7. Calibrate and package an ensemble

Members are selected and thresholds are calibrated only on the frozen
validation role. Export each selected bank to an authenticated
`.tmgmbundle`, then export the matching frontend constants:

```bash
python scripts/export_native_ensemble_bundle.py --help
python scripts/export_strict_cap16_v3_frontend.py \
  --package runs/release-package \
  --output runs/release-package/native-frontend/strict-cap16-v3.tmgmfront \
  --without-2222-audio
```

The deployment package must contain:

```text
bundles/plain.tmgmbundle
bundles/hcontrast-d2.tmgmbundle
bundles/hcontrast-d3.tmgmbundle
bundles/hprofile-d3.tmgmbundle
bundles/cattack-d3.tmgmbundle
native-frontend/strict-cap16-v3.tmgmfront
```

Copy a newly authenticated package into a new versioned directory under
`models/`; never overwrite a released model in place. Update its model card and
SHA-256 manifest, run all native/frontend parity tests, then build the plugins
with `-DTMGM_MODEL_PACKAGE=<path>`.

## 8. Required validation before release

- Python unit tests;
- native CTest suite, including CPU/reference and frontend parity;
- block-partition invariance at 44.1 and 48 kHz;
- balanced MIDI lifecycle: no duplicate NoteOn, orphan NoteOff, or stuck notes;
- held-out contiguous-track metrics;
- listening comparison on complete tracks from all three datasets;
- real-host checks in REAPER and at least one additional VST3/LV2 host;
- archive inspection proving the model resources are inside every bundle.

Treat teacher-relative metrics as distillation metrics, not ground-truth claims.
