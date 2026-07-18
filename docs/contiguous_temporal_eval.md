# Contiguous temporal evaluation

`artifacts/contiguous-test-v1/manifest.json` fixes six complete, sequential
teacher timelines. No row from the sparse sampled validation TMGD is used as a
target. GOAT and GuitarSet tracks are from their official `test` split.
Guitar-TECHS has no `test` rows in this corpus, so two held-out `validation`
direct-input performances are explicitly marked `evaluation_role=test`.

Each manifest track has two independently exported feature files:

- `plain_d2w3`: the original d2/w3 binarizer and frontend;
- `hcontrast15_d2w3`: corrected harmonic-local-contrast features at 1.5
  semitones.

The WAV export sidecars bind the exact source WAV SHA-256, binarizer,
frontend/context geometry, frame count and strict-causality declaration. The
evaluator rejects a changed WAV/events file, a mismatched export sidecar, or a
score TSV with a different length/MIDI geometry.

## Producing score TSVs later

For a selected feature set and matching activity/onset models, write files with
the exact names `<track-key>.activity.tsv` and `<track-key>.onset.tsv`:

```powershell
$featureSet = 'plain_d2w3'
$activityModel = 'PATH\activity\model.tmgmmod'
$onsetModel = 'PATH\onset\model.tmgmmod'
$scores = "artifacts\contiguous-test-v1\scores\MODEL_NAME\$featureSet"
$manifest = Get-Content artifacts\contiguous-test-v1\manifest.json -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force $scores | Out-Null
foreach ($track in $manifest.tracks) {
  $dataset = Join-Path artifacts\contiguous-test-v1 $track.feature_sets.$featureSet.dataset
  native\BUILD\Release\tmgm_predict.exe $dataset $activityModel --output (Join-Path $scores "$($track.key).activity.tsv")
  native\BUILD\Release\tmgm_predict.exe $dataset $onsetModel --output (Join-Path $scores "$($track.key).onset.tsv")
}
```

Do not calibrate thresholds on these six test tracks. Threshold selection must
come from a separate validation split.

## Evaluation

For d2/w3 training labels (delay 2, width 3):

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_contiguous_tracks.py `
  --manifest artifacts\contiguous-test-v1\manifest.json `
  --feature-set plain_d2w3 `
  --scores-root artifacts\contiguous-test-v1\scores\MODEL_NAME\plain_d2w3 `
  --training-onset-delay-frames 2 --onset-width-frames 3 `
  --target-aligned-tolerances 2 3 4 `
  --wall-clock-tolerances 2 3 4 6 `
  --output artifacts\contiguous-test-v1\scores\MODEL_NAME\plain_d2w3\evaluation.json
```

Change only `--feature-set` and the score directory for the contrast frontend.
For a d3 model pass `--training-onset-delay-frames 3`; the wall-clock view
always keeps teacher delay zero.

The JSON includes:

- frame-exact activity and training-aligned onset metrics;
- one-to-one same-pitch onset event matching at exact and tolerant windows;
- a separate wall-clock onset view, preventing a larger training delay from
  hiding inference latency;
- single-note, polyphonic/chord, target-polyphony and MIDI 40-59 metrics;
- teacher-target recall for retriggers, notes active on the preceding frame,
  chord onsets and single onsets;
- micro aggregates per track, per source and over all six tracks.
