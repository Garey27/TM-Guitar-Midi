# Datasets and local corpus layout

Audio datasets are not mirrored by this repository. Download each dataset from
its official source and accept its license before training. Exact citations,
links, and license identifiers are maintained in [CREDITS.md](../CREDITS.md).

The preview corpus used three sources:

- GuitarSet;
- Guitar-TECHS;
- GOAT.

NeuralNote/Basic Pitch was run offline as a teacher for the audio corpus. The
training pipeline consumes both posterior frames (`.nnpg`) and decoded note
events (`.events.tsv`). Original annotations remain useful for independent
validation and timing audits; they are not blindly mixed with teacher labels.

## Recommended local layout

```text
datasets/
  audio/
    guitarset/...
    guitar-techs/...
    goat/...
  teacher/
    corpus.tsv
    guitarset/<split>/<track>.nnpg
    guitarset/<split>/<track>.events.tsv
    guitar-techs/<split>/<track>.nnpg
    guitar-techs/<split>/<track>.events.tsv
    goat/<split>/<track>.nnpg
    goat/<split>/<track>.events.tsv
```

`corpus.tsv` is UTF-8, tab-separated, and has this header:

```text
split\tsource\tid\tinput\toutput_rel\tgroup
```

- `split`: `train`, `validation`, or `test`;
- `source`: stable source identifier;
- `id`: globally unique track identifier;
- `input`: absolute or manifest-relative path to the source WAV;
- `output_rel`: path below `teacher/`, without relying on a local drive letter;
- `group`: performer/session/project group used to avoid leakage.

## Split policy

- GuitarSet: performers `00..03` train, `04` validation, `05` test;
- Guitar-TECHS: performer/session grouping is preserved, and direct/mic captures
  of the same performance stay in the same role;
- GOAT: use the dataset's project split and keep its official holdout separate;
- alternate captures and derived views must never be split across train/test;
- validation thresholds are fitted only on validation; listening examples and
  personal recordings are never reported as unseen test data.

The corpus reader performs deterministic source- and group-stratified selection
for small experiments. Full export uses every row in the requested split.

## Audio/teacher contract

- teacher sample rate: 22,050 Hz;
- hop: 256 samples;
- channel policy: channel zero before resampling;
- output pitch range: MIDI 40..88;
- event time is absolute seconds, not raw MIDI ticks;
- Guitar-TECHS tempo maps must be resolved to absolute time before comparison.

Do not commit downloaded audio, teacher outputs, sampled matrices, or local
manifests containing machine-specific paths. They are ignored by `.gitignore`.
