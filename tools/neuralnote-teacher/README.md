# NeuralNote teacher exporter

This directory contains the project's modified headless exporter used to create
the `.nnpg`, `.events.tsv`, MIDI, and metadata files consumed by training. It is
offline tooling and is not linked into release plugins.

The exporter is based on
[DamRsn/NeuralNote](https://github.com/DamRsn/NeuralNote) commit
`f979e51dfeab54d5921858af39403308ab06e60c`. The changed/new files are marked
and remain under NeuralNote's Apache-2.0 license; see
`LICENSE-NeuralNote-Apache-2.0.txt`.

## Build

```bash
git clone https://github.com/DamRsn/NeuralNote.git NeuralNote
git -C NeuralNote checkout f979e51dfeab54d5921858af39403308ab06e60c
git -C NeuralNote apply /path/to/TM-Guitar-Midi/tools/neuralnote-teacher/neuralnote-teacher.patch
cp /path/to/TM-Guitar-Midi/tools/neuralnote-teacher/NeuralNoteBatch.cpp NeuralNote/Tools/

cmake -S NeuralNote -B NeuralNote/build-batch \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_BATCH_TOOLS=ON
cmake --build NeuralNote/build-batch --config Release \
  --target NeuralNoteBatch --parallel 2
```

Follow NeuralNote's platform-specific dependency setup for ONNX Runtime and its
submodules before configuring.

## Input manifest

Create a UTF-8 tab-separated file:

```text
id\tinput\toutput_rel
item_0\t/path/to/item_0.wav\tgoat/train/item_0
```

Then run:

```bash
NeuralNoteBatch --output-root datasets/teacher \
  --input-list datasets/teacher/jobs.tsv \
  --note-sensitivity 0.7 --split-sensitivity 0.5 \
  --minimum-note-ms 125
```

Use `--limit N` for a smoke run and `--force` to replace an existing track.
The exporter records the upstream commit, source identity, channel policy,
teacher grid, thresholds, and elapsed time in each `.meta.tsv` file.
