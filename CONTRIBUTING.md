# Contributing

Contributions are welcome after the repository becomes public.

Before opening a pull request:

1. keep audio, datasets, local manifests, and generated checkpoints out of git;
2. preserve causal/realtime behaviour and avoid allocation in `processBlock`;
3. add or update tests for model formats, MIDI lifecycle, and block partitioning;
4. run Python tests and the native CTest suite;
5. document dataset/model provenance for any new weights;
6. do not replace a released model artifact in place.

Source contributions must be compatible with AGPL-3.0-only. Dataset and model
contributions must state their independent licenses and attribution requirements.
