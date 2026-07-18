# Credits and data provenance

TM Guitar MIDI is built on original realtime DSP and Tsetlin Machine code, with
the third-party software, research, and datasets credited below. Legal notices
for redistributed components are collected in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Important distinction: source code and model weights

The source code in this repository and a trained model are separate works with
separate provenance.

The current `strict-cap16-v3` preview weights were trained on offline teacher
labels generated with NeuralNote/Spotify Basic Pitch from 562 source tracks:

| Training source | Tracks |
| --- | ---: |
| GuitarSet | 240 |
| Guitar-TECHS | 184 |
| GOAT | 138 |

An additional 101 tracks were used for validation: 60 GuitarSet, 24
Guitar-TECHS, and 17 GOAT. No raw dataset audio, annotations, prepared feature
caches, or teacher posteriorgrams are distributed by this repository.

**Release restriction:** GOAT is a restricted-access dataset licensed under
CC BY-NC 4.0. Until separate written permission is obtained from its
rightsholders, the GOAT-derived preview weights must be treated as a
non-commercial research artifact. They must not be released under the source
code's AGPL license, described as unrestricted "open weights", or bundled in a
commercially reusable plug-in distribution. A clean public model should be
retrained without GOAT, for example from GuitarSet and Guitar-TECHS, and should
have its own model card and model license.

Whether trained weights are legally an adaptation of particular training data
can vary by jurisdiction. The conservative release policy above does not
replace legal advice or any additional terms accepted when access to a dataset
was granted.

## Training datasets

### GuitarSet

- Dataset: [GuitarSet v1.1.0 on Zenodo](https://zenodo.org/records/3371780)
- DOI: [10.5281/zenodo.3371780](https://doi.org/10.5281/zenodo.3371780)
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Creators: Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye, and Juan
  P. Bello

Please cite:

> Q. Xi, R. M. Bittner, J. Pauwels, X. Ye, and J. P. Bello, "GuitarSet: A
> Dataset for Guitar Transcription," Proceedings of the 19th International
> Society for Music Information Retrieval Conference (ISMIR), 2018,
> pp. 453-460. [Paper](https://ismir2018.ismir.net/doc/pdfs/188_Paper.pdf)

The downloaded dataset is CC BY 4.0. The separate
[GuitarSet code repository](https://github.com/marl/GuitarSet) is MIT-licensed;
that does not change the dataset license.

### Guitar-TECHS

- Dataset: [Guitar-TECHS on Zenodo](https://zenodo.org/records/14963133)
- Project site: [guitar-techs.github.io](https://guitar-techs.github.io/)
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Dataset-record creators: Hegel Emmanuel Pedroza Villalobos, Termeh Taheri,
  Wallace Abreu, Ryan Corey, and Iran R. Roman

Please cite:

> H. Pedroza, W. Abreu, R. M. Corey, and I. R. Roman, "Guitar-TECHS: An
> Electric Guitar Dataset Covering Techniques, Musical Excerpts, Chords and
> Scales Using a Diverse Array of Hardware," IEEE ICASSP, 2025, pp. 1-5.
> [DOI: 10.1109/ICASSP49660.2025.10887996](https://doi.org/10.1109/ICASSP49660.2025.10887996)

### GOAT

- Dataset record: [GOAT on Zenodo](https://zenodo.org/records/15690894)
- Dataset DOI: [10.5281/zenodo.15690894](https://doi.org/10.5281/zenodo.15690894)
- Official repository: [JackJamesLoth/GOAT-Dataset](https://github.com/JackJamesLoth/GOAT-Dataset)
- Access: restricted/request-only
- License: [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)
- Authors: Jackson Loth, Pedro Sarmento, Saurjya Sarkar, Zixun Guo, Mathieu
  Barthet, and Mark Sandler

Please cite:

> J. Loth, P. Sarmento, S. Sarkar, Z. Guo, M. Barthet, and M. Sandler, "GOAT:
> A Large Dataset of Paired Guitar Audio Recordings and Tablatures," Proceedings
> of the 26th International Society for Music Information Retrieval Conference
> (ISMIR), 2025. [arXiv:2509.22655](https://arxiv.org/abs/2509.22655)

The GOAT archive, annotations, derived features, and teacher caches must not be
placed in this repository or its releases. Redistribution must additionally
respect any access terms supplied directly by the dataset authors.

## Offline transcription teacher

The released realtime plug-in does **not** run NeuralNote or Basic Pitch.
During training, they were used offline to generate activity and onset teacher
targets.

### NeuralNote

- Project: [DamRsn/NeuralNote](https://github.com/DamRsn/NeuralNote)
- Teacher baseline: commit
  [`f979e51dfeab54d5921858af39403308ab06e60c`](https://github.com/DamRsn/NeuralNote/tree/f979e51dfeab54d5921858af39403308ab06e60c)
- License: [Apache License 2.0](https://github.com/DamRsn/NeuralNote/blob/f979e51dfeab54d5921858af39403308ab06e60c/LICENSE)

The batch exporter in `tools/neuralnote-teacher` adds a headless batch entry
point and access to model posteriorgrams. Its new and modified files are marked,
retain the Apache-2.0 license, and are applied to the pinned upstream checkout.
The complete NeuralNote source tree, bundled third-party libraries, binaries,
and model files are not redistributed here.

### Spotify Basic Pitch

- Project: [spotify/basic-pitch](https://github.com/spotify/basic-pitch)
- License: [Apache License 2.0](https://github.com/spotify/basic-pitch/blob/main/LICENSE)

Please cite:

> R. M. Bittner, J. J. Bosch, D. Rubinstein, G. Meseguer-Brocal, and S. Ewert,
> "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription
> and Multipitch Estimation," IEEE ICASSP, 2022.
> [arXiv:2203.09893](https://arxiv.org/abs/2203.09893)

## Tsetlin Machine software and research

The training experiments used
[CAIR/TMU](https://github.com/cair/tmu), pinned to commit
[`5d6d9da7d3e8c3a15e40f93b94ec882db518c57c`](https://github.com/cair/tmu/tree/5d6d9da7d3e8c3a15e40f93b94ec882db518c57c),
as the behavioral and experimental reference. TMU is MIT-licensed, Copyright
(c) 2025 Centre for Artificial Intelligence Research (CAIR) and the University
of Agder.

The native realtime inference and training implementation in this repository
is independent project code; it does not link TMU at plug-in runtime. The
optional Python training environment installs the pinned TMU dependency.

Relevant research:

- Ole-Christoffer Granmo, ["The Tsetlin Machine -- A Game Theoretic Bandit
  Driven Approach to Optimal Pattern Recognition with Propositional
  Logic"](https://arxiv.org/abs/1804.01508), arXiv:1804.01508.
- Sondre Glimsdal and Ole-Christoffer Granmo, ["Coalesced Multi-Output Tsetlin
  Machines with Clause Sharing"](https://arxiv.org/abs/2108.07594),
  arXiv:2108.07594.

## Runtime and build foundations

### JUCE

The plug-in targets JUCE 8.0.13.

- Project: [juce-framework/JUCE](https://github.com/juce-framework/JUCE)
- Version: [8.0.13](https://github.com/juce-framework/JUCE/releases/tag/8.0.13)
- License: JUCE modules are dual-licensed under
  [AGPLv3 or the commercial JUCE 8 license](https://github.com/juce-framework/JUCE/blob/8.0.13/LICENSE.md)

This repository uses the AGPLv3 route and is licensed AGPL-3.0-only. A party
that does not comply with the AGPL must obtain an appropriate commercial JUCE
license. JUCE also carries the format-SDK and third-party notices listed in its
tagged `LICENSE.md`, including the MIT-licensed VST3 SDK and ISC-licensed LV2
SDK.

### pocketfft

The spectral frontend vendors a locally modified copy of
[pocketfft](https://github.com/mreineck/pocketfft/tree/cpp), obtained through
NumPy's pocketfft submodule at revision
[`33ae5dc94c9cdc7f1c78346504a85de87cadaa12`](https://github.com/mreineck/pocketfft/tree/33ae5dc94c9cdc7f1c78346504a85de87cadaa12).
It is licensed under the BSD 3-Clause License. The local change adds a
caller-provided scratch-buffer path for allocation-free realtime FFT
execution. Copyright and license text are retained in the header and in
`native/third_party/pocketfft/LICENSE.md`.

## No endorsement

The names of dataset authors, research authors, and third-party projects are
used only for attribution. They do not imply sponsorship, endorsement, or
affiliation with TM Guitar MIDI.
