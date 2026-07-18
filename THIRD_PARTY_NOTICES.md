# Third-party notices

This file records third-party software and data provenance for TM Guitar MIDI.
The project source is licensed under AGPL-3.0-only; see [`LICENSE`](LICENSE).
Model files require a separate model license and must also respect their
training-data provenance described in [`CREDITS.md`](CREDITS.md).

## Components present in plug-in builds

### JUCE 8.0.13

Project: <https://github.com/juce-framework/JUCE>

License for the exact version:
<https://github.com/juce-framework/JUCE/blob/8.0.13/LICENSE.md>

JUCE Framework modules are dual-licensed under the GNU Affero General Public
License version 3 and the commercial JUCE 8 license. TM Guitar MIDI uses JUCE
under the AGPLv3 option. The complete AGPLv3 text is in this repository's root
`LICENSE` file.

JUCE contains additional dependencies and format SDKs listed in its tagged
`LICENSE.md`. Of particular relevance to the release formats requested by this
project, JUCE 8.0.13 identifies its bundled VST3 SDK as MIT-licensed and its LV2
SDK as ISC-licensed. Release source archives must preserve the corresponding
JUCE notices.

### pocketfft

Project: <https://github.com/mreineck/pocketfft/tree/cpp>

Upstream revision:
<https://github.com/mreineck/pocketfft/tree/33ae5dc94c9cdc7f1c78346504a85de87cadaa12>

Location: `native/third_party/pocketfft/pocketfft_hdronly.h`

This vendored header has been modified to support caller-provided scratch
storage for allocation-free realtime execution.

Copyright (C) 2010-2022 Max-Planck-Society

Copyright (C) 2019-2020 Peter Bell

For the odd-sized DCT-IV transforms:

Copyright (C) 2003, 2007-14 Matteo Frigo

Copyright (C) 2003, 2007-14 Massachusetts Institute of Technology

For the `prev_good_size` search:

Copyright (C) 2024 Tan Ping Liang, Peter Bell

Authors: Martin Reinecke, Peter Bell

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Training and reference software

The following software is not linked into the realtime plug-in. It is credited
because it was used for training, teacher generation, or behavioral reference.

### CAIR/TMU

Project: <https://github.com/cair/tmu>

Pinned reference commit:
<https://github.com/cair/tmu/tree/5d6d9da7d3e8c3a15e40f93b94ec882db518c57c>

MIT License

Copyright (c) 2025 Centre for Artificial Intelligence Research (CAIR) and the
University of Agder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### NeuralNote

Project: <https://github.com/DamRsn/NeuralNote>

Teacher baseline commit:
<https://github.com/DamRsn/NeuralNote/tree/f979e51dfeab54d5921858af39403308ab06e60c>

License: Apache License 2.0
<https://github.com/DamRsn/NeuralNote/blob/f979e51dfeab54d5921858af39403308ab06e60c/LICENSE>

NeuralNote was used offline to generate teacher targets. The new headless
exporter and patch under `tools/neuralnote-teacher` are distributed under the
retained Apache-2.0 terms and are clearly marked as modified. The complete
NeuralNote source tree, binaries, third-party dependency trees, and model files
are not distributed in this repository.

### Spotify Basic Pitch

Project: <https://github.com/spotify/basic-pitch>

License: Apache License 2.0
<https://github.com/spotify/basic-pitch/blob/main/LICENSE>

Copyright 2022 Spotify AB.

Basic Pitch was used indirectly through NeuralNote as the offline transcription
teacher. Its implementation and model files are not distributed in this
repository.

## Dataset notices and model provenance

No raw dataset material is distributed in this repository. The current preview
weights nevertheless have the following training provenance. See `CREDITS.md`
for complete citations and creator lists.

### GuitarSet

Source: <https://zenodo.org/records/3371780>

DOI: <https://doi.org/10.5281/zenodo.3371780>

License: Creative Commons Attribution 4.0 International
<https://creativecommons.org/licenses/by/4.0/>

Creators: Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye, and Juan
P. Bello.

### Guitar-TECHS

Source: <https://zenodo.org/records/14963133>

Project site: <https://guitar-techs.github.io/>

License: Creative Commons Attribution 4.0 International
<https://creativecommons.org/licenses/by/4.0/>

Dataset-record creators: Hegel Emmanuel Pedroza Villalobos, Termeh Taheri,
Wallace Abreu, Ryan Corey, and Iran R. Roman.

### GOAT

Source: <https://zenodo.org/records/15690894>

DOI: <https://doi.org/10.5281/zenodo.15690894>

Official repository: <https://github.com/JackJamesLoth/GOAT-Dataset>

License: Creative Commons Attribution-NonCommercial 4.0 International
<https://creativecommons.org/licenses/by-nc/4.0/>

Authors: Jackson Loth, Pedro Sarmento, Saurjya Sarkar, Zixun Guo, Mathieu
Barthet, and Mark Sandler.

The record is restricted-access. Current `strict-cap16-v3` weights were trained
using GOAT-derived teacher examples. Pending separate written permission from
the GOAT rightsholders, treat those weights as non-commercial research output;
do not distribute them under AGPL-3.0-only or call them unrestricted open
weights. Never redistribute GOAT audio, annotations, features, teacher caches,
or the gated archive through this repository.

## Python dependencies not bundled in plug-in binaries

The training tools install their own resolved dependency versions. Their
upstream licenses remain controlling:

| Package | Role | Upstream license |
| --- | --- | --- |
| [NumPy](https://numpy.org/) | array processing | BSD 3-Clause |
| [SciPy](https://scipy.org/) | signal/scientific utilities | BSD 3-Clause |
| [python-soundfile](https://github.com/bastibe/python-soundfile) | audio I/O | BSD 3-Clause; its libsndfile dependency is LGPL-2.1-or-later |
| [Mido](https://github.com/mido/mido) | MIDI file I/O | MIT |

These packages are fetched by the user or CI and are not vendored in this
repository. Binary redistributors must review notices for the exact resolved
versions and any native libraries included in their package.

## NVIDIA CUDA

CUDA is an optional trainer build dependency and is not open-source project
code. NVIDIA's CUDA Toolkit and redistributable runtime components are governed
by NVIDIA's own license terms:
<https://docs.nvidia.com/cuda/eula/index.html>. Do not copy CUDA Toolkit files
into source or release archives except where NVIDIA explicitly permits their
redistribution.

## No endorsement

All names are used for factual attribution only. No third-party author,
institution, dataset, or software project endorses or is affiliated with TM
Guitar MIDI.
