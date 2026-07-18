# Building

## Requirements

- Git;
- CMake 3.24 or newer;
- Ninja 1.10 or newer;
- a C++20 compiler;
- internet access during the first configure, or a local JUCE 8.0.13 checkout;
- the repository's versioned `models/strict-cap16-v3` directory.

CUDA is optional and is used only for native training. Plugin inference is CPU
only.

## Configure options

- `TMGM_BUILD_VST3`: build VST3 (enabled on Windows and macOS);
- `TMGM_BUILD_LV2`: build LV2 (enabled on Linux);
- `TMGM_BUILD_TESTS`: build native/engine tests;
- `TMGM_ENABLE_CUDA`: build the optional native CUDA trainer;
- `TMGM_JUCE_SOURCE_DIR`: use an existing JUCE checkout instead of fetching;
- `TMGM_MODEL_PACKAGE`: override the inference package directory.

## Windows VST3

From a Visual Studio developer shell:

```powershell
cmake --preset windows-vst3
cmake --build --preset windows-vst3
ctest --preset windows-vst3
```

Install the complete `.vst3` directory into:

```text
C:\Program Files\Common Files\VST3
```

## macOS VST3

```bash
cmake --preset macos-vst3
cmake --build --preset macos-vst3
ctest --preset macos-vst3
```

Install into `~/Library/Audio/Plug-Ins/VST3` or
`/Library/Audio/Plug-Ins/VST3`. CI preview artifacts are unsigned; signing and
notarisation are intentionally deferred until the release workflow is given
Apple credentials.

## Linux LV2

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git ninja-build libasound2-dev \
  libfreetype6-dev libfontconfig1-dev libx11-dev libxcomposite-dev \
  libxcursor-dev libxext-dev libxinerama-dev libxrandr-dev libxrender-dev \
  libwebkit2gtk-4.1-dev libglu1-mesa-dev mesa-common-dev

cmake --preset linux-lv2
cmake --build --preset linux-lv2
ctest --preset linux-lv2
```

Copy the complete `.lv2` directory to `~/.lv2` or `/usr/lib/lv2`.
The current LV2 descriptor exposes one mono audio input/output pair plus MIDI
output. VST3 hosts can negotiate either mono or stereo audio buses.

## Offline/dependency-controlled configure

Clone JUCE at the pinned revision and pass it explicitly:

```bash
git clone --branch 8.0.13 --depth 1 https://github.com/juce-framework/JUCE.git third_party/JUCE
cmake --preset linux-lv2 -DTMGM_JUCE_SOURCE_DIR="$PWD/third_party/JUCE"
```

No package manager is used by the realtime engine itself.

The committed `CMakePresets.json` always selects the Ninja generator. Put
machine-specific paths such as a local JUCE checkout in the ignored
`CMakeUserPresets.json`, or pass them as `-D` overrides as shown above.
