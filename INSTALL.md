# Installing TM Guitar MIDI

Every platform archive is self-contained. The model weights are already inside
the plug-in bundle, so there is no separate model download or copy step.

Keep the bundle intact. In particular, do not remove or relocate:

- `TM Guitar MIDI.vst3/Contents/Resources/TMModel` on Windows and macOS;
- `TM Guitar MIDI.lv2/TMModel` on Linux.

## Windows x64 (VST3)

1. Extract the Windows ZIP.
2. Copy the complete `TM Guitar MIDI.vst3` folder to either
   `%LOCALAPPDATA%\Programs\Common\VST3` or
   `C:\Program Files\Common Files\VST3`.
3. Rescan VST3 plug-ins in the DAW.

## macOS Intel / Apple Silicon (VST3)

1. Extract the macOS ZIP.
2. Copy the complete `TM Guitar MIDI.vst3` bundle to
   `~/Library/Audio/Plug-Ins/VST3`.
3. Rescan VST3 plug-ins in the DAW.

## Linux x64 (LV2)

1. Extract the Linux archive.
2. Copy the complete `TM Guitar MIDI.lv2` directory to `~/.lv2`.
3. Rescan LV2 plug-ins in the host.

Put **TM Guitar MIDI** before a MIDI instrument in the effects chain. The
plug-in status should report that the model is loaded.
