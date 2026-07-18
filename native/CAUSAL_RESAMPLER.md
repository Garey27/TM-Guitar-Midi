# Causal host-rate resampler

`CausalMonoResampler22050` is the allocation-free host-rate seam before
`StrictCap16V3StreamingFrontend`. It accepts mono float32 blocks at exactly
44100 or 48000 Hz and emits instance-owned mono float32 blocks at 22050 Hz.

## Contract

- `prepare(input_rate, maximum_input_block_size)` generates all polyphase
  kernels and allocates output storage outside the audio callback.
- `process_block(...)` and `reset()` are `noexcept` and allocate no memory.
- From reset, cumulative input count `N` produces exactly
  `ceil(N * 22050 / input_rate)` samples, independently of host partitions.
- Output sample zero is scheduled at input sample zero. The causal linear-phase
  FIR contributes 128 input samples of signal delay: 64 output samples at
  44100 Hz or 58.8 output samples at 48000 Hz.
- The 48000-Hz clock is the exact rational ratio `147/320`; no floating-point
  time accumulator is used.
- Returned output storage is borrowed and must be consumed before the next
  `process_block` or `reset` call.
- Non-finite input is treated as silence. Finite audio is not clipped; only a
  value outside finite float representability is saturated.

## Filter

The resampler uses 257-tap, 8.6-beta Kaiser-windowed sinc phase kernels. The
passband edge is 10 kHz, the stopband edge is 11.025 kHz, and the design cutoff
is their midpoint, 10.5125 kHz. There is one phase for 44100 Hz and 147 phases
for 48000 Hz. Kernel generation and normalization happen only in `prepare`.

## Verification

`causal_resampler_test.cpp` checks both input rates with contiguous, irregular,
and one-sample host blocks. It verifies exact partition/reset equality,
count/phase and impulse delay, DC gain, 1/10-kHz passband gain, 12-kHz alias
suppression, non-finite safety, and zero allocations around every realtime
call.

The current Release reference run reports approximately:

| Input rate | 1 kHz gain | 10 kHz gain | 12 kHz stopband |
| --- | ---: | ---: | ---: |
| 44100 Hz | 0.999999 | 1.00004 | -97.75 dB |
| 48000 Hz | 1.000000 | 0.999948 | -109.88 dB |

Integration order is:

`host mono block -> CausalMonoResampler22050 -> StrictCap16V3StreamingFrontend -> StrictCap16V3Coordinator`.
