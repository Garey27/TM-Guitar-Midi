#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tmgm::native {

inline constexpr std::uint32_t kResamplerOutputSampleRate = 22050U;
inline constexpr std::uint32_t kResamplerFirTapCount = 257U;
inline constexpr std::uint32_t kResamplerDelayInputSamples = 128U;

struct ResampledMonoBlock {
    // Instance-owned storage. Valid until the next process_block/reset call.
    const float* samples = nullptr;
    std::size_t sample_count = 0U;
};

enum class CausalResamplerStatus : std::uint8_t {
    success = 0U,
    unprepared,
    null_input,
    input_block_too_large,
    processing_failure,
};

// Allocation-free streaming mono conversion from a supported host rate to
// 22050 Hz. prepare() generates the polyphase Kaiser-windowed sinc kernels and
// allocates the maximum output block outside the audio callback.
//
// Supported exact rational ratios:
//   44100 -> 22050: 1 / 2
//   48000 -> 22050: 147 / 320
//
// From reset, N cumulative input samples produce exactly
// ceil(N * 22050 / input_sample_rate) output samples. Output sample zero is
// scheduled at input sample zero; the symmetric FIR contributes 128 input
// samples of causal signal delay. Non-finite input samples are treated as
// silence. Finite input is not audio-range clipped; only an unrepresentable
// result is saturated to the finite float range.
class CausalMonoResampler22050 {
public:
    ~CausalMonoResampler22050();

    CausalMonoResampler22050(const CausalMonoResampler22050&) = delete;
    CausalMonoResampler22050& operator=(
        const CausalMonoResampler22050&) = delete;
    CausalMonoResampler22050(CausalMonoResampler22050&&) noexcept;
    CausalMonoResampler22050& operator=(
        CausalMonoResampler22050&&) noexcept;

    [[nodiscard]] static bool supports_input_sample_rate(
        std::uint32_t input_sample_rate) noexcept;

    [[nodiscard]] static CausalMonoResampler22050 prepare(
        std::uint32_t input_sample_rate,
        std::size_t maximum_input_block_size);

    void reset() noexcept;

    [[nodiscard]] CausalResamplerStatus process_block(
        const float* input_samples,
        std::size_t input_sample_count,
        ResampledMonoBlock& output) noexcept;

    [[nodiscard]] std::uint32_t input_sample_rate() const noexcept;
    [[nodiscard]] std::size_t maximum_input_block_size() const noexcept;
    [[nodiscard]] std::uint64_t consumed_input_sample_count() const noexcept;
    [[nodiscard]] std::uint64_t produced_output_sample_count() const noexcept;
    [[nodiscard]] double latency_output_samples() const noexcept;

private:
    struct Impl;
    explicit CausalMonoResampler22050(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tmgm::native
