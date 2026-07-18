#include "tmgm/causal_resampler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tmgm::native {
namespace {

constexpr std::uint32_t kRate44100 = 44100U;
constexpr std::uint32_t kRate48000 = 48000U;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kKaiserBeta = 8.6;
constexpr double kPassbandEdgeHz = 10000.0;
constexpr double kStopbandEdgeHz = 11025.0;
constexpr double kCutoffHz =
    (kPassbandEdgeHz + kStopbandEdgeHz) * 0.5;

struct Ratio {
    std::uint32_t numerator;
    std::uint32_t denominator;
};

[[nodiscard]] Ratio ratio_for(const std::uint32_t input_rate) {
    if (input_rate == kRate44100) {
        return {1U, 2U};
    }
    if (input_rate == kRate48000) {
        return {147U, 320U};
    }
    throw std::invalid_argument(
        "CausalMonoResampler22050 supports only 44100 and 48000 Hz input");
}

[[nodiscard]] double bessel_i0(const double value) noexcept {
    const double quarter_square = value * value * 0.25;
    double term = 1.0;
    double sum = 1.0;
    for (std::uint32_t order = 1U; order <= 64U; ++order) {
        const double divisor = static_cast<double>(order) * order;
        term *= quarter_square / divisor;
        sum += term;
        if (term <= sum * 1.0e-17) {
            break;
        }
    }
    return sum;
}

[[nodiscard]] std::vector<float> build_kernels(
    const std::uint32_t input_rate,
    const std::uint32_t phase_count) {
    constexpr std::size_t taps = kResamplerFirTapCount;
    constexpr double delay = kResamplerDelayInputSamples;
    const double normalized_cutoff = kCutoffHz / input_rate;
    const double inverse_i0_beta = 1.0 / bessel_i0(kKaiserBeta);
    std::array<double, taps> window{};
    for (std::size_t tap = 0U; tap < taps; ++tap) {
        const double position =
            (static_cast<double>(tap) - delay) / delay;
        const double inside = std::max(0.0, 1.0 - position * position);
        window[tap] = bessel_i0(kKaiserBeta * std::sqrt(inside)) *
            inverse_i0_beta;
    }

    std::vector<float> kernels(
        static_cast<std::size_t>(phase_count) * taps, 0.0F);
    std::array<double, taps> temporary{};
    for (std::uint32_t phase = 0U; phase < phase_count; ++phase) {
        const double fraction =
            static_cast<double>(phase) / phase_count;
        double sum = 0.0;
        for (std::size_t tap = 0U; tap < taps; ++tap) {
            const double distance = delay - fraction -
                static_cast<double>(tap);
            const double ideal = std::abs(distance) < 1.0e-14
                ? 2.0 * normalized_cutoff
                : std::sin(2.0 * kPi * normalized_cutoff * distance) /
                    (kPi * distance);
            temporary[tap] = ideal * window[tap];
            sum += temporary[tap];
        }
        if (!std::isfinite(sum) || std::abs(sum) < 1.0e-12) {
            throw std::runtime_error("resampler kernel normalization failed");
        }
        auto* destination = kernels.data() +
            static_cast<std::size_t>(phase) * taps;
        double float_sum = 0.0;
        for (std::size_t tap = 0U; tap < taps; ++tap) {
            destination[tap] = static_cast<float>(temporary[tap] / sum);
            float_sum += static_cast<double>(destination[tap]);
        }
        const auto correction_index = static_cast<std::size_t>(
            kResamplerDelayInputSamples);
        destination[correction_index] = static_cast<float>(
            static_cast<double>(destination[correction_index]) +
            (1.0 - float_sum));
    }
    return kernels;
}

[[nodiscard]] float finite_float(const double value) noexcept {
    constexpr double maximum =
        static_cast<double>(std::numeric_limits<float>::max());
    if (!std::isfinite(value)) {
        return value < 0.0
            ? -std::numeric_limits<float>::max()
            : (value > 0.0 ? std::numeric_limits<float>::max() : 0.0F);
    }
    if (value > maximum) {
        return std::numeric_limits<float>::max();
    }
    if (value < -maximum) {
        return -std::numeric_limits<float>::max();
    }
    return static_cast<float>(value);
}

}  // namespace

struct CausalMonoResampler22050::Impl {
    std::uint32_t input_rate;
    std::uint32_t ratio_numerator;
    std::uint32_t ratio_denominator;
    std::size_t maximum_block;
    std::vector<float> kernels;
    std::vector<float> output_buffer;
    std::array<float, kResamplerFirTapCount> history{};
    std::size_t write_index = 0U;
    std::uint64_t consumed = 0U;
    std::uint64_t produced = 0U;
    std::uint64_t next_output_input_index = 0U;
    std::uint32_t next_output_phase = 0U;

    Impl(
        const std::uint32_t prepared_input_rate,
        const std::size_t maximum_input_block_size)
        : input_rate(prepared_input_rate),
          ratio_numerator(ratio_for(prepared_input_rate).numerator),
          ratio_denominator(ratio_for(prepared_input_rate).denominator),
          maximum_block(maximum_input_block_size),
          kernels(build_kernels(prepared_input_rate, ratio_numerator)),
          output_buffer(maximum_input_block_size, 0.0F) {
        reset();
    }

    void reset() noexcept {
        history.fill(0.0F);
        std::fill(output_buffer.begin(), output_buffer.end(), 0.0F);
        write_index = 0U;
        consumed = 0U;
        produced = 0U;
        next_output_input_index = 0U;
        next_output_phase = 0U;
    }

    [[nodiscard]] float read_delay(const std::size_t delay) const noexcept {
        auto index = write_index + kResamplerFirTapCount - 1U - delay;
        if (index >= kResamplerFirTapCount) {
            index -= kResamplerFirTapCount;
        }
        return history[index];
    }

    [[nodiscard]] float convolve(const std::uint32_t phase) const noexcept {
        const auto* coefficients = kernels.data() +
            static_cast<std::size_t>(phase) * kResamplerFirTapCount;
        double sum = 0.0;
        for (std::size_t tap = 0U;
             tap < kResamplerFirTapCount; ++tap) {
            sum += static_cast<double>(coefficients[tap]) *
                static_cast<double>(read_delay(tap));
        }
        return finite_float(sum);
    }

    void advance_output_clock() noexcept {
        const auto phase_total =
            next_output_phase + ratio_denominator;
        next_output_input_index += phase_total / ratio_numerator;
        next_output_phase = phase_total % ratio_numerator;
    }
};

CausalMonoResampler22050::CausalMonoResampler22050(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

CausalMonoResampler22050::~CausalMonoResampler22050() = default;
CausalMonoResampler22050::CausalMonoResampler22050(
    CausalMonoResampler22050&&) noexcept = default;
CausalMonoResampler22050& CausalMonoResampler22050::operator=(
    CausalMonoResampler22050&&) noexcept = default;

bool CausalMonoResampler22050::supports_input_sample_rate(
    const std::uint32_t input_sample_rate) noexcept {
    return input_sample_rate == kRate44100 || input_sample_rate == kRate48000;
}

CausalMonoResampler22050 CausalMonoResampler22050::prepare(
    const std::uint32_t input_sample_rate,
    const std::size_t maximum_input_block_size) {
    if (maximum_input_block_size == 0U) {
        throw std::invalid_argument("maximum input block size must be positive");
    }
    if (!supports_input_sample_rate(input_sample_rate)) {
        static_cast<void>(ratio_for(input_sample_rate));
    }
    return CausalMonoResampler22050(std::make_unique<Impl>(
        input_sample_rate, maximum_input_block_size));
}

void CausalMonoResampler22050::reset() noexcept {
    if (impl_) {
        impl_->reset();
    }
}

CausalResamplerStatus CausalMonoResampler22050::process_block(
    const float* input_samples,
    const std::size_t input_sample_count,
    ResampledMonoBlock& output) noexcept {
    output = {};
    if (!impl_) {
        return CausalResamplerStatus::unprepared;
    }
    if (input_sample_count != 0U && input_samples == nullptr) {
        return CausalResamplerStatus::null_input;
    }
    if (input_sample_count > impl_->maximum_block) {
        return CausalResamplerStatus::input_block_too_large;
    }
    try {
        std::size_t output_count = 0U;
        for (std::size_t offset = 0U; offset < input_sample_count; ++offset) {
            const float source = input_samples[offset];
            impl_->history[impl_->write_index] =
                std::isfinite(source) ? source : 0.0F;
            impl_->write_index =
                (impl_->write_index + 1U) % kResamplerFirTapCount;
            const auto input_index = impl_->consumed;
            ++impl_->consumed;
            if (input_index == impl_->next_output_input_index) {
                impl_->output_buffer[output_count] =
                    impl_->convolve(impl_->next_output_phase);
                ++output_count;
                ++impl_->produced;
                impl_->advance_output_clock();
            }
        }
        output.samples = impl_->output_buffer.data();
        output.sample_count = output_count;
        return CausalResamplerStatus::success;
    } catch (...) {
        output = {};
        return CausalResamplerStatus::processing_failure;
    }
}

std::uint32_t CausalMonoResampler22050::input_sample_rate() const noexcept {
    return impl_ ? impl_->input_rate : 0U;
}

std::size_t CausalMonoResampler22050::maximum_input_block_size() const noexcept {
    return impl_ ? impl_->maximum_block : 0U;
}

std::uint64_t CausalMonoResampler22050::consumed_input_sample_count() const noexcept {
    return impl_ ? impl_->consumed : 0U;
}

std::uint64_t CausalMonoResampler22050::produced_output_sample_count() const noexcept {
    return impl_ ? impl_->produced : 0U;
}

double CausalMonoResampler22050::latency_output_samples() const noexcept {
    if (!impl_) {
        return 0.0;
    }
    return static_cast<double>(kResamplerDelayInputSamples) *
        kResamplerOutputSampleRate / impl_->input_rate;
}

}  // namespace tmgm::native
