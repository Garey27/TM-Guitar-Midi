#include "CleanroomAcousticFrontend.hpp"

#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

namespace tmgm::preview {
namespace {

constexpr std::size_t kShortFftSize = 512U;
constexpr std::size_t kLongFftSize = 4096U;
constexpr std::size_t kHarmonicCount = 11U;
constexpr std::size_t kHarmonicValueCount =
    kCleanroomAcousticPitchCount * kHarmonicCount;
constexpr double kPi = 3.141592653589793238462643383279502884;

constexpr float kHarmonicDecay = 1.0F;
constexpr float kOwnershipStrength = 1.05F;
constexpr double kOwnershipToleranceCents = 18.0;
constexpr float kAcousticFloor = 0.0015F;
constexpr float kAcousticReference = 0.10F;
constexpr float kAttackReference = 0.035F;
constexpr float kSlowAttackAlpha = 0.08F;

struct OwnerLink {
    std::uint8_t owner_pitch = 0U;
    std::uint8_t owner_harmonic = 0U;
};

struct OwnerLinkList {
    std::array<OwnerLink, kCleanroomAcousticPitchCount> values{};
    std::uint8_t size = 0U;
};

[[nodiscard]] double midi_to_hz(const int midi) noexcept {
    return 440.0 * std::pow(2.0, (static_cast<double>(midi) - 69.0) / 12.0);
}

template <std::size_t Size>
void build_hann(std::array<float, Size>& destination) noexcept {
    static_assert(Size > 1U);
    for (std::size_t index = 0U; index < Size; ++index) {
        destination[index] = static_cast<float>(
            0.5 - 0.5 * std::cos(
                2.0 * kPi * static_cast<double>(index) /
                static_cast<double>(Size - 1U)));
    }
}

template <std::size_t Size>
[[nodiscard]] float window_scale(
    const std::array<float, Size>& window) noexcept {
    // Both frozen Hann sizes have an exactly representable float32 sum
    // ((N - 1) / 2). Accumulating in float mirrors ndarray.sum(dtype=float32).
    float sum = 0.0F;
    for (const float value : window) {
        sum += value;
    }
    return 2.0F / std::max(sum, 1.0F);
}

[[nodiscard]] float sample_bin(
    const float* values,
    const std::size_t value_count,
    const double position) noexcept {
    const auto maximum = value_count - 1U;
    if (!(position <= static_cast<double>(maximum))) {
        return 0.0F;
    }
    const double clipped = std::max(0.0, position);
    const auto lower = static_cast<std::size_t>(std::floor(clipped));
    const auto upper = std::min(lower + 1U, maximum);
    const double fraction = clipped - static_cast<double>(lower);
    return static_cast<float>(
        static_cast<double>(values[lower]) * (1.0 - fraction) +
        static_cast<double>(values[upper]) * fraction);
}

[[nodiscard]] float compress_acoustic(const float value) noexcept {
    const float positive = std::max(value - kAcousticFloor, 0.0F);
    const float span = kAcousticReference - kAcousticFloor;
    return 1.0F - std::exp(-positive / span);
}

[[nodiscard]] float onset_unit(const float attack) noexcept {
    const float positive = std::max(attack - kAcousticFloor, 0.0F);
    return 1.0F - std::exp(-positive / kAttackReference);
}

}  // namespace

struct CleanroomAcousticFrontend::Impl {
    using ShortPlan = pocketfft::detail::pocketfft_r<double>;
    using LongPlan = pocketfft::detail::pocketfft_r<double>;

    std::unique_ptr<ShortPlan> short_plan;
    std::unique_ptr<LongPlan> long_plan;

    std::array<float, kLongFftSize> ring{};
    std::array<float, kLongFftSize> ordered_long{};
    std::array<float, kShortFftSize> short_window{};
    std::array<float, kLongFftSize> long_window{};
    float short_scale = 1.0F;
    float long_scale = 1.0F;

    std::array<std::complex<double>, kShortFftSize / 2U + 1U>
        short_fft_buffer{};
    std::array<std::complex<double>, kLongFftSize / 2U + 1U>
        long_fft_buffer{};
    std::array<double, kShortFftSize> short_fft_scratch{};
    std::array<double, kLongFftSize> long_fft_scratch{};
    std::array<float, kShortFftSize / 2U + 1U> short_magnitude{};
    std::array<float, kLongFftSize / 2U + 1U> long_magnitude{};

    std::array<double, kCleanroomAcousticPitchCount> frequencies{};
    std::array<double, kHarmonicValueCount> harmonic_frequencies{};
    std::array<double, kHarmonicValueCount> short_bin_positions{};
    std::array<double, kHarmonicValueCount> long_bin_positions{};
    std::array<double, kHarmonicCount> harmonic_weights{};
    std::array<OwnerLinkList, kHarmonicValueCount> owner_links{};

    std::array<float, kHarmonicValueCount> short_raw{};
    std::array<float, kHarmonicValueCount> long_raw{};
    std::array<float, kHarmonicValueCount> short_residual{};
    std::array<float, kHarmonicValueCount> long_residual{};
    std::array<float, kHarmonicValueCount> short_unit{};
    std::array<float, kHarmonicValueCount> long_unit{};
    std::array<float, kCleanroomAcousticPitchCount> tuning_mask{};
    std::array<float, kCleanroomAcousticPitchCount> slow_short{};

    std::size_t write_index = 0U;
    std::uint64_t sample_count = 0U;
    std::uint64_t next_frame_sample = 1U;
    std::uint64_t frame_count = 0U;
    bool failed = false;

    Impl()
        : short_plan(std::make_unique<ShortPlan>(kShortFftSize)),
          long_plan(std::make_unique<LongPlan>(kLongFftSize)) {
        build_hann(short_window);
        build_hann(long_window);
        short_scale = window_scale(short_window);
        long_scale = window_scale(long_window);

        double harmonic_weight_sum = 0.0;
        for (std::size_t harmonic = 0U;
             harmonic < kHarmonicCount; ++harmonic) {
            harmonic_weights[harmonic] =
                1.0 / static_cast<double>(harmonic + 1U);
            harmonic_weight_sum += harmonic_weights[harmonic];
        }
        for (double& weight : harmonic_weights) {
            weight /= harmonic_weight_sum;
        }

        for (std::size_t pitch = 0U;
             pitch < kCleanroomAcousticPitchCount; ++pitch) {
            frequencies[pitch] = midi_to_hz(
                kCleanroomAcousticMidiMin + static_cast<int>(pitch));
            for (std::size_t harmonic = 0U;
                 harmonic < kHarmonicCount; ++harmonic) {
                const auto index = pitch * kHarmonicCount + harmonic;
                harmonic_frequencies[index] = frequencies[pitch] *
                    static_cast<double>(harmonic + 1U);
                short_bin_positions[index] = harmonic_frequencies[index] *
                    static_cast<double>(kShortFftSize) /
                    static_cast<double>(kCleanroomAcousticSampleRate);
                long_bin_positions[index] = harmonic_frequencies[index] *
                    static_cast<double>(kLongFftSize) /
                    static_cast<double>(kCleanroomAcousticSampleRate);
            }
        }
        build_owner_links();
        reset();
    }

    void build_owner_links() noexcept {
        for (std::size_t pitch = 0U;
             pitch < kCleanroomAcousticPitchCount; ++pitch) {
            for (std::size_t harmonic = 0U;
                 harmonic < kHarmonicCount; ++harmonic) {
                const auto target_index = pitch * kHarmonicCount + harmonic;
                auto& list = owner_links[target_index];
                for (std::size_t owner = 0U; owner < pitch; ++owner) {
                    const double ratio =
                        harmonic_frequencies[target_index] /
                        frequencies[owner];
                    const auto owner_harmonic =
                        static_cast<int>(std::nearbyint(ratio));
                    if (owner_harmonic < 2 ||
                        owner_harmonic > static_cast<int>(kHarmonicCount)) {
                        continue;
                    }
                    const double cents = std::abs(
                        1200.0 * std::log2(
                            ratio / static_cast<double>(owner_harmonic)));
                    if (cents <= kOwnershipToleranceCents &&
                        list.size < list.values.size()) {
                        list.values[list.size++] = {
                            static_cast<std::uint8_t>(owner),
                            static_cast<std::uint8_t>(owner_harmonic),
                        };
                    }
                }
            }
        }
    }

    void reset() noexcept {
        ring.fill(0.0F);
        ordered_long.fill(0.0F);
        short_fft_buffer.fill({0.0, 0.0});
        long_fft_buffer.fill({0.0, 0.0});
        short_fft_scratch.fill(0.0);
        long_fft_scratch.fill(0.0);
        short_magnitude.fill(0.0F);
        long_magnitude.fill(0.0F);
        short_raw.fill(0.0F);
        long_raw.fill(0.0F);
        short_residual.fill(0.0F);
        long_residual.fill(0.0F);
        short_unit.fill(0.0F);
        long_unit.fill(0.0F);
        tuning_mask.fill(0.0F);
        slow_short.fill(0.0F);
        write_index = 0U;
        sample_count = 0U;
        next_frame_sample = 1U;
        frame_count = 0U;
        failed = false;
    }

    template <std::size_t FftSize>
    void transform(
        const float* frame,
        const std::array<float, FftSize>& window,
        const float scale,
        pocketfft::detail::pocketfft_r<double>& plan,
        std::array<std::complex<double>, FftSize / 2U + 1U>& buffer,
        std::array<double, FftSize>& scratch,
        std::array<float, FftSize / 2U + 1U>& magnitude) {
        auto* storage = reinterpret_cast<double*>(buffer.data());
        for (std::size_t index = 0U; index < FftSize; ++index) {
            const float windowed = frame[index] * window[index];
            storage[index + 1U] = static_cast<double>(windowed);
        }
        storage[FftSize + 1U] = 0.0;
        plan.exec_with_scratch(
            storage + 1U, scratch.data(), 1.0, pocketfft::FORWARD);
        buffer[0] = buffer[0].imag();
        for (std::size_t index = 0U;
             index < magnitude.size(); ++index) {
            float value = static_cast<float>(std::hypot(
                buffer[index].real(), buffer[index].imag()));
            value *= scale;
            magnitude[index] = value;
        }
    }

    void ordered_frame() noexcept {
        for (std::size_t index = 0U; index < kLongFftSize; ++index) {
            ordered_long[index] =
                ring[(write_index + index) % kLongFftSize];
        }
    }

    template <std::size_t SpectrumSize>
    void harmonic_amplitudes(
        const std::array<float, SpectrumSize>& magnitude,
        const std::array<double, kHarmonicValueCount>& positions,
        std::array<float, kHarmonicValueCount>& destination) noexcept {
        for (std::size_t index = 0U;
             index < kHarmonicValueCount; ++index) {
            destination[index] = sample_bin(
                magnitude.data(), magnitude.size(), positions[index]);
        }
    }

    void reject_harmonic_owners(
        const std::array<float, kHarmonicValueCount>& source,
        std::array<float, kHarmonicValueCount>& destination) noexcept {
        destination = source;
        for (std::size_t pitch = 0U;
             pitch < kCleanroomAcousticPitchCount; ++pitch) {
            for (std::size_t harmonic = 0U;
                 harmonic < kHarmonicCount; ++harmonic) {
                const auto target = pitch * kHarmonicCount + harmonic;
                double explained = 0.0;
                const auto& links = owner_links[target];
                for (std::size_t link_index = 0U;
                     link_index < links.size; ++link_index) {
                    const auto link = links.values[link_index];
                    const auto owner = static_cast<std::size_t>(
                        link.owner_pitch);
                    const auto owner_harmonic = static_cast<std::size_t>(
                        link.owner_harmonic);
                    const double expected =
                        static_cast<double>(source[
                            owner * kHarmonicCount]) /
                        std::pow(
                            static_cast<double>(owner_harmonic),
                            static_cast<double>(kHarmonicDecay)) *
                        static_cast<double>(kOwnershipStrength);
                    const double measured = static_cast<double>(source[
                        owner * kHarmonicCount + owner_harmonic - 1U]);
                    explained += std::min(measured, expected);
                }
                destination[target] = static_cast<float>(std::max(
                    static_cast<double>(source[target]) - explained, 0.0));
            }
        }
    }

    void build_tuning_mask() noexcept {
        for (std::size_t pitch = 0U;
             pitch < kCleanroomAcousticPitchCount; ++pitch) {
            float neighbour = 0.0F;
            if (pitch > 0U) {
                neighbour = std::max(
                    neighbour,
                    long_raw[(pitch - 1U) * kHarmonicCount]);
            }
            if (pitch + 1U < kCleanroomAcousticPitchCount) {
                neighbour = std::max(
                    neighbour,
                    long_raw[(pitch + 1U) * kHarmonicCount]);
            }
            const float fundamental = long_raw[pitch * kHarmonicCount];
            const float ratio = fundamental / std::max(neighbour, 1.0e-8F);
            tuning_mask[pitch] = std::clamp(
                (ratio - 1.02F) / 0.18F, 0.0F, 1.0F);
        }
    }

    [[nodiscard]] double weighted_harmonics(
        const std::array<float, kHarmonicValueCount>& values,
        const std::size_t pitch) const noexcept {
        double sum = 0.0;
        for (std::size_t harmonic = 0U;
             harmonic < kHarmonicCount; ++harmonic) {
            sum += static_cast<double>(
                values[pitch * kHarmonicCount + harmonic]) *
                harmonic_weights[harmonic];
        }
        return sum;
    }

    void emit_frame(CleanroomAcousticFrame& output) {
        ordered_frame();
        transform(
            ordered_long.data(), long_window, long_scale,
            *long_plan, long_fft_buffer, long_fft_scratch, long_magnitude);
        transform(
            ordered_long.data() + (kLongFftSize - kShortFftSize),
            short_window, short_scale,
            *short_plan, short_fft_buffer, short_fft_scratch, short_magnitude);
        harmonic_amplitudes(
            long_magnitude, long_bin_positions, long_raw);
        harmonic_amplitudes(
            short_magnitude, short_bin_positions, short_raw);
        reject_harmonic_owners(long_raw, long_residual);
        reject_harmonic_owners(short_raw, short_residual);
        build_tuning_mask();

        for (std::size_t pitch = 0U;
             pitch < kCleanroomAcousticPitchCount; ++pitch) {
            const auto fundamental = pitch * kHarmonicCount;
            long_residual[fundamental] *= tuning_mask[pitch];
            short_residual[fundamental] *= tuning_mask[pitch];
        }
        for (std::size_t index = 0U;
             index < kHarmonicValueCount; ++index) {
            long_unit[index] = compress_acoustic(long_residual[index]);
            short_unit[index] = compress_acoustic(short_residual[index]);
        }

        output.frame_index = frame_count;
        output.sample_index = sample_count;
        for (std::size_t pitch = 0U;
             pitch < kCleanroomAcousticPitchCount; ++pitch) {
            const auto fundamental = pitch * kHarmonicCount;
            const double long_harmonic = weighted_harmonics(long_unit, pitch);
            const double short_harmonic = weighted_harmonics(short_unit, pitch);
            const double long_pitch = static_cast<double>(
                0.72F * long_unit[fundamental]) + 0.28 * long_harmonic;
            const double short_pitch = static_cast<double>(
                0.72F * short_unit[fundamental]) + 0.28 * short_harmonic;
            output.activity[pitch] = static_cast<float>(std::clamp(
                0.78 * long_pitch + 0.22 * short_pitch, 0.0, 1.0));

            const double short_residual_harmonic =
                weighted_harmonics(short_residual, pitch);
            const float short_acoustic = static_cast<float>(
                static_cast<double>(
                    0.78F * short_residual[fundamental]) +
                0.22 * short_residual_harmonic);
            const float attack = std::max(
                short_acoustic - slow_short[pitch], 0.0F);
            slow_short[pitch] += kSlowAttackAlpha *
                (short_acoustic - slow_short[pitch]);
            output.attack_energy[pitch] = attack;
            output.onset[pitch] = onset_unit(attack);
            output.fundamental_amplitude[pitch] =
                long_residual[fundamental];
        }
        ++frame_count;
    }

    [[nodiscard]] bool push_sample(
        const float sample,
        CleanroomAcousticFrame& output) noexcept {
        if (failed) {
            return false;
        }
        ring[write_index] = sample;
        write_index = (write_index + 1U) % kLongFftSize;
        ++sample_count;
        if (sample_count != next_frame_sample) {
            return false;
        }
        try {
            emit_frame(output);
            next_frame_sample += kCleanroomAcousticHopSize;
            return true;
        } catch (...) {
            failed = true;
            return false;
        }
    }
};

CleanroomAcousticFrontend::CleanroomAcousticFrontend()
    : impl_(std::make_unique<Impl>()) {}

CleanroomAcousticFrontend::~CleanroomAcousticFrontend() = default;
CleanroomAcousticFrontend::CleanroomAcousticFrontend(
    CleanroomAcousticFrontend&&) noexcept = default;
CleanroomAcousticFrontend& CleanroomAcousticFrontend::operator=(
    CleanroomAcousticFrontend&&) noexcept = default;

void CleanroomAcousticFrontend::reset() noexcept {
    if (impl_) {
        impl_->reset();
    }
}

bool CleanroomAcousticFrontend::push_sample(
    const float model_rate_sample,
    CleanroomAcousticFrame& output) noexcept {
    return impl_ && impl_->push_sample(model_rate_sample, output);
}

std::uint64_t CleanroomAcousticFrontend::consumed_sample_count()
    const noexcept {
    return impl_ ? impl_->sample_count : 0U;
}

std::uint64_t CleanroomAcousticFrontend::emitted_frame_count()
    const noexcept {
    return impl_ ? impl_->frame_count : 0U;
}

bool CleanroomAcousticFrontend::healthy() const noexcept {
    return impl_ && !impl_->failed;
}

}  // namespace tmgm::preview
