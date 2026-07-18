#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace tmgm::preview {

inline constexpr std::uint32_t kCleanroomAcousticSampleRate = 22050U;
inline constexpr std::uint32_t kCleanroomAcousticHopSize = 256U;
inline constexpr std::size_t kCleanroomAcousticPitchCount = 49U;
inline constexpr int kCleanroomAcousticMidiMin = 40;
inline constexpr int kCleanroomAcousticMidiMax = 88;

// Numeric output of one strictly-causal frame.  This is the native contract
// corresponding to tmgm_rt.tracking.SpectralEvidence.  The TM-only preview
// currently consumes attack_energy, while the remaining fields are preserved
// so the port can be audited directly against the Python source of truth.
struct CleanroomAcousticFrame {
    std::uint64_t frame_index = 0U;
    std::uint64_t sample_index = 0U;
    std::array<float, kCleanroomAcousticPitchCount> activity{};
    std::array<float, kCleanroomAcousticPitchCount> onset{};
    std::array<float, kCleanroomAcousticPitchCount> attack_energy{};
    std::array<float, kCleanroomAcousticPitchCount> fundamental_amplitude{};
};

// Allocation-free streaming port of Python CausalDualResolutionFrontend.
// Construction creates the two PocketFFT plans and all fixed scratch storage;
// construct/prepare the containing engine off the audio callback.  reset() and
// push_sample() do no allocation, I/O, locking, or exception propagation.
//
// Frame zero is emitted after model-rate sample one, then every 256 samples.
// push_sample() returns true exactly when `output` was replaced by a new frame.
class CleanroomAcousticFrontend {
public:
    CleanroomAcousticFrontend();
    ~CleanroomAcousticFrontend();

    CleanroomAcousticFrontend(const CleanroomAcousticFrontend&) = delete;
    CleanroomAcousticFrontend& operator=(
        const CleanroomAcousticFrontend&) = delete;
    CleanroomAcousticFrontend(CleanroomAcousticFrontend&&) noexcept;
    CleanroomAcousticFrontend& operator=(
        CleanroomAcousticFrontend&&) noexcept;

    void reset() noexcept;

    [[nodiscard]] bool push_sample(
        float model_rate_sample,
        CleanroomAcousticFrame& output) noexcept;

    [[nodiscard]] std::uint64_t consumed_sample_count() const noexcept;
    [[nodiscard]] std::uint64_t emitted_frame_count() const noexcept;
    [[nodiscard]] bool healthy() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tmgm::preview
