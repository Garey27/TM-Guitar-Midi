#pragma once

#include "tmgm/strict_cap16_v3.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace tmgm::native {

inline constexpr std::uint32_t kStrictCap16V3FrontendSampleRate = 22050U;
inline constexpr std::uint32_t kStrictCap16V3FrontendHopSize = 256U;
inline constexpr std::uint32_t kStrictCap16V3FrontendFftSize = 2048U;

enum class StrictCap16V3FrontendStatus : std::uint8_t {
    success = 0U,
    unprepared,
    null_audio,
    null_frame_callback,
    processing_failure,
};

// Called synchronously for every generated frame. The four packed rows point
// to frontend-owned storage and remain valid only until the next emitted frame
// or reset/process call. The callback must itself be realtime safe.
using StrictCap16V3FrontendFrameCallback = void (*)(
    void* user,
    std::uint64_t frame_index,
    const StrictCap16V3FrameInput& rows) noexcept;

// Frozen causal frontend from already resampled mono float32 22050-Hz audio to
// the four packed rows consumed by StrictCap16V3Coordinator. Artifact loading,
// authentication, PocketFFT plan creation, and every allocation happen in
// load(). process_block/reset are noexcept and perform no allocation, I/O,
// locking, or exception throwing.
//
// Frame zero is emitted after consuming sample one. Later frames are emitted
// every 256 samples, exactly matching CausalSTFTPlus.next_frame_sample.
class StrictCap16V3StreamingFrontend {
public:
    ~StrictCap16V3StreamingFrontend();

    StrictCap16V3StreamingFrontend(
        const StrictCap16V3StreamingFrontend&) = delete;
    StrictCap16V3StreamingFrontend& operator=(
        const StrictCap16V3StreamingFrontend&) = delete;
    StrictCap16V3StreamingFrontend(
        StrictCap16V3StreamingFrontend&&) noexcept;
    StrictCap16V3StreamingFrontend& operator=(
        StrictCap16V3StreamingFrontend&&) noexcept;

    [[nodiscard]] static StrictCap16V3StreamingFrontend load(
        const std::filesystem::path& artifact_path);

    void reset() noexcept;

    [[nodiscard]] StrictCap16V3FrontendStatus process_block(
        const float* mono_samples,
        std::size_t sample_count,
        StrictCap16V3FrontendFrameCallback callback,
        void* callback_user) noexcept;

    [[nodiscard]] std::uint64_t consumed_sample_count() const noexcept;
    [[nodiscard]] std::uint64_t emitted_frame_count() const noexcept;

private:
    struct Impl;
    explicit StrictCap16V3StreamingFrontend(
        std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tmgm::native
