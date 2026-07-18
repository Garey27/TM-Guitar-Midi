#pragma once

#include "CleanroomAcousticFrontend.hpp"
#include "tmgm/strict_cap16_v3.hpp"
#include "tmgm/strict_cap16_v3_frontend.hpp"
#include "tmgm/causal_resampler.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace tmgm::preview {

inline constexpr double kModelSampleRate = 22050.0;
inline constexpr std::size_t kPitchCount = 49U;
inline constexpr int kMidiMin = 40;
inline constexpr int kMidiMax = 88;

struct MidiEvent {
    enum class Kind : std::uint8_t { note_on, note_off };
    Kind kind = Kind::note_off;
    std::uint8_t pitch = 0U;
    std::uint8_t velocity = 0U;
    std::uint32_t sample_offset = 0U;

    friend bool operator==(const MidiEvent&, const MidiEvent&) = default;
};

using MidiEventCallback = void (*)(void*, const MidiEvent&) noexcept;

enum class EngineProcessStatus : std::uint8_t {
    success = 0U,
    not_loaded,
    not_prepared,
    unsupported_sample_rate,
    frontend_failure,
    inference_failure,
};

// Native, JUCE-independent realtime engine.  File authentication and all heap
// work are performed by load/prepare. process_block is noexcept and owns no
// TCN or legacy-model fallback path.
class TmRealtimeEngine {
public:
    TmRealtimeEngine() = default;
    ~TmRealtimeEngine() = default;

    TmRealtimeEngine(const TmRealtimeEngine&) = delete;
    TmRealtimeEngine& operator=(const TmRealtimeEngine&) = delete;

    [[nodiscard]] bool load_package(
        const std::filesystem::path& package_root,
        std::string& error) noexcept;
    [[nodiscard]] bool prepare(
        double host_sample_rate,
        std::size_t maximum_input_block_size,
        std::string& error) noexcept;
    void reset() noexcept;

    [[nodiscard]] EngineProcessStatus process_block(
        const float* mono,
        std::size_t sample_count,
        float linear_gain,
        MidiEventCallback callback,
        void* callback_user) noexcept;

    std::size_t release_active_notes(
        std::array<MidiEvent, kPitchCount>& destination,
        std::uint32_t sample_offset = 0U) noexcept;

    [[nodiscard]] bool package_loaded() const noexcept { return package_loaded_; }
    [[nodiscard]] bool prepared() const noexcept { return prepared_; }
    [[nodiscard]] std::uint32_t latency_samples() const noexcept {
        return resampler_
            ? native::kResamplerDelayInputSamples
            : 0U;
    }
    [[nodiscard]] float input_peak() const noexcept {
        return input_peak_.load(std::memory_order_relaxed);
    }

private:
    struct PitchState {
        bool active = false;
        bool onset_above = false;
        int attack_count = 0;
        int release_count = 0;
        int refractory = 0;
        float pending_attack = 0.0F;
        int pending_attack_age = 0;
        int pending_onset_age = 1'000'000;
    };

    struct FrameCallbackContext {
        TmRealtimeEngine* engine = nullptr;
        MidiEventCallback callback = nullptr;
        void* callback_user = nullptr;
        std::uint32_t host_offset = 0U;
        EngineProcessStatus status = EngineProcessStatus::success;
    };

    static void frontend_frame_callback(
        void* user,
        std::uint64_t frame_index,
        const native::StrictCap16V3FrameInput& rows) noexcept;

    void process_decoder_frame(
        const std::array<std::uint8_t, kPitchCount>& activity,
        const std::array<std::uint8_t, kPitchCount>& onset,
        const std::array<float, kPitchCount>& attack,
        std::uint32_t host_offset,
        MidiEventCallback callback,
        void* callback_user) noexcept;
    static std::uint8_t velocity(float attack_energy) noexcept;
    static void emit(
        MidiEventCallback callback,
        void* user,
        MidiEvent::Kind kind,
        int pitch,
        int velocity,
        std::uint32_t offset) noexcept;

    std::optional<native::StrictCap16V3Coordinator> coordinator_;
    std::optional<native::StrictCap16V3StreamingFrontend> frontend_;
    std::optional<native::CausalMonoResampler22050> resampler_;
    CleanroomAcousticFrontend acoustic_;
    CleanroomAcousticFrame acoustic_frame_{};
    bool acoustic_frame_ready_ = false;
    std::array<PitchState, kPitchCount> states_{};

    std::array<float, kPitchCount> normalized_activity_{};
    std::array<std::int32_t, kPitchCount> quantized_activity_{};
    std::array<std::uint8_t, kPitchCount> activity_predictions_{};
    std::array<float, kPitchCount> normalized_onset_{};
    std::array<std::int32_t, kPitchCount> quantized_onset_{};
    std::array<std::uint8_t, kPitchCount> onset_predictions_{};
    std::array<float, kPitchCount> acoustic_attack_{};

    float global_attack_ = 0.0F;
    int global_attack_age_ = 4;
    bool package_loaded_ = false;
    bool prepared_ = false;
    std::atomic<float> input_peak_{0.0F};
};

}  // namespace tmgm::preview
