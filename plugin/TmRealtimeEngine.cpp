#include "TmRealtimeEngine.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>

namespace tmgm::preview {
namespace {

constexpr float kVelocityFloor = 0.0015F;
constexpr float kVelocityReference = 0.16F;
constexpr float kNewNoteAttackFloor = 0.0020F;
constexpr float kRetriggerAttackFloor = 0.0025F;
constexpr float kGlobalAttackFloor = 0.015F;
constexpr int kAttackFrames = 2;
constexpr int kReleaseFrames = 4;
constexpr int kRetriggerRefractoryFrames = 6;
constexpr int kAttackMemoryFrames = 12;
constexpr int kGlobalAttackMemoryFrames = 3;

}  // namespace

bool TmRealtimeEngine::load_package(
    const std::filesystem::path& package_root,
    std::string& error) noexcept {
    package_loaded_ = false;
    prepared_ = false;
    coordinator_.reset();
    frontend_.reset();
    try {
        auto coordinator = native::StrictCap16V3Coordinator::load(package_root);
        auto frontend = native::StrictCap16V3StreamingFrontend::load(
            package_root / "native-frontend" / "strict-cap16-v3.tmgmfront");
        coordinator_.emplace(std::move(coordinator));
        frontend_.emplace(std::move(frontend));
        package_loaded_ = true;
        error.clear();
        reset();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
    } catch (...) {
        error = "unknown strict-cap16-v3 package loading failure";
    }
    coordinator_.reset();
    frontend_.reset();
    return false;
}

bool TmRealtimeEngine::prepare(
    const double host_sample_rate,
    const std::size_t maximum_input_block_size,
    std::string& error) noexcept {
    prepared_ = false;
    if (!package_loaded_ || !coordinator_ || !frontend_) {
        error = "authenticated strict-cap16-v3 package is not loaded";
        return false;
    }
    const auto rounded_rate = static_cast<std::uint32_t>(
        std::llround(host_sample_rate));
    if (std::abs(host_sample_rate - static_cast<double>(rounded_rate)) >= 0.5
        || !native::CausalMonoResampler22050::supports_input_sample_rate(
            rounded_rate)) {
        error = "preview supports only 44100 Hz and 48000 Hz host sample rates";
        return false;
    }
    try {
        resampler_.emplace(native::CausalMonoResampler22050::prepare(
            rounded_rate, std::max<std::size_t>(maximum_input_block_size, 1U)));
    } catch (const std::exception& exception) {
        error = exception.what();
        resampler_.reset();
        return false;
    }
    reset();
    prepared_ = true;
    error.clear();
    return true;
}

void TmRealtimeEngine::reset() noexcept {
    if (resampler_) {
        resampler_->reset();
    }
    acoustic_.reset();
    acoustic_frame_ = CleanroomAcousticFrame{};
    acoustic_frame_ready_ = false;
    if (frontend_) {
        frontend_->reset();
    }
    states_.fill(PitchState{});
    global_attack_ = 0.0F;
    global_attack_age_ = kGlobalAttackMemoryFrames + 1;
    input_peak_.store(0.0F, std::memory_order_relaxed);
}

EngineProcessStatus TmRealtimeEngine::process_block(
    const float* const mono,
    const std::size_t sample_count,
    const float linear_gain,
    const MidiEventCallback callback,
    void* const callback_user) noexcept {
    if (!package_loaded_) {
        return EngineProcessStatus::not_loaded;
    }
    if (!prepared_) {
        return EngineProcessStatus::not_prepared;
    }
    if (mono == nullptr && sample_count != 0U) {
        return EngineProcessStatus::frontend_failure;
    }

    float peak = 0.0F;
    for (std::size_t index = 0U; index < sample_count; ++index) {
        const float source = std::isfinite(mono[index]) ? mono[index] : 0.0F;
        peak = std::max(peak, std::abs(source * linear_gain));
    }
    input_peak_.store(peak, std::memory_order_relaxed);

    FrameCallbackContext context{
        this, callback, callback_user, 0U, EngineProcessStatus::success};
    // One host sample per call preserves an exact JUCE sample offset for every
    // model frame, including at arbitrary host block boundaries. The native
    // rational clock and FIR state are still shared and allocation-free.
    for (std::size_t host_offset = 0U; host_offset < sample_count; ++host_offset) {
        const float source = std::isfinite(mono[host_offset])
            ? mono[host_offset]
            : 0.0F;
        native::ResampledMonoBlock output;
        if (resampler_->process_block(&source, 1U, output)
            != native::CausalResamplerStatus::success) {
            context.status = EngineProcessStatus::frontend_failure;
            break;
        }
        context.host_offset = static_cast<std::uint32_t>(host_offset);
        for (std::size_t output_index = 0U;
             output_index < output.sample_count;
             ++output_index) {
            acoustic_frame_ready_ = acoustic_.push_sample(
                output.samples[output_index] * linear_gain,
                acoustic_frame_);
            const auto frontend_status = frontend_->process_block(
                output.samples + output_index,
                1U,
                &TmRealtimeEngine::frontend_frame_callback,
                &context);
            if (frontend_status != native::StrictCap16V3FrontendStatus::success) {
                context.status = EngineProcessStatus::frontend_failure;
                break;
            }
        }
        if (context.status != EngineProcessStatus::success) {
            break;
        }
    }
    return context.status;
}

void TmRealtimeEngine::frontend_frame_callback(
    void* const user,
    const std::uint64_t frame_index,
    const native::StrictCap16V3FrameInput& rows) noexcept {
    auto& context = *static_cast<FrameCallbackContext*>(user);
    auto& engine = *context.engine;
    if (!engine.acoustic_frame_ready_
        || engine.acoustic_frame_.frame_index != frame_index) {
        context.status = EngineProcessStatus::frontend_failure;
        return;
    }
    native::StrictCap16V3FrameOutputBuffers outputs{
        engine.normalized_activity_.data(), engine.normalized_activity_.size(),
        engine.quantized_activity_.data(), engine.quantized_activity_.size(),
        engine.activity_predictions_.data(), engine.activity_predictions_.size(),
        engine.normalized_onset_.data(), engine.normalized_onset_.size(),
        engine.quantized_onset_.data(), engine.quantized_onset_.size(),
        engine.onset_predictions_.data(), engine.onset_predictions_.size()};
    if (engine.coordinator_->predict_frame(rows, outputs)
        != native::StrictCap16V3PredictStatus::success) {
        context.status = EngineProcessStatus::inference_failure;
        return;
    }
    engine.acoustic_attack_ = engine.acoustic_frame_.attack_energy;
    engine.acoustic_frame_ready_ = false;
    engine.process_decoder_frame(
        engine.activity_predictions_,
        engine.onset_predictions_,
        engine.acoustic_attack_,
        context.host_offset,
        context.callback,
        context.callback_user);
}

void TmRealtimeEngine::process_decoder_frame(
    const std::array<std::uint8_t, kPitchCount>& activity,
    const std::array<std::uint8_t, kPitchCount>& onset,
    const std::array<float, kPitchCount>& attack,
    const std::uint32_t host_offset,
    const MidiEventCallback callback,
    void* const callback_user) noexcept {
    const float frame_global_attack =
        *std::max_element(attack.begin(), attack.end());
    if (frame_global_attack >= global_attack_) {
        global_attack_ = frame_global_attack;
        global_attack_age_ = 0;
    } else {
        global_attack_age_ = std::min(global_attack_age_ + 1, 1'000'000);
        if (global_attack_age_ > kGlobalAttackMemoryFrames) {
            global_attack_ = 0.0F;
        }
    }
    const bool global_attack_fresh =
        global_attack_age_ <= kGlobalAttackMemoryFrames
        && global_attack_ >= kGlobalAttackFloor;

    for (std::size_t note = 0U; note < kPitchCount; ++note) {
        auto& state = states_[note];
        if (attack[note] > state.pending_attack) {
            state.pending_attack = attack[note];
            state.pending_attack_age = 0;
        } else {
            state.pending_attack_age = std::min(
                state.pending_attack_age + 1, 1'000'000);
            if (state.pending_attack_age > kAttackMemoryFrames) {
                state.pending_attack = 0.0F;
            }
        }

        const bool onset_above = onset[note] != 0U;
        const bool rising_onset = onset_above && !state.onset_above;
        state.onset_above = onset_above;
        if (rising_onset) {
            state.pending_onset_age = 0;
        } else {
            state.pending_onset_age = std::min(
                state.pending_onset_age + 1, 1'000'000);
        }
        if (state.refractory > 0) {
            --state.refractory;
        }

        if (!state.active) {
            if (activity[note] != 0U) {
                state.attack_count = std::min(state.attack_count + 1, kAttackFrames);
            } else {
                state.attack_count = 0;
            }
            const bool attack_fresh =
                state.pending_attack_age <= kAttackMemoryFrames;
            const bool onset_fresh =
                state.pending_onset_age <= kAttackMemoryFrames;
            if (state.attack_count >= kAttackFrames
                && ((onset_fresh && global_attack_fresh)
                    || (attack_fresh
                        && state.pending_attack >= kNewNoteAttackFloor))) {
                const float event_attack = std::max(
                    state.pending_attack, 0.5F * global_attack_);
                emit(callback, callback_user, MidiEvent::Kind::note_on,
                    kMidiMin + static_cast<int>(note), velocity(event_attack),
                    host_offset);
                state.active = true;
                state.attack_count = 0;
                state.release_count = 0;
                state.refractory = kRetriggerRefractoryFrames;
                state.pending_attack = 0.0F;
                state.pending_attack_age = 0;
                state.pending_onset_age = 1'000'000;
            }
            continue;
        }

        const bool attack_fresh =
            state.pending_attack_age <= kAttackMemoryFrames;
        if (rising_onset && state.refractory == 0
            && ((attack_fresh
                    && state.pending_attack >= kRetriggerAttackFloor)
                || global_attack_fresh)) {
            const int pitch = kMidiMin + static_cast<int>(note);
            emit(callback, callback_user, MidiEvent::Kind::note_off,
                pitch, 0, host_offset);
            const float event_attack = std::max(
                state.pending_attack, 0.5F * global_attack_);
            emit(callback, callback_user, MidiEvent::Kind::note_on,
                pitch, velocity(event_attack), host_offset);
            state.refractory = kRetriggerRefractoryFrames;
            state.pending_attack = 0.0F;
            state.pending_attack_age = 0;
            state.pending_onset_age = 1'000'000;
            state.release_count = 0;
        } else if (activity[note] == 0U) {
            if (++state.release_count >= kReleaseFrames) {
                emit(callback, callback_user, MidiEvent::Kind::note_off,
                    kMidiMin + static_cast<int>(note), 0, host_offset);
                state.active = false;
                state.attack_count = 0;
                state.release_count = 0;
                state.refractory = 0;
                state.pending_onset_age = 1'000'000;
            }
        } else {
            state.release_count = 0;
        }
    }
}

std::uint8_t TmRealtimeEngine::velocity(const float attack_energy) noexcept {
    const float span = kVelocityReference - kVelocityFloor;
    const float unit = 1.0F - std::exp(
        -std::max(attack_energy - kVelocityFloor, 0.0F) / span);
    const auto value = static_cast<int>(std::nearbyint(
        1.0F + 126.0F * std::sqrt(std::max(unit, 0.0F))));
    return static_cast<std::uint8_t>(std::clamp(value, 1, 127));
}

void TmRealtimeEngine::emit(
    const MidiEventCallback callback,
    void* const user,
    const MidiEvent::Kind kind,
    const int pitch,
    const int velocity_value,
    const std::uint32_t offset) noexcept {
    if (callback == nullptr) {
        return;
    }
    callback(user, MidiEvent{
        kind,
        static_cast<std::uint8_t>(pitch),
        static_cast<std::uint8_t>(velocity_value),
        offset});
}

std::size_t TmRealtimeEngine::release_active_notes(
    std::array<MidiEvent, kPitchCount>& destination,
    const std::uint32_t sample_offset) noexcept {
    std::size_t count = 0U;
    for (std::size_t note = 0U; note < kPitchCount; ++note) {
        if (states_[note].active) {
            destination[count++] = MidiEvent{
                MidiEvent::Kind::note_off,
                static_cast<std::uint8_t>(kMidiMin + static_cast<int>(note)),
                0U,
                sample_offset};
        }
        states_[note] = PitchState{};
    }
    return count;
}

}  // namespace tmgm::preview
