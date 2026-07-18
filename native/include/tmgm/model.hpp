#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace tmgm::native {

struct NativeDataset;

inline constexpr std::uint32_t kTmModelLegacyFormatVersion = 1U;
inline constexpr std::uint32_t kTmModelPreviousFormatVersion = 2U;
inline constexpr std::uint32_t kTmModelFormatVersion = 3U;
inline constexpr std::uint32_t kTmModelHeaderBytes = 256U;
inline constexpr std::uint32_t kTmModelWordBits = 32U;

enum class TmModelHead : std::uint32_t {
    activity = 1U,
    onset = 2U,
};

// Dimensions also describe the two flat payload layouts:
//   ta_bitplanes[clause][state_bit][literal_word]
//   weights[output][clause]
// A literal word contains 32 literal automata, least-significant bit first.
struct TmModelDimensions {
    std::uint32_t feature_count = 0U;
    std::uint32_t output_count = 0U;
    std::uint32_t clause_count = 0U;
    std::uint32_t state_bits = 8U;
};

struct TmModelTrainingConfig {
    std::int32_t threshold = 0;
    float specificity = 0.0F;
    float negative_samples = 0.0F;
    float type_i_ii_ratio = 1.0F;
    std::uint32_t max_included_literals = 0U;  // Zero means all literals.
    std::uint32_t epochs_trained = 0U;
    std::uint64_t seed = 0U;
    bool feature_negation = true;
    bool boost_true_positive_feedback = true;
    bool onset_sustain_hard_negatives = false;
    float onset_sustain_hard_negative_probability = 0.0F;
    bool onset_sustain_hard_negative_weight_only = false;
};

// Outputs map contiguously to MIDI notes [minimum_note, maximum_note].
// MIDI channels are one-based here (1..16), matching the UI convention.
struct TmModelMidiMetadata {
    std::int32_t minimum_note = 0;
    std::int32_t maximum_note = 0;
    std::uint32_t channel = 1U;
    std::uint32_t audio_sample_rate = 0U;
    std::uint32_t analysis_hop_samples = 0U;
};

struct NativeTmModel {
    TmModelHead head = TmModelHead::activity;
    TmModelDimensions dimensions{};
    TmModelTrainingConfig training{};
    TmModelMidiMetadata midi{};

    // Calibrated inference threshold. This is intentionally separate from the
    // positive TM training threshold and may be negative.
    std::int32_t score_threshold = 0;

    // SHA-256 of the canonical binary feature-semantics descriptor used by
    // the training dataset. All-zero identifies legacy v1/v2 models whose
    // feature meaning cannot be proven from the binary artifact alone.
    std::array<std::uint8_t, 32> feature_fingerprint_sha256{};

    std::vector<std::uint32_t> ta_bitplanes;
    std::vector<std::int32_t> weights;
};

[[nodiscard]] std::uint32_t tm_model_literal_count(
    const NativeTmModel& model) noexcept;
[[nodiscard]] std::uint32_t tm_model_literal_word_count(
    const NativeTmModel& model) noexcept;

// Throws std::invalid_argument when an in-memory model is inconsistent.
void validate_tm_model(const NativeTmModel& model);

// Verifies that a preprocessed dataset uses exactly the feature/output schema
// and audio/MIDI time base stored with the model. Keeping this check separate
// from CUDA inference makes incompatibilities fail before any GPU work starts.
void validate_tm_dataset_compatibility(
    const NativeDataset& dataset,
    const NativeTmModel& model,
    bool allow_legacy_feature_contract = false);

// SHA-256 covers the complete canonical little-endian header (with the digest
// field zeroed) and both payloads, so config as well as learned state is
// protected against accidental corruption.
[[nodiscard]] std::array<std::uint8_t, 32> calculate_tm_model_checksum(
    const NativeTmModel& model);
[[nodiscard]] std::string tm_model_checksum_hex(
    const std::array<std::uint8_t, 32>& digest);

void save_tm_model(
    const std::filesystem::path& path,
    const NativeTmModel& model);
[[nodiscard]] NativeTmModel load_tm_model(
    const std::filesystem::path& path,
    bool verify_checksum = true);

}  // namespace tmgm::native
