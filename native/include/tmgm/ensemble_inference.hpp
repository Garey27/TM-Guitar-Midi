#pragma once

#include "tmgm/ensemble_bundle.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace tmgm::native {

struct NativeDataset;

// raw_member_scores[m] is row-major [frame][output] for bundle.members[m].
// All activity members precede all onset members in bundle format v1.
struct EnsembleCpuPrediction {
    std::uint64_t frame_count = 0U;
    std::uint32_t output_count = 0U;
    std::vector<std::vector<std::int32_t>> raw_member_scores;
    std::vector<std::int32_t> fused_activity_scores;
    std::vector<std::uint8_t> activity_predictions;
    std::vector<std::int32_t> fused_onset_scores;
    std::vector<std::uint8_t> onset_predictions;
};

// Caller-owned output storage for one frame. raw_member_scores is laid out as
// [member][output], in the authenticated member order stored by the bundle.
// All six buffers are required, must not overlap, and may be larger than the
// corresponding minimum sizes reported by EnsembleCpuFramePredictor.
struct EnsembleFrameOutputBuffers {
    std::int32_t* raw_member_scores = nullptr;
    std::size_t raw_member_score_count = 0U;
    std::int32_t* fused_activity_scores = nullptr;
    std::size_t fused_activity_score_count = 0U;
    std::uint8_t* activity_predictions = nullptr;
    std::size_t activity_prediction_count = 0U;
    std::int32_t* fused_onset_scores = nullptr;
    std::size_t fused_onset_score_count = 0U;
    std::uint8_t* onset_predictions = nullptr;
    std::size_t onset_prediction_count = 0U;
};

enum class EnsembleFramePredictStatus : std::uint8_t {
    success = 0U,
    null_feature_words,
    wrong_feature_word_count,
    null_output_buffer,
    output_buffer_too_small,
};

// Prepared, owning, sparse CPU inference engine for the realtime audio thread.
// Construction validates and takes ownership of the authenticated bundle and
// performs the only scratch allocation. Pass std::move(bundle) after loading to
// avoid copying model payloads. Legacy bundles require an explicit audit-only
// opt-in because they cannot bind the frontend's feature semantics.
// predict_frame performs no allocation, file I/O, locking, or exception
// throwing. One prepared instance must be owned by only one realtime stream;
// use a separate instance (and scratch) for each concurrently executing stream.
//
// Input features use canonical LSB-first uint32 packing: feature c is bit
// c % 32 of packed_feature_words[c / 32]. Padding bits in the final word are
// ignored. This is intentionally a single-feature-contract/single-bank API;
// cross-bank fusion belongs to a higher layer that owns one prepared predictor
// per feature contract.
class EnsembleCpuFramePredictor {
public:
    explicit EnsembleCpuFramePredictor(
        EnsembleBundle bundle,
        bool allow_legacy_feature_contract = false);

    EnsembleCpuFramePredictor(const EnsembleCpuFramePredictor&) = delete;
    EnsembleCpuFramePredictor& operator=(
        const EnsembleCpuFramePredictor&) = delete;
    EnsembleCpuFramePredictor(EnsembleCpuFramePredictor&&) noexcept = default;
    EnsembleCpuFramePredictor& operator=(
        EnsembleCpuFramePredictor&&) noexcept = default;

    [[nodiscard]] std::uint32_t feature_count() const noexcept {
        return bundle_.feature_count;
    }
    [[nodiscard]] std::size_t packed_feature_word_count() const noexcept {
        return packed_feature_word_count_;
    }
    [[nodiscard]] std::uint32_t output_count() const noexcept {
        return bundle_.output_count;
    }
    [[nodiscard]] std::uint32_t sample_rate() const noexcept {
        return bundle_.sample_rate;
    }
    [[nodiscard]] std::uint32_t hop_size() const noexcept {
        return bundle_.hop_size;
    }
    [[nodiscard]] const std::array<std::uint8_t, 32>&
    feature_fingerprint_sha256() const noexcept {
        return bundle_.feature_fingerprint_sha256;
    }
    [[nodiscard]] std::size_t member_count() const noexcept {
        return bundle_.members.size();
    }
    [[nodiscard]] std::size_t raw_member_score_count() const noexcept {
        return raw_member_score_count_;
    }
    // Preparation-time metadata access for an external cross-bank coordinator
    // (member thresholds/scales and authenticated feature fingerprint). Do not
    // traverse model vectors from the realtime callback.
    [[nodiscard]] const EnsembleBundle& prepared_bundle() const noexcept {
        return bundle_;
    }

    [[nodiscard]] EnsembleFramePredictStatus predict_frame(
        const std::uint32_t* packed_feature_words,
        std::size_t packed_feature_word_count,
        const EnsembleFrameOutputBuffers& outputs) noexcept;

private:
    EnsembleBundle bundle_;
    std::size_t activity_member_count_ = 0U;
    std::size_t packed_feature_word_count_ = 0U;
    std::size_t raw_member_score_count_ = 0U;
    std::vector<std::int64_t> output_accumulators_;
};

void validate_ensemble_dataset_compatibility(
    const NativeDataset& dataset,
    const EnsembleBundle& bundle,
    bool allow_legacy_feature_contract = false);

// Deterministic sparse CPU oracle/implementation. Mean fusion deliberately
// follows NumPy float32 reduction in fixed member order, and quantization uses
// explicit round-to-nearest, ties-to-even semantics.
[[nodiscard]] EnsembleCpuPrediction predict_ensemble_cpu(
    const NativeDataset& dataset,
    const EnsembleBundle& bundle,
    bool allow_legacy_feature_contract = false);

}  // namespace tmgm::native
