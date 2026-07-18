#include "tmgm/ensemble_inference.hpp"

#include "tmgm/dataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tmgm::native {
namespace {

[[nodiscard]] std::invalid_argument compatibility_error(
    const std::string& message) {
    return std::invalid_argument(
        "dataset is incompatible with native TM ensemble bundle: " + message);
}

[[nodiscard]] std::size_t checked_score_count(
    const std::uint64_t frames,
    const std::uint32_t outputs) {
    if (outputs != 0U && frames >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) /
            outputs) {
        throw std::overflow_error("ensemble score matrix does not fit address space");
    }
    return static_cast<std::size_t>(frames * outputs);
}

[[nodiscard]] bool feature_value(
    const std::uint64_t* row,
    const std::uint32_t feature) noexcept {
    return (row[feature / 64U] & (std::uint64_t{1U} << (feature % 64U))) != 0U;
}

[[nodiscard]] bool feature_value_u32(
    const std::uint32_t* row,
    const std::uint32_t feature) noexcept {
    return (row[feature / 32U] &
            (std::uint32_t{1U} << (feature % 32U))) != 0U;
}

[[nodiscard]] bool clause_fires(
    const SparseTmEnsembleMember& member,
    const std::uint32_t clause,
    const std::uint64_t* feature_row) noexcept {
    const auto begin = member.clause_offsets[clause];
    const auto end = member.clause_offsets[clause + 1U];
    // TMU/native CUDA explicitly suppress all-exclude clauses at inference.
    if (begin == end) {
        return false;
    }
    for (auto index = begin; index < end; ++index) {
        const auto literal = static_cast<std::uint32_t>(member.literal_ids[index]);
        const auto negative = literal >= member.feature_count;
        const auto feature = negative ? literal - member.feature_count : literal;
        const auto value = feature_value(feature_row, feature);
        if ((!negative && !value) || (negative && value)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool clause_fires_u32(
    const SparseTmEnsembleMember& member,
    const std::uint32_t clause,
    const std::uint32_t* feature_row) noexcept {
    const auto begin = member.clause_offsets[clause];
    const auto end = member.clause_offsets[clause + 1U];
    if (begin == end) {
        return false;
    }
    for (auto index = begin; index < end; ++index) {
        const auto literal = static_cast<std::uint32_t>(member.literal_ids[index]);
        const auto negative = literal >= member.feature_count;
        const auto feature = negative ? literal - member.feature_count : literal;
        const auto value = feature_value_u32(feature_row, feature);
        if ((!negative && !value) || (negative && value)) {
            return false;
        }
    }
    return true;
}

void predict_member(
    const NativeDataset& dataset,
    const SparseTmEnsembleMember& member,
    std::vector<std::int32_t>& scores) {
    const auto score_count = checked_score_count(
        dataset.header.frame_count, member.output_count);
    scores.assign(score_count, 0);
    std::vector<std::uint8_t> clause_outputs(member.clause_count, 0U);
    for (std::uint64_t frame = 0; frame < dataset.header.frame_count; ++frame) {
        const auto* feature_row = dataset.feature_words.data() +
            static_cast<std::size_t>(frame) *
                dataset.header.feature_words_per_row;
        for (std::uint32_t clause = 0; clause < member.clause_count; ++clause) {
            clause_outputs[clause] =
                clause_fires(member, clause, feature_row) ? 1U : 0U;
        }
        const auto score_base = static_cast<std::size_t>(frame) *
            member.output_count;
        for (std::uint32_t output = 0; output < member.output_count; ++output) {
            std::int64_t sum = 0;
            for (std::uint32_t clause = 0; clause < member.clause_count; ++clause) {
                if (clause_outputs[clause] != 0U) {
                    sum += member.weights[
                        static_cast<std::size_t>(clause) * member.output_count +
                        output];
                }
            }
            sum = std::max<std::int64_t>(
                std::numeric_limits<std::int32_t>::min(),
                std::min<std::int64_t>(
                    std::numeric_limits<std::int32_t>::max(), sum));
            scores[score_base + output] = static_cast<std::int32_t>(sum);
        }
    }
}

// NumPy np.rint uses round-to-nearest, ties-to-even. std::round is unsuitable
// because it rounds half values away from zero. Values are clamped before the
// integer conversion so this is independent of the process FP rounding mode.
[[nodiscard]] std::int32_t quantize_ties_to_even(
    const float value,
    const std::uint32_t quantization) noexcept {
    const auto scaled =
        static_cast<double>(value) * static_cast<double>(quantization);
    constexpr auto minimum = std::numeric_limits<std::int32_t>::min();
    constexpr auto maximum = std::numeric_limits<std::int32_t>::max() - 1;
    if (scaled <= static_cast<double>(minimum)) {
        return minimum;
    }
    if (scaled >= static_cast<double>(maximum)) {
        return maximum;
    }
    const auto lower_double = std::floor(scaled);
    auto lower = static_cast<std::int64_t>(lower_double);
    const auto fraction = scaled - lower_double;
    if (fraction > 0.5 || (fraction == 0.5 && lower % 2 != 0)) {
        ++lower;
    }
    return static_cast<std::int32_t>(lower);
}

}  // namespace

EnsembleCpuFramePredictor::EnsembleCpuFramePredictor(
    EnsembleBundle bundle,
    const bool allow_legacy_feature_contract)
    : bundle_(std::move(bundle)) {
    validate_ensemble_bundle(bundle_);
    if (bundle_.format_version == kEnsembleBundleLegacyFormatVersion &&
        !allow_legacy_feature_contract) {
        throw std::invalid_argument(
            "cannot prepare legacy TM ensemble without explicit "
            "feature-contract opt-in");
    }
    activity_member_count_ = static_cast<std::size_t>(std::count_if(
        bundle_.members.begin(),
        bundle_.members.end(),
        [](const SparseTmEnsembleMember& member) {
            return member.head == EnsembleMemberHead::activity;
        }));
    packed_feature_word_count_ =
        (static_cast<std::size_t>(bundle_.feature_count) + 31U) / 32U;
    if (bundle_.members.size() >
        std::numeric_limits<std::size_t>::max() / bundle_.output_count) {
        throw std::overflow_error(
            "per-frame ensemble raw score storage does not fit address space");
    }
    raw_member_score_count_ =
        bundle_.members.size() * static_cast<std::size_t>(bundle_.output_count);
    output_accumulators_.resize(bundle_.output_count);
}

EnsembleFramePredictStatus EnsembleCpuFramePredictor::predict_frame(
    const std::uint32_t* packed_feature_words,
    const std::size_t packed_feature_word_count,
    const EnsembleFrameOutputBuffers& outputs) noexcept {
    if (packed_feature_words == nullptr) {
        return EnsembleFramePredictStatus::null_feature_words;
    }
    if (packed_feature_word_count != packed_feature_word_count_) {
        return EnsembleFramePredictStatus::wrong_feature_word_count;
    }
    if (outputs.raw_member_scores == nullptr ||
        outputs.fused_activity_scores == nullptr ||
        outputs.activity_predictions == nullptr ||
        outputs.fused_onset_scores == nullptr ||
        outputs.onset_predictions == nullptr) {
        return EnsembleFramePredictStatus::null_output_buffer;
    }
    if (outputs.raw_member_score_count < raw_member_score_count_ ||
        outputs.fused_activity_score_count < bundle_.output_count ||
        outputs.activity_prediction_count < bundle_.output_count ||
        outputs.fused_onset_score_count < bundle_.output_count ||
        outputs.onset_prediction_count < bundle_.output_count) {
        return EnsembleFramePredictStatus::output_buffer_too_small;
    }

    const auto output_count = static_cast<std::size_t>(bundle_.output_count);
    for (std::size_t member_index = 0U;
         member_index < bundle_.members.size();
         ++member_index) {
        const auto& member = bundle_.members[member_index];
        std::fill(
            output_accumulators_.begin(),
            output_accumulators_.end(),
            std::int64_t{0});
        for (std::uint32_t clause = 0U;
             clause < member.clause_count;
             ++clause) {
            if (!clause_fires_u32(member, clause, packed_feature_words)) {
                continue;
            }
            const auto weight_base =
                static_cast<std::size_t>(clause) * output_count;
            for (std::size_t output = 0U; output < output_count; ++output) {
                output_accumulators_[output] +=
                    member.weights[weight_base + output];
            }
        }
        const auto raw_base = member_index * output_count;
        for (std::size_t output = 0U; output < output_count; ++output) {
            const auto clamped = std::max<std::int64_t>(
                std::numeric_limits<std::int32_t>::min(),
                std::min<std::int64_t>(
                    std::numeric_limits<std::int32_t>::max(),
                    output_accumulators_[output]));
            outputs.raw_member_scores[raw_base + output] =
                static_cast<std::int32_t>(clamped);
        }
    }

    const auto fuse_head = [&](const std::size_t member_begin,
                               const std::size_t member_end,
                               const EnsembleHeadConfig& config,
                               std::int32_t* fused_scores,
                               std::uint8_t* predictions) noexcept {
        const auto member_count_f32 =
            static_cast<float>(member_end - member_begin);
        for (std::size_t output = 0U; output < output_count; ++output) {
            // Deliberately sequential float32 operations in authenticated
            // member order, matching predict_ensemble_cpu and NumPy float32.
            float sum = 0.0F;
            for (std::size_t member_index = member_begin;
                 member_index < member_end;
                 ++member_index) {
                const auto& member = bundle_.members[member_index];
                const auto raw = outputs.raw_member_scores[
                    member_index * output_count + output];
                const float centered = static_cast<float>(raw) -
                    static_cast<float>(member.score_threshold);
                const float normalized = centered / member.robust_scale;
                sum = sum + normalized;
            }
            const float mean = sum / member_count_f32;
            const auto fused =
                quantize_ties_to_even(mean, config.quantization);
            fused_scores[output] = fused;
            predictions[output] =
                fused >= config.ensemble_threshold ? 1U : 0U;
        }
    };
    fuse_head(
        0U,
        activity_member_count_,
        bundle_.activity,
        outputs.fused_activity_scores,
        outputs.activity_predictions);
    fuse_head(
        activity_member_count_,
        bundle_.members.size(),
        bundle_.onset,
        outputs.fused_onset_scores,
        outputs.onset_predictions);
    return EnsembleFramePredictStatus::success;
}

void validate_ensemble_dataset_compatibility(
    const NativeDataset& dataset,
    const EnsembleBundle& bundle,
    const bool allow_legacy_feature_contract) {
    if (dataset.header.feature_count != bundle.feature_count) {
        throw compatibility_error("feature count differs");
    }
    if (dataset.header.note_count != bundle.output_count) {
        throw compatibility_error("output count differs");
    }
    if (dataset.header.midi_min != bundle.midi_min ||
        dataset.header.midi_max != bundle.midi_max) {
        throw compatibility_error("MIDI range differs");
    }
    if (dataset.header.sample_rate != bundle.sample_rate) {
        throw compatibility_error("sample rate differs");
    }
    if (dataset.header.hop_size != bundle.hop_size) {
        throw compatibility_error("analysis hop differs");
    }
    const auto dataset_legacy = std::all_of(
        dataset.header.feature_fingerprint_sha256.begin(),
        dataset.header.feature_fingerprint_sha256.end(),
        [](const std::uint8_t value) { return value == 0U; });
    const auto bundle_legacy =
        bundle.format_version == kEnsembleBundleLegacyFormatVersion;
    if (dataset_legacy || bundle_legacy) {
        if (!allow_legacy_feature_contract) {
            throw compatibility_error(
                "feature-semantics fingerprint is unavailable for a legacy "
                "artifact; pass an explicit legacy opt-in only for audit use");
        }
    } else if (dataset.header.feature_fingerprint_sha256 !=
               bundle.feature_fingerprint_sha256) {
        throw compatibility_error("feature-semantics fingerprint differs");
    }
}

EnsembleCpuPrediction predict_ensemble_cpu(
    const NativeDataset& dataset,
    const EnsembleBundle& bundle,
    const bool allow_legacy_feature_contract) {
    dataset.validate();
    validate_ensemble_bundle(bundle);
    validate_ensemble_dataset_compatibility(
        dataset, bundle, allow_legacy_feature_contract);

    EnsembleCpuPrediction prediction;
    prediction.frame_count = dataset.header.frame_count;
    prediction.output_count = bundle.output_count;
    const auto score_count = checked_score_count(
        prediction.frame_count, prediction.output_count);
    prediction.raw_member_scores.resize(bundle.members.size());
    for (std::size_t member = 0; member < bundle.members.size(); ++member) {
        predict_member(
            dataset, bundle.members[member], prediction.raw_member_scores[member]);
    }

    std::size_t activity_count = 0U;
    while (activity_count < bundle.members.size() &&
           bundle.members[activity_count].head ==
               EnsembleMemberHead::activity) {
        ++activity_count;
    }
    prediction.fused_activity_scores.resize(score_count, 0);
    prediction.activity_predictions.resize(score_count, 0U);
    prediction.fused_onset_scores.resize(score_count, 0);
    prediction.onset_predictions.resize(score_count, 0U);
    const auto fuse_head = [&](const std::size_t member_begin,
                               const std::size_t member_end,
                               const EnsembleHeadConfig& config,
                               std::vector<std::int32_t>& fused_scores,
                               std::vector<std::uint8_t>& fused_predictions) {
        const auto member_count = member_end - member_begin;
        const auto member_count_f32 = static_cast<float>(member_count);
        for (std::size_t index = 0; index < score_count; ++index) {
            // Keep this explicitly sequential and float32. For the production
            // member counts this mirrors np.mean(axis=0, dtype=float32).
            float sum = 0.0F;
            for (std::size_t member_index = member_begin;
                 member_index < member_end;
                 ++member_index) {
                const auto& member = bundle.members[member_index];
                const auto raw =
                    prediction.raw_member_scores[member_index][index];
                const float centered = static_cast<float>(raw) -
                                       static_cast<float>(member.score_threshold);
                const float normalized = centered / member.robust_scale;
                sum = sum + normalized;
            }
            const float mean = sum / member_count_f32;
            const auto fused =
                quantize_ties_to_even(mean, config.quantization);
            fused_scores[index] = fused;
            fused_predictions[index] =
                fused >= config.ensemble_threshold ? 1U : 0U;
        }
    };
    fuse_head(
        0U,
        activity_count,
        bundle.activity,
        prediction.fused_activity_scores,
        prediction.activity_predictions);
    fuse_head(
        activity_count,
        bundle.members.size(),
        bundle.onset,
        prediction.fused_onset_scores,
        prediction.onset_predictions);
    return prediction;
}

}  // namespace tmgm::native
