#include "tmgm/dataset.hpp"
#include "tmgm/ensemble_bundle.hpp"
#include "tmgm/ensemble_inference.hpp"

#include <array>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
bool g_track_allocations = false;
std::size_t g_allocation_count = 0U;
}  // namespace

void* operator new(const std::size_t size) {
    if (g_track_allocations) {
        ++g_allocation_count;
    }
    if (auto* memory = std::malloc(size == 0U ? 1U : size)) {
        return memory;
    }
    throw std::bad_alloc();
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

[[nodiscard]] std::array<std::uint8_t, 32> digest_from_hex(
    const std::string& value) {
    if (value.size() != 64U) {
        throw std::runtime_error("bad test digest length");
    }
    const auto digit = [](const char character) -> std::uint8_t {
        if (character >= '0' && character <= '9') {
            return static_cast<std::uint8_t>(character - '0');
        }
        if (character >= 'a' && character <= 'f') {
            return static_cast<std::uint8_t>(character - 'a' + 10);
        }
        throw std::runtime_error("bad test digest digit");
    };
    std::array<std::uint8_t, 32> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<std::uint8_t>(
            (digit(value[index * 2U]) << 4U) |
            digit(value[index * 2U + 1U]));
    }
    return result;
}

[[nodiscard]] tmgm::native::SparseTmEnsembleMember activity_member(
    const std::string& identifier,
    const std::uint8_t source_hash_byte) {
    tmgm::native::SparseTmEnsembleMember member;
    member.identifier = identifier;
    member.head = tmgm::native::EnsembleMemberHead::activity;
    member.score_threshold = 0;
    member.robust_scale = 1.0F;
    member.feature_count = 65;
    member.output_count = 1;
    member.clause_count = 7;
    member.literal_count = 130;
    member.source_model_sha256.fill(source_hash_byte);
    // Clauses: f0, !f1, empty, f2&!f2, f64, f31, f32.
    member.clause_offsets = {0, 1, 2, 2, 4, 5, 6, 7};
    member.literal_ids = {0, 66, 2, 67, 64, 31, 32};
    member.weights = {3, 5, 100, 100, 7, 11, 13};
    return member;
}

[[nodiscard]] tmgm::native::SparseTmEnsembleMember onset_member(
    const std::string& identifier,
    const std::vector<std::int16_t>& weights,
    const std::uint8_t source_hash_byte) {
    tmgm::native::SparseTmEnsembleMember member;
    member.identifier = identifier;
    member.head = tmgm::native::EnsembleMemberHead::onset;
    member.score_threshold = 0;
    member.robust_scale = 1024.0F;
    member.feature_count = 65;
    member.output_count = 1;
    member.clause_count = 4;
    member.literal_count = 130;
    member.source_model_sha256.fill(source_hash_byte);
    member.clause_offsets = {0, 1, 2, 3, 4};
    member.literal_ids = {10, 11, 12, 13};
    member.weights = weights;
    return member;
}

[[nodiscard]] tmgm::native::EnsembleBundle bundle_fixture() {
    tmgm::native::EnsembleBundle bundle;
    bundle.feature_count = 65;
    bundle.output_count = 1;
    bundle.midi_min = 40;
    bundle.midi_max = 40;
    bundle.sample_rate = 22050;
    bundle.hop_size = 256;
    bundle.activity.fusion = tmgm::native::EnsembleFusion::mean;
    bundle.activity.quantization = 1;
    bundle.activity.ensemble_threshold = 10;
    bundle.activity.member_order_sha256 = digest_from_hex(
        "a7e4878fd0ac9e94f0c734a3d4e06b128e7fcf5b719b4b158af24399182c4ce4");
    bundle.onset.fusion = tmgm::native::EnsembleFusion::mean;
    bundle.onset.quantization = 1024;
    bundle.onset.ensemble_threshold = 0;
    bundle.onset.member_order_sha256 = digest_from_hex(
        "95c32c14c795bd2e20ec966046ab083b263176704827da28b411b562aa3073d1");
    bundle.feature_fingerprint_sha256.fill(0x44U);
    bundle.bundle_checksum_sha256.fill(0x55U);
    bundle.members.push_back(activity_member("a1", 0x11U));
    bundle.members.push_back(activity_member("a2", 0x12U));
    bundle.members.push_back(onset_member("q1", {1, 3, -1, -3}, 0x22U));
    bundle.members.push_back(onset_member("q2", {0, 0, 0, 0}, 0x33U));
    return bundle;
}

[[nodiscard]] tmgm::native::NativeDataset dataset_fixture() {
    tmgm::native::NativeDataset dataset;
    dataset.header.frame_count = 4;
    dataset.header.feature_count = 65;
    dataset.header.feature_words_per_row = 2;
    dataset.header.note_count = 1;
    dataset.header.label_words_per_row = 1;
    dataset.header.midi_min = 40;
    dataset.header.midi_max = 40;
    dataset.header.sample_rate = 22050;
    dataset.header.hop_size = 256;
    dataset.header.onset_index_count = 0;
    dataset.header.feature_fingerprint_sha256.fill(0x44U);
    dataset.feature_words.resize(8, 0U);
    dataset.activity_words.resize(4, 0U);
    dataset.onset_words.resize(4, 0U);

    dataset.set_feature(0, 0, true);
    dataset.set_feature(0, 31, true);
    dataset.set_feature(0, 32, true);
    dataset.set_feature(0, 64, true);
    dataset.set_feature(0, 10, true);
    dataset.set_feature(1, 1, true);
    dataset.set_feature(1, 11, true);
    dataset.set_feature(2, 12, true);
    dataset.set_feature(3, 13, true);
    return dataset;
}

[[nodiscard]] std::vector<std::uint32_t> packed_u32_frame(
    const tmgm::native::NativeDataset& dataset,
    const std::uint64_t frame) {
    const auto count =
        (static_cast<std::size_t>(dataset.header.feature_count) + 31U) / 32U;
    std::vector<std::uint32_t> result(count, 0U);
    const auto* source = dataset.feature_words.data() +
        static_cast<std::size_t>(frame) *
            dataset.header.feature_words_per_row;
    for (std::size_t word = 0U; word < count; ++word) {
        const auto source_word = source[word / 2U];
        result[word] = static_cast<std::uint32_t>(
            word % 2U == 0U ? source_word : source_word >> 32U);
    }
    return result;
}

}  // namespace

int main() {
    try {
        auto bundle = bundle_fixture();
        const auto dataset = dataset_fixture();
        tmgm::native::validate_ensemble_bundle(bundle);
        const auto prediction =
            tmgm::native::predict_ensemble_cpu(dataset, bundle);

        require(prediction.frame_count == 4U, "wrong prediction frame count");
        require(prediction.output_count == 1U, "wrong prediction output count");
        require(prediction.raw_member_scores.size() == 4U, "wrong raw member count");
        require(
            prediction.raw_member_scores[0] ==
                std::vector<std::int32_t>({39, 0, 5, 5}),
            "positive/negative/empty/contradictory or boundary clause semantics differ");
        require(
            prediction.raw_member_scores[1] ==
                std::vector<std::int32_t>({39, 0, 5, 5}),
            "second activity raw scores differ");
        require(
            prediction.fused_activity_scores ==
                std::vector<std::int32_t>({39, 0, 5, 5}),
            "multi-member activity mean fusion differs");
        require(
            prediction.raw_member_scores[2] ==
                std::vector<std::int32_t>({1, 3, -1, -3}),
            "first onset raw scores differ");
        require(
            prediction.raw_member_scores[3] ==
                std::vector<std::int32_t>({0, 0, 0, 0}),
            "second onset raw scores differ");
        require(
            prediction.activity_predictions ==
                std::vector<std::uint8_t>({1, 0, 0, 0}),
            "activity threshold differs");
        // (1 + 0) / 2 -> 0.5 -> 0; 3/2 -> 1.5 -> 2;
        // -1/2 -> -0.5 -> 0; -3/2 -> -1.5 -> -2.
        require(
            prediction.fused_onset_scores ==
                std::vector<std::int32_t>({0, 2, 0, -2}),
            "float32 mean or ties-to-even quantization differs");
        require(
            prediction.onset_predictions ==
                std::vector<std::uint8_t>({1, 1, 1, 0}),
            "ensemble threshold differs");

        tmgm::native::EnsembleCpuFramePredictor frame_predictor(
            std::move(bundle));
        require(frame_predictor.feature_count() == 65U,
                "wrong prepared feature count");
        require(frame_predictor.packed_feature_word_count() == 3U,
                "wrong prepared uint32 feature word count");
        require(frame_predictor.member_count() == 4U,
                "wrong prepared member count");
        require(frame_predictor.raw_member_score_count() == 4U,
                "wrong prepared raw score count");
        std::vector<std::int32_t> raw(frame_predictor.raw_member_score_count());
        std::vector<std::int32_t> activity(frame_predictor.output_count());
        std::vector<std::uint8_t> activity_predictions(
            frame_predictor.output_count());
        std::vector<std::int32_t> onset(frame_predictor.output_count());
        std::vector<std::uint8_t> onset_predictions(
            frame_predictor.output_count());
        const tmgm::native::EnsembleFrameOutputBuffers frame_outputs{
            raw.data(),
            raw.size(),
            activity.data(),
            activity.size(),
            activity_predictions.data(),
            activity_predictions.size(),
            onset.data(),
            onset.size(),
            onset_predictions.data(),
            onset_predictions.size(),
        };
        for (std::uint64_t frame = 0U;
             frame < dataset.header.frame_count;
             ++frame) {
            const auto features = packed_u32_frame(dataset, frame);
            g_allocation_count = 0U;
            g_track_allocations = true;
            const auto frame_status = frame_predictor.predict_frame(
                features.data(), features.size(), frame_outputs);
            g_track_allocations = false;
            require(
                frame_status ==
                    tmgm::native::EnsembleFramePredictStatus::success,
                "prepared per-frame inference failed");
            require(g_allocation_count == 0U,
                    "prepared per-frame inference allocated memory");
            for (std::size_t member = 0U;
                 member < frame_predictor.member_count();
                 ++member) {
                require(
                    raw[member] ==
                        prediction.raw_member_scores[member][frame],
                    "per-frame raw score differs from batch CPU inference");
            }
            require(activity[0] == prediction.fused_activity_scores[frame],
                    "per-frame activity fusion differs from batch CPU inference");
            require(activity_predictions[0] ==
                        prediction.activity_predictions[frame],
                    "per-frame activity decision differs from batch CPU inference");
            require(onset[0] == prediction.fused_onset_scores[frame],
                    "per-frame onset fusion differs from batch CPU inference");
            require(onset_predictions[0] == prediction.onset_predictions[frame],
                    "per-frame onset decision differs from batch CPU inference");
        }

        const auto valid_features = packed_u32_frame(dataset, 0U);
        require(
            frame_predictor.predict_frame(
                valid_features.data(),
                valid_features.size() - 1U,
                frame_outputs) ==
                tmgm::native::EnsembleFramePredictStatus::wrong_feature_word_count,
            "per-frame inference accepted a truncated packed row");

        auto incompatible = dataset;
        incompatible.header.hop_size = 128;
        require_throws(
            [&] {
                static_cast<void>(
                    tmgm::native::predict_ensemble_cpu(
                        incompatible, bundle_fixture()));
            },
            "incompatible dataset timebase was accepted");

        incompatible = dataset;
        incompatible.header.feature_fingerprint_sha256.fill(0x45U);
        require_throws(
            [&] {
                static_cast<void>(
                    tmgm::native::predict_ensemble_cpu(
                        incompatible, bundle_fixture()));
            },
            "same-width feature-semantics mismatch was accepted");

        incompatible = dataset;
        incompatible.header.feature_fingerprint_sha256.fill(0U);
        require_throws(
            [&] {
                static_cast<void>(
                    tmgm::native::predict_ensemble_cpu(
                        incompatible, bundle_fixture()));
            },
            "legacy dataset was accepted without explicit opt-in");
        static_cast<void>(
            tmgm::native::predict_ensemble_cpu(
                incompatible, bundle_fixture(), true));

        auto legacy_bundle = bundle_fixture();
        legacy_bundle.format_version =
            tmgm::native::kEnsembleBundleLegacyFormatVersion;
        require_throws(
            [&] {
                static_cast<void>(
                    tmgm::native::predict_ensemble_cpu(dataset, legacy_bundle));
            },
            "legacy bundle was accepted without explicit opt-in");
        static_cast<void>(
            tmgm::native::predict_ensemble_cpu(dataset, legacy_bundle, true));
        require_throws(
            [&] {
                tmgm::native::EnsembleCpuFramePredictor rejected(
                    legacy_bundle);
                static_cast<void>(rejected);
            },
            "per-frame predictor accepted a legacy bundle without opt-in");
        tmgm::native::EnsembleCpuFramePredictor legacy_audit_predictor(
            legacy_bundle, true);
        static_cast<void>(legacy_audit_predictor);

        auto reordered = bundle_fixture();
        std::swap(reordered.members[2], reordered.members[3]);
        require_throws(
            [&] { tmgm::native::validate_ensemble_bundle(reordered); },
            "member order fingerprint mismatch was accepted");

        std::cout << "TMGMBND sparse CPU inference tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
