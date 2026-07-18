#include "tmgm/dataset.hpp"
#include "tmgm/ensemble_bundle.hpp"
#include "tmgm/ensemble_inference.hpp"

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] std::int32_t parse_i32(
    const char* begin,
    const char* end,
    const char* label) {
    std::int32_t value = 0;
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::runtime_error(std::string("invalid ") + label + " in TSV");
    }
    return value;
}

[[nodiscard]] std::vector<std::int32_t> read_score_prefix(
    const std::filesystem::path& path,
    const std::uint64_t frame_count,
    const std::uint32_t output_count) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open expected score TSV: " + path.string());
    }
    std::vector<std::int32_t> scores;
    scores.reserve(static_cast<std::size_t>(frame_count) * output_count);
    std::string line;
    bool columns_seen = false;
    std::uint64_t expected_frame = 0U;
    while (expected_frame < frame_count && std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            throw std::runtime_error("blank line in expected score TSV");
        }
        if (line.front() == '#') {
            if (columns_seen) {
                throw std::runtime_error("metadata after score columns");
            }
            continue;
        }
        if (!columns_seen) {
            columns_seen = true;
            continue;
        }

        std::size_t begin = 0U;
        auto next_field = [&]() -> std::pair<const char*, const char*> {
            const auto end = line.find('\t', begin);
            const auto length = end == std::string::npos
                ? line.size() - begin
                : end - begin;
            const auto* first = line.data() + begin;
            const auto* last = first + length;
            begin = end == std::string::npos ? line.size() : end + 1U;
            return {first, last};
        };
        const auto frame_field = next_field();
        const auto frame = parse_i32(
            frame_field.first, frame_field.second, "frame index");
        if (frame < 0 || static_cast<std::uint64_t>(frame) != expected_frame) {
            throw std::runtime_error("unexpected frame index in score TSV");
        }
        for (std::uint32_t output = 0; output < output_count; ++output) {
            const auto field = next_field();
            scores.push_back(parse_i32(field.first, field.second, "raw score"));
        }
        ++expected_frame;
    }
    if (!columns_seen || expected_frame != frame_count) {
        throw std::runtime_error("expected score TSV has too few rows");
    }
    return scores;
}

void pack_frame_u32(
    const tmgm::native::NativeDataset& dataset,
    const std::uint64_t frame,
    std::vector<std::uint32_t>& destination) {
    const auto* source = dataset.feature_words.data() +
        static_cast<std::size_t>(frame) *
            dataset.header.feature_words_per_row;
    for (std::size_t word = 0U; word < destination.size(); ++word) {
        const auto source_word = source[word / 2U];
        destination[word] = static_cast<std::uint32_t>(
            word % 2U == 0U ? source_word : source_word >> 32U);
    }
}

}  // namespace

int main(const int argc, char** argv) {
    namespace fs = std::filesystem;
    if (argc != 2) {
        std::cerr << "usage: ensemble_production_parity_test <repository-root>\n";
        return 2;
    }
    try {
        const fs::path root = argv[1];
        const auto artifact_root = root /
            "artifacts/native-full-natural-d2w3";
        const auto production_root = artifact_root /
            "production-ensemble/onset-qgrid-v1";
        const auto dataset = tmgm::native::load_dataset(
            production_root / "parity-128.tmgd");
        auto bundle = tmgm::native::load_ensemble_bundle(
            production_root / "production.tmgmbundle");
        const auto prediction =
            tmgm::native::predict_ensemble_cpu(dataset, bundle, true);

        const std::vector<fs::path> raw_oracles{
            artifact_root /
                "ablations/c256-natural-d2w3-q05/activity/validation.tsv",
            artifact_root /
                "ablations/c512-t256-natural-d2w3-q8/activity/validation.tsv",
            artifact_root /
                "ablations/c256-natural-d2w3-q05/onset/validation.tsv",
            artifact_root /
                "ablations/c256-natural-d2w3-q1/onset/validation.tsv",
            artifact_root /
                "ablations/c256-natural-d2w3-q2/onset/validation.tsv",
            artifact_root /
                "ablations/c256-natural-d2w3-q4/onset/validation.tsv",
            artifact_root /
                "ablations/c256-natural-d2w3-q8/onset/validation.tsv",
        };
        require(
            raw_oracles.size() == prediction.raw_member_scores.size(),
            "production raw oracle count differs");
        for (std::size_t index = 0; index < raw_oracles.size(); ++index) {
            const auto expected = read_score_prefix(
                raw_oracles[index],
                prediction.frame_count,
                prediction.output_count);
            require(
                prediction.raw_member_scores[index] == expected,
                "production raw member scores differ from native oracle");
        }

        const auto expected_activity = read_score_prefix(
            production_root / "parity-cpu/activity-python-v2.tsv",
            prediction.frame_count,
            prediction.output_count);
        require(
            prediction.fused_activity_scores == expected_activity,
            "production activity mean differs from Python oracle");
        const auto expected_onset = read_score_prefix(
            production_root / "fitset-validation.oracle.tsv",
            prediction.frame_count,
            prediction.output_count);
        require(
            prediction.fused_onset_scores == expected_onset,
            "production onset mean differs from Python oracle");

        tmgm::native::EnsembleCpuFramePredictor frame_predictor(
            std::move(bundle), true);
        std::vector<std::uint32_t> packed(
            frame_predictor.packed_feature_word_count());
        std::vector<std::int32_t> raw(
            frame_predictor.raw_member_score_count());
        std::vector<std::int32_t> activity(frame_predictor.output_count());
        std::vector<std::uint8_t> activity_predictions(
            frame_predictor.output_count());
        std::vector<std::int32_t> onset(frame_predictor.output_count());
        std::vector<std::uint8_t> onset_predictions(
            frame_predictor.output_count());
        const tmgm::native::EnsembleFrameOutputBuffers frame_outputs{
            raw.data(), raw.size(),
            activity.data(), activity.size(),
            activity_predictions.data(), activity_predictions.size(),
            onset.data(), onset.size(),
            onset_predictions.data(), onset_predictions.size(),
        };
        for (std::uint64_t frame = 0U;
             frame < prediction.frame_count;
             ++frame) {
            pack_frame_u32(dataset, frame, packed);
            require(
                frame_predictor.predict_frame(
                    packed.data(), packed.size(), frame_outputs) ==
                    tmgm::native::EnsembleFramePredictStatus::success,
                "production per-frame predictor failed");
            const auto frame_base = static_cast<std::size_t>(frame) *
                prediction.output_count;
            for (std::size_t member = 0U;
                 member < frame_predictor.member_count();
                 ++member) {
                for (std::uint32_t output = 0U;
                     output < prediction.output_count;
                     ++output) {
                    require(
                        raw[member * prediction.output_count + output] ==
                            prediction.raw_member_scores[member][
                                frame_base + output],
                        "production per-frame raw score differs from batch");
                }
            }
            for (std::uint32_t output = 0U;
                 output < prediction.output_count;
                 ++output) {
                const auto index = frame_base + output;
                require(activity[output] ==
                            prediction.fused_activity_scores[index],
                        "production per-frame activity fusion differs from batch");
                require(activity_predictions[output] ==
                            prediction.activity_predictions[index],
                        "production per-frame activity decision differs from batch");
                require(onset[output] == prediction.fused_onset_scores[index],
                        "production per-frame onset fusion differs from batch");
                require(onset_predictions[output] ==
                            prediction.onset_predictions[index],
                        "production per-frame onset decision differs from batch");
            }
        }

        std::cout <<
            "TMGMBND production batch/per-frame raw/fused parity tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
