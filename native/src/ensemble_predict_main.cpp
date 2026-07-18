#include "tmgm/dataset.hpp"
#include "tmgm/ensemble_bundle.hpp"
#include "tmgm/ensemble_inference.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Arguments {
    std::filesystem::path dataset;
    std::filesystem::path bundle;
    std::filesystem::path activity_output;
    std::filesystem::path onset_output;
    std::filesystem::path member_output_directory;
    bool allow_legacy_feature_contract = false;
};

[[noreturn]] void usage(const char* executable) {
    throw std::runtime_error(
        std::string("Usage: ") + executable +
        " <input.tmgd> <ensemble.tmgmbundle>"
        " --activity-output <activity.tsv> --onset-output <onset.tsv>"
        " [--member-output-dir <directory>]"
        " [--allow-legacy-feature-contract]");
}

[[nodiscard]] Arguments parse_arguments(const int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
    }
    Arguments result;
    result.dataset = argv[1];
    result.bundle = argv[2];
    for (int index = 3; index < argc; ++index) {
        const std::string option = argv[index];
        auto value = [&]() -> std::filesystem::path {
            if (++index >= argc) {
                throw std::runtime_error(option + " requires a value");
            }
            return argv[index];
        };
        if (option == "--activity-output") {
            result.activity_output = value();
        } else if (option == "--onset-output") {
            result.onset_output = value();
        } else if (option == "--member-output-dir") {
            result.member_output_directory = value();
        } else if (option == "--allow-legacy-feature-contract") {
            result.allow_legacy_feature_contract = true;
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }
    if (result.activity_output.empty() || result.onset_output.empty()) {
        throw std::runtime_error(
            "--activity-output and --onset-output are required");
    }
    return result;
}

[[nodiscard]] const char* head_name(
    const tmgm::native::EnsembleMemberHead head) noexcept {
    return head == tmgm::native::EnsembleMemberHead::activity
        ? "activity"
        : "onset";
}

void write_scores(
    const std::filesystem::path& path,
    const tmgm::native::NativeDataset& dataset,
    const char* head,
    const std::int32_t threshold,
    const std::vector<std::int32_t>& scores,
    const std::vector<std::uint8_t>& predictions,
    const std::vector<std::pair<std::string, std::string>>& metadata) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot create score output: " + path.string());
    }
    stream << "#TMGM_SCORES_V1\n"
           << "#head=" << head << '\n'
           << "#frames=" << dataset.header.frame_count << '\n'
           << "#outputs=" << dataset.header.note_count << '\n'
           << "#midi_min=" << dataset.header.midi_min << '\n'
           << "#sample_rate=" << dataset.header.sample_rate << '\n'
           << "#hop_size=" << dataset.header.hop_size << '\n'
           << "#threshold=" << threshold << '\n';
    for (const auto& item : metadata) {
        stream << '#' << item.first << '=' << item.second << '\n';
    }
    stream << "frame";
    for (std::uint32_t output = 0; output < dataset.header.note_count; ++output) {
        stream << "\tscore_"
               << dataset.header.midi_min + static_cast<std::int32_t>(output);
    }
    for (std::uint32_t output = 0; output < dataset.header.note_count; ++output) {
        stream << "\tpred_"
               << dataset.header.midi_min + static_cast<std::int32_t>(output);
    }
    stream << '\n';
    for (std::uint64_t frame = 0; frame < dataset.header.frame_count; ++frame) {
        stream << frame;
        const auto base = static_cast<std::size_t>(frame) *
            dataset.header.note_count;
        for (std::uint32_t output = 0; output < dataset.header.note_count; ++output) {
            stream << '\t' << scores[base + output];
        }
        for (std::uint32_t output = 0; output < dataset.header.note_count; ++output) {
            stream << '\t' << static_cast<unsigned>(predictions[base + output]);
        }
        stream << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing prediction scores");
    }
}

[[nodiscard]] std::vector<std::uint8_t> apply_threshold(
    const std::vector<std::int32_t>& scores,
    const std::int32_t threshold) {
    std::vector<std::uint8_t> predictions(scores.size(), 0U);
    for (std::size_t index = 0; index < scores.size(); ++index) {
        predictions[index] = scores[index] >= threshold ? 1U : 0U;
    }
    return predictions;
}

[[nodiscard]] std::string head_ids(
    const tmgm::native::EnsembleBundle& bundle,
    const tmgm::native::EnsembleMemberHead head) {
    std::string result;
    for (const auto& member : bundle.members) {
        if (member.head != head) {
            continue;
        }
        if (!result.empty()) {
            result.push_back(',');
        }
        result += member.identifier;
    }
    return result;
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        const auto dataset = tmgm::native::load_dataset(arguments.dataset);
        const auto bundle = tmgm::native::load_ensemble_bundle(arguments.bundle);
        const auto prediction = tmgm::native::predict_ensemble_cpu(
            dataset, bundle, arguments.allow_legacy_feature_contract);
        const auto checksum =
            tmgm::native::ensemble_sha256_hex(bundle.bundle_checksum_sha256);
        const auto feature_fingerprint =
            tmgm::native::ensemble_sha256_hex(
                bundle.feature_fingerprint_sha256);
        const std::vector<std::pair<std::string, std::string>> common_metadata{
            {"bundle_format", bundle.format_version ==
                    tmgm::native::kEnsembleBundleFormatVersion
                ? "TMGM_NATIVE_ENSEMBLE_BUNDLE_V2"
                : "TMGM_NATIVE_ENSEMBLE_BUNDLE_V1"},
            {"bundle_checksum_sha256", checksum},
            {"feature_fingerprint_sha256", feature_fingerprint},
        };

        auto activity_metadata = common_metadata;
        activity_metadata.emplace_back("fusion", "mean");
        activity_metadata.emplace_back(
            "member_ids",
            head_ids(bundle, tmgm::native::EnsembleMemberHead::activity));
        activity_metadata.emplace_back(
            "quantization", std::to_string(bundle.activity.quantization));
        write_scores(
            arguments.activity_output,
            dataset,
            "activity",
            bundle.activity.ensemble_threshold,
            prediction.fused_activity_scores,
            prediction.activity_predictions,
            activity_metadata);

        auto onset_metadata = common_metadata;
        onset_metadata.emplace_back("fusion", "mean");
        onset_metadata.emplace_back(
            "member_ids",
            head_ids(bundle, tmgm::native::EnsembleMemberHead::onset));
        onset_metadata.emplace_back(
            "quantization", std::to_string(bundle.onset.quantization));
        write_scores(
            arguments.onset_output,
            dataset,
            "onset",
            bundle.onset.ensemble_threshold,
            prediction.fused_onset_scores,
            prediction.onset_predictions,
            onset_metadata);

        if (!arguments.member_output_directory.empty()) {
            std::filesystem::create_directories(
                arguments.member_output_directory);
            for (std::size_t index = 0; index < bundle.members.size(); ++index) {
                const auto& member = bundle.members[index];
                auto metadata = common_metadata;
                metadata.emplace_back("member_id", member.identifier);
                const auto predictions = apply_threshold(
                    prediction.raw_member_scores[index], member.score_threshold);
                write_scores(
                    arguments.member_output_directory /
                        (member.identifier + ".tsv"),
                    dataset,
                    head_name(member.head),
                    member.score_threshold,
                    prediction.raw_member_scores[index],
                    predictions,
                    metadata);
            }
        }

        std::cout << "frames=" << prediction.frame_count << '\n'
                  << "outputs=" << prediction.output_count << '\n'
                  << "members=" << bundle.members.size() << '\n'
                  << "bundle_checksum_sha256=" << checksum << '\n'
                  << "feature_fingerprint_sha256=" << feature_fingerprint << '\n'
                  << "activity_output=" << arguments.activity_output.string() << '\n'
                  << "onset_output=" << arguments.onset_output.string() << '\n';
        if (!arguments.member_output_directory.empty()) {
            std::cout << "member_output_dir="
                      << arguments.member_output_directory.string() << '\n';
        }
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
