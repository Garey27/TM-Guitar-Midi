#include "tmgm/dataset.hpp"
#include "tmgm/tm_cuda.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

#if defined(TMGM_ENABLE_OUTPUT_ORDER_ABLATION)
constexpr const char* kOutputOrderAblationUsage =
    " [--rotate-output-update-order]";
#else
constexpr const char* kOutputOrderAblationUsage = "";
#endif

struct Arguments {
    std::filesystem::path dataset;
    std::filesystem::path output;
    std::filesystem::path model;
    std::filesystem::path validation;
    tmgm::native::TargetHead head = tmgm::native::TargetHead::activity;
    tmgm::native::TmCudaConfig config;
    bool specificity_set = false;
    bool negative_samples_set = false;
    bool allow_legacy_feature_contract = false;
};

[[noreturn]] void usage(const char* executable) {
    throw std::runtime_error(
        std::string("Usage: ") + executable +
        " <dataset.tmgd> --output <scores.tsv> [--model <model.tmgmmod>]"
        " [--validation <validation.tmgd>]"
        " [--head activity|onset]"
        " [--epochs 30] [--clauses 256] [--threshold 128]"
        " [--specificity 5] [--negative-samples 8]"
        " [--onset-sustain-hard-negatives]"
        " [--onset-sustain-hard-negative-probability P]"
        " [--onset-sustain-hard-negative-weight-only]"
        " [--max-literals 64] [--samples-per-launch 128 (1..512)]"
        " [--validation-patience N (0=off)] [--seed N]"
        " [--allow-legacy-feature-contract]" +
        kOutputOrderAblationUsage);
}

[[nodiscard]] std::uint32_t parse_u32(const std::string& value, const char* label) {
    std::size_t parsed = 0;
    const auto result = std::stoull(value, &parsed);
    if (parsed != value.size() || result > 0xffffffffull) {
        throw std::runtime_error(std::string("invalid ") + label + ": " + value);
    }
    return static_cast<std::uint32_t>(result);
}

[[nodiscard]] std::uint64_t parse_u64(const std::string& value, const char* label) {
    std::size_t parsed = 0;
    const auto result = std::stoull(value, &parsed);
    if (parsed != value.size()) {
        throw std::runtime_error(std::string("invalid ") + label + ": " + value);
    }
    return result;
}

[[nodiscard]] float parse_float(const std::string& value, const char* label) {
    std::size_t parsed = 0;
    const auto result = std::stof(value, &parsed);
    if (parsed != value.size()) {
        throw std::runtime_error(std::string("invalid ") + label + ": " + value);
    }
    return result;
}

[[nodiscard]] Arguments parse_arguments(const int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
    }
    Arguments result;
    result.dataset = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string option = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error(option + " requires a value");
            }
            return argv[index];
        };
        if (option == "--output") {
            result.output = value();
        } else if (option == "--model") {
            result.model = value();
        } else if (option == "--validation") {
            result.validation = value();
        } else if (option == "--head") {
            const auto selected = value();
            if (selected == "activity") {
                result.head = tmgm::native::TargetHead::activity;
            } else if (selected == "onset") {
                result.head = tmgm::native::TargetHead::onset;
            } else {
                throw std::runtime_error("--head must be activity or onset");
            }
        } else if (option == "--epochs") {
            result.config.epochs = parse_u32(value(), "epoch count");
        } else if (option == "--clauses") {
            result.config.clauses = parse_u32(value(), "clause count");
        } else if (option == "--threshold") {
            result.config.threshold = static_cast<std::int32_t>(
                parse_u32(value(), "TM threshold"));
        } else if (option == "--specificity") {
            result.config.specificity = parse_float(value(), "specificity");
            result.specificity_set = true;
        } else if (option == "--negative-samples") {
            result.config.negative_samples = parse_float(value(), "negative samples");
            result.negative_samples_set = true;
        } else if (option == "--onset-sustain-hard-negatives") {
            result.config.onset_sustain_hard_negatives = true;
        } else if (option == "--onset-sustain-hard-negative-probability") {
            result.config.onset_sustain_hard_negatives = true;
            result.config.onset_sustain_hard_negative_probability =
                parse_float(value(), "onset sustain hard-negative probability");
        } else if (option == "--onset-sustain-hard-negative-weight-only") {
            result.config.onset_sustain_hard_negatives = true;
            result.config.onset_sustain_hard_negative_weight_only = true;
        } else if (option == "--max-literals") {
            result.config.max_included_literals = parse_u32(value(), "max literals");
        } else if (option == "--samples-per-launch") {
            result.config.samples_per_launch = parse_u32(value(), "samples per launch");
            if (result.config.samples_per_launch == 0U ||
                result.config.samples_per_launch >
                    tmgm::native::kTmCudaMaxSamplesPerLaunch) {
                throw std::runtime_error(
                    "samples per launch must be in [1, 512]");
            }
        } else if (option == "--validation-patience") {
            result.config.validation_patience =
                parse_u32(value(), "validation patience");
        } else if (option == "--seed") {
            result.config.seed = parse_u64(value(), "seed");
#if defined(TMGM_ENABLE_OUTPUT_ORDER_ABLATION)
        } else if (option == "--rotate-output-update-order") {
            result.config.rotate_output_update_order = true;
#endif
        } else if (option == "--allow-legacy-feature-contract") {
            result.allow_legacy_feature_contract = true;
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }
    if (result.output.empty()) {
        throw std::runtime_error("--output is required");
    }
    if (result.model.empty()) {
        result.model = result.output;
        result.model.replace_extension(".tmgmmod");
    }
    if (result.config.validation_patience != 0U && result.validation.empty()) {
        throw std::runtime_error(
            "--validation-patience requires --validation");
    }
    if (result.head == tmgm::native::TargetHead::onset) {
        if (!result.specificity_set) {
            result.config.specificity = 4.0f;
        }
        if (!result.negative_samples_set) {
            result.config.negative_samples = 4.0f;
        }
        if (result.config.seed == 20260718ull) {
            result.config.seed += 10000ull;
        }
    } else if (result.config.onset_sustain_hard_negatives) {
        throw std::runtime_error(
            "--onset-sustain-hard-negatives requires --head onset");
    }
    return result;
}

void write_scores(
    const std::filesystem::path& path,
    const tmgm::native::NativeDataset& dataset,
    const tmgm::native::TargetHead head,
    const tmgm::native::TmCudaResult& result) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot create score output: " + path.string());
    }
    const auto threshold = result.model.score_threshold;
    stream << "#TMGM_SCORES_V1\n"
           << "#head=" << (head == tmgm::native::TargetHead::activity ? "activity" : "onset") << '\n'
           << "#frames=" << result.frame_count << '\n'
           << "#outputs=" << result.output_count << '\n'
           << "#midi_min=" << dataset.header.midi_min << '\n'
           << "#sample_rate=" << dataset.header.sample_rate << '\n'
           << "#hop_size=" << dataset.header.hop_size << '\n'
           << "#threshold=" << threshold << '\n'
           << "#checkpoint_epoch=" << result.model.training.epochs_trained << '\n'
           << "frame";
    for (std::uint32_t output = 0; output < result.output_count; ++output) {
        stream << "\tscore_" << dataset.header.midi_min + static_cast<int>(output);
    }
    for (std::uint32_t output = 0; output < result.output_count; ++output) {
        stream << "\tpred_" << dataset.header.midi_min + static_cast<int>(output);
    }
    stream << '\n';
    for (std::uint64_t frame = 0; frame < result.frame_count; ++frame) {
        stream << frame;
        const auto base = frame * result.output_count;
        for (std::uint32_t output = 0; output < result.output_count; ++output) {
            stream << '\t' << result.scores[base + output];
        }
        for (std::uint32_t output = 0; output < result.output_count; ++output) {
            stream << '\t' << static_cast<int>(result.predictions[base + output]);
        }
        stream << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing score output");
    }
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        const auto dataset = tmgm::native::load_dataset(arguments.dataset);
        std::optional<tmgm::native::NativeDataset> validation;
        if (!arguments.validation.empty()) {
            validation.emplace(tmgm::native::load_dataset(arguments.validation));
        }
        std::cout << "cuda_tm_supported="
                  << (tmgm::native::cuda_tm_supported() ? "true" : "false") << '\n'
                  << "head="
                  << (arguments.head == tmgm::native::TargetHead::activity
                          ? "activity"
                          : "onset")
                  << '\n'
                  << "frames=" << dataset.header.frame_count << '\n'
                  << "features=" << dataset.header.feature_count << '\n'
                  << "outputs=" << dataset.header.note_count << '\n'
                  << "clauses=" << arguments.config.clauses << '\n'
                  << "epochs=" << arguments.config.epochs << '\n';
        if (arguments.head == tmgm::native::TargetHead::onset) {
            std::cout << "onset_sustain_hard_negatives="
                      << (arguments.config.onset_sustain_hard_negatives
                              ? "true"
                              : "false")
                      << '\n';
            if (arguments.config.onset_sustain_hard_negatives) {
                std::cout << "onset_sustain_hard_negative_probability="
                          << arguments.config
                                 .onset_sustain_hard_negative_probability
                          << '\n';
                std::cout << "onset_sustain_hard_negative_weight_only="
                          << (arguments.config
                                      .onset_sustain_hard_negative_weight_only
                                  ? "true"
                                  : "false")
                          << '\n';
            }
        }
        if (validation.has_value()) {
            std::cout << "validation_frames="
                      << validation->header.frame_count << '\n';
            if (arguments.config.validation_patience != 0U) {
                std::cout << "validation_patience="
                          << arguments.config.validation_patience << '\n';
            }
        }
        std::cout << std::flush;

        const auto result = tmgm::native::train_tm_cuda(
            dataset,
            arguments.head,
            arguments.config,
            validation.has_value() ? &*validation : nullptr,
            arguments.allow_legacy_feature_contract);
        write_scores(arguments.output, dataset, arguments.head, result);
        if (!arguments.model.parent_path().empty()) {
            std::filesystem::create_directories(arguments.model.parent_path());
        }
        tmgm::native::save_tm_model(arguments.model, result.model);
        const auto& final = result.epochs.back();
        if (validation.has_value()) {
            const auto& best = result.epochs.at(
                static_cast<std::size_t>(result.best_validation_epoch - 1U));
            std::cout << "best_validation_epoch="
                      << result.best_validation_epoch << '\n'
                      << "best_validation_f1=" << best.validation_f1 << '\n'
                      << "best_validation_precision="
                      << best.validation_precision << '\n'
                      << "best_validation_recall=" << best.validation_recall << '\n'
                      << "last_train_f1=" << final.f1 << '\n'
                      << "score_threshold=" << result.model.score_threshold << '\n';
        } else {
            std::cout << "final_f1=" << final.f1 << '\n'
                      << "final_precision=" << final.precision << '\n'
                      << "final_recall=" << final.recall << '\n'
                      << "score_threshold=" << final.score_threshold << '\n';
        }
        if (arguments.config.validation_patience != 0U) {
            std::cout << "early_stopped="
                      << (result.early_stopped ? "true" : "false") << '\n'
                      << "epochs_executed=" << result.epochs_executed << '\n';
        }
        std::cout << "saved_scores=" << arguments.output.string() << '\n';
        std::cout << "saved_model=" << arguments.model.string() << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
