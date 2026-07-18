#include "tmgm/dataset.hpp"
#include "tmgm/model.hpp"
#include "tmgm/tm_calibration.hpp"
#include "tmgm/tm_cuda.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Arguments {
    std::filesystem::path dataset;
    std::filesystem::path model;
    std::filesystem::path output;
    std::filesystem::path calibrated_model;
    double maximum_polyphony_ratio = 0.0;
    bool maximum_polyphony_ratio_set = false;
    bool allow_legacy_feature_contract = false;
};

[[noreturn]] void usage(const char* executable) {
    throw std::runtime_error(
        std::string("Usage: ") + executable +
        " <validation.tmgd> <model.tmgmmod> --output <scores.tsv>"
        " [--calibrate-model <calibrated.tmgmmod>]"
        " [--maximum-polyphony-ratio <ratio>]"
        " [--allow-legacy-feature-contract]");
}

[[nodiscard]] double parse_double(const std::string& value, const char* label) {
    std::size_t parsed = 0;
    const auto result = std::stod(value, &parsed);
    if (parsed != value.size()) {
        throw std::runtime_error(std::string("invalid ") + label + ": " + value);
    }
    return result;
}

[[nodiscard]] Arguments parse_arguments(const int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
    }
    Arguments result;
    result.dataset = argv[1];
    result.model = argv[2];
    for (int index = 3; index < argc; ++index) {
        const std::string option = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error(option + " requires a value");
            }
            return argv[index];
        };
        if (option == "--output") {
            result.output = value();
        } else if (option == "--calibrate-model") {
            result.calibrated_model = value();
        } else if (option == "--maximum-polyphony-ratio") {
            result.maximum_polyphony_ratio = parse_double(
                value(), "maximum polyphony ratio");
            result.maximum_polyphony_ratio_set = true;
        } else if (option == "--allow-legacy-feature-contract") {
            result.allow_legacy_feature_contract = true;
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }
    if (result.output.empty()) {
        throw std::runtime_error("--output is required");
    }
    if (result.calibrated_model.empty() && result.maximum_polyphony_ratio_set) {
        throw std::runtime_error(
            "--maximum-polyphony-ratio requires --calibrate-model");
    }
    return result;
}

void write_scores(
    const std::filesystem::path& path,
    const tmgm::native::NativeDataset& dataset,
    const tmgm::native::NativeTmModel& model,
    const tmgm::native::TmCudaPrediction& prediction) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot create score output: " + path.string());
    }
    const auto head = model.head == tmgm::native::TmModelHead::activity
        ? "activity"
        : "onset";
    stream << "#TMGM_SCORES_V1\n"
           << "#head=" << head << '\n'
           << "#frames=" << prediction.frame_count << '\n'
           << "#outputs=" << prediction.output_count << '\n'
           << "#midi_min=" << dataset.header.midi_min << '\n'
           << "#sample_rate=" << dataset.header.sample_rate << '\n'
           << "#hop_size=" << dataset.header.hop_size << '\n'
           << "#threshold=" << model.score_threshold << '\n'
           << "frame";
    for (std::uint32_t output = 0; output < prediction.output_count; ++output) {
        stream << "\tscore_" << dataset.header.midi_min + static_cast<int>(output);
    }
    for (std::uint32_t output = 0; output < prediction.output_count; ++output) {
        stream << "\tpred_" << dataset.header.midi_min + static_cast<int>(output);
    }
    stream << '\n';
    for (std::uint64_t frame = 0; frame < prediction.frame_count; ++frame) {
        stream << frame;
        const auto base = frame * prediction.output_count;
        for (std::uint32_t output = 0; output < prediction.output_count; ++output) {
            stream << '\t' << prediction.scores[base + output];
        }
        for (std::uint32_t output = 0; output < prediction.output_count; ++output) {
            stream << '\t' << static_cast<int>(prediction.predictions[base + output]);
        }
        stream << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing prediction scores");
    }
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        const auto dataset = tmgm::native::load_dataset(arguments.dataset);
        auto model = tmgm::native::load_tm_model(arguments.model);
        auto prediction = tmgm::native::predict_tm_cuda(
            dataset, model, arguments.allow_legacy_feature_contract);

        tmgm::native::TmScoreCalibration calibration;
        const auto previous_threshold = model.score_threshold;
        if (!arguments.calibrated_model.empty()) {
            const auto maximum_ratio = arguments.maximum_polyphony_ratio_set
                ? arguments.maximum_polyphony_ratio
                : tmgm::native::default_maximum_polyphony_ratio(model.head);
            calibration = tmgm::native::calibrate_score_threshold(
                dataset, model.head, prediction.scores, maximum_ratio);
            model.score_threshold = calibration.threshold;
            tmgm::native::apply_score_threshold(
                prediction.scores,
                model.score_threshold,
                prediction.predictions);
            if (!arguments.calibrated_model.parent_path().empty()) {
                std::filesystem::create_directories(
                    arguments.calibrated_model.parent_path());
            }
            tmgm::native::save_tm_model(arguments.calibrated_model, model);
        }

        write_scores(arguments.output, dataset, model, prediction);
        std::cout << "frames=" << prediction.frame_count << '\n'
                  << "outputs=" << prediction.output_count << '\n'
                  << "score_threshold=" << model.score_threshold << '\n'
                  << "saved_scores=" << arguments.output.string() << '\n';
        if (!arguments.calibrated_model.empty()) {
            std::cout << "previous_score_threshold=" << previous_threshold << '\n'
                      << "calibration_precision=" << calibration.precision << '\n'
                      << "calibration_recall=" << calibration.recall << '\n'
                      << "calibration_f1=" << calibration.f1 << '\n'
                      << "predicted_mean_polyphony="
                      << calibration.predicted_mean_polyphony << '\n'
                      << "target_mean_polyphony="
                      << calibration.target_mean_polyphony << '\n'
                      << "saved_calibrated_model="
                      << arguments.calibrated_model.string() << '\n';
        }
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
