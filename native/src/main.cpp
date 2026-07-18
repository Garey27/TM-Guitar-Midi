#include "tmgm/cuda_smoke.hpp"
#include "tmgm/dataset.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[nodiscard]] tmgm::native::NativeDataset make_demo_dataset() {
    tmgm::native::NativeDataset dataset;
    dataset.header.frame_count = 3;
    dataset.header.feature_count = 70;
    dataset.header.feature_words_per_row = 2;
    dataset.header.note_count = 49;
    dataset.header.label_words_per_row = 1;
    dataset.header.midi_min = 40;
    dataset.header.midi_max = 88;
    dataset.header.sample_rate = 22050;
    dataset.header.hop_size = 256;
    dataset.header.onset_index_count = 4;
    dataset.header.seed = 20260718;
    dataset.feature_words.resize(6, 0u);
    dataset.activity_words.resize(3, 0u);
    dataset.onset_words.resize(3, 0u);
    dataset.onset_indices = {0, 1, 1, 2};

    dataset.set_feature(0, 0, true);
    dataset.set_feature(0, 69, true);
    dataset.set_feature(1, 7, true);
    dataset.set_feature(2, 63, true);
    dataset.set_feature(2, 64, true);
    dataset.set_activity(0, 1, true);
    dataset.set_activity(1, 0, true);
    dataset.set_activity(1, 48, true);
    dataset.set_onset(0, 1, true);
    dataset.set_onset(2, 20, true);
    return dataset;
}

[[nodiscard]] std::uint64_t popcount(std::uint64_t value) noexcept {
    std::uint64_t count = 0;
    while (value != 0) {
        value &= value - 1u;
        ++count;
    }
    return count;
}

[[nodiscard]] std::uint64_t count_ones(const std::vector<std::uint64_t>& words) {
    std::uint64_t count = 0;
    for (const auto word : words) {
        count += popcount(word);
    }
    return count;
}

void print_usage(const char* executable) {
    std::cerr
        << "Usage:\n"
        << "  " << executable << " <dataset.tmgm> [--cuda-smoke]\n"
        << "  " << executable << " --make-demo <dataset.tmgm> [--cuda-smoke]\n";
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 2;
        }

        std::string dataset_path;
        bool make_demo = false;
        bool run_smoke = false;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--make-demo") {
                make_demo = true;
                if (++index >= argc) {
                    throw std::runtime_error("--make-demo requires an output path");
                }
                dataset_path = argv[index];
            } else if (argument == "--cuda-smoke") {
                run_smoke = true;
            } else if (dataset_path.empty()) {
                dataset_path = argument;
            } else {
                throw std::runtime_error("unexpected argument: " + argument);
            }
        }
        if (dataset_path.empty()) {
            throw std::runtime_error("dataset path is required");
        }

        if (make_demo) {
            tmgm::native::save_dataset(dataset_path, make_demo_dataset());
            std::cout << "created_demo=" << dataset_path << '\n';
        }

        const auto dataset = tmgm::native::load_dataset(dataset_path);
        const auto digest = tmgm::native::calculate_payload_sha256(dataset);
        const auto feature_ones = count_ones(dataset.feature_words);
        const auto activity_ones = count_ones(dataset.activity_words);
        const auto onset_ones = count_ones(dataset.onset_words);
        const auto feature_cells = dataset.header.frame_count * dataset.header.feature_count;
        const auto label_cells = dataset.header.frame_count * dataset.header.note_count;

        std::cout << "frames=" << dataset.header.frame_count << '\n'
                  << "features=" << dataset.header.feature_count << '\n'
                  << "feature_words_per_row=" << dataset.header.feature_words_per_row << '\n'
                  << "notes=" << dataset.header.note_count << '\n'
                  << "midi_range=" << dataset.header.midi_min << ':' << dataset.header.midi_max << '\n'
                  << "sample_rate=" << dataset.header.sample_rate << '\n'
                  << "hop_size=" << dataset.header.hop_size << '\n'
                  << "onset_training_indices=" << dataset.header.onset_index_count << '\n'
                  << "feature_payload_bytes=" << dataset.header.features_bytes << '\n'
                  << "activity_payload_bytes=" << dataset.header.activity_bytes << '\n'
                  << "onset_payload_bytes=" << dataset.header.onset_bytes << '\n'
                  << "onset_indices_bytes=" << dataset.header.onset_indices_bytes << '\n'
                  << "feature_ones=" << feature_ones << '\n'
                  << "activity_ones=" << activity_ones << '\n'
                  << "onset_ones=" << onset_ones << '\n'
                  << std::fixed << std::setprecision(6)
                  << "feature_density="
                  << (feature_cells == 0
                          ? 0.0
                          : static_cast<double>(feature_ones) / static_cast<double>(feature_cells))
                  << '\n'
                  << "activity_density="
                  << (label_cells == 0
                          ? 0.0
                          : static_cast<double>(activity_ones) / static_cast<double>(label_cells))
                  << '\n'
                  << "onset_density="
                  << (label_cells == 0
                          ? 0.0
                          : static_cast<double>(onset_ones) / static_cast<double>(label_cells))
                  << '\n'
                  << "payload_sha256=" << tmgm::native::sha256_hex(digest) << '\n'
                  << "cuda_compiled="
                  << (tmgm::native::cuda_backend_compiled() ? "true" : "false") << '\n';

        if (run_smoke) {
            const auto smoke = tmgm::native::run_cuda_smoke();
            std::cout << "cuda_devices=" << smoke.device_count << '\n'
                      << "cuda_device=" << smoke.device_name << '\n'
                      << "cuda_smoke=" << (smoke.passed ? "PASS" : "FAIL") << '\n'
                      << "cuda_detail=" << smoke.detail << '\n';
            if (!smoke.passed) {
                return 3;
            }
        }
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
