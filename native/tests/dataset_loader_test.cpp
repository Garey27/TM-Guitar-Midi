#include "tmgm/dataset.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

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

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const auto path = fs::temp_directory_path() / "tmgmdat_roundtrip_test.tmgm";
    const auto corrupt_path = fs::temp_directory_path() / "tmgmdat_corrupt_test.tmgm";

    try {
        tmgm::native::NativeDataset source;
        source.header.frame_count = 2;
        source.header.feature_count = 65;
        source.header.feature_words_per_row = 2;
        source.header.note_count = 49;
        source.header.label_words_per_row = 1;
        source.header.midi_min = 40;
        source.header.midi_max = 88;
        source.header.sample_rate = 22050;
        source.header.hop_size = 256;
        source.header.onset_index_count = 3;
        source.header.seed = 20260718;
        source.header.feature_fingerprint_sha256.fill(0x5aU);
        source.feature_words.resize(4, 0u);
        source.activity_words.resize(2, 0u);
        source.onset_words.resize(2, 0u);
        source.onset_indices = {0, 1, 1};
        source.set_feature(0, 0, true);
        source.set_feature(0, 64, true);
        source.set_feature(1, 17, true);
        source.set_activity(0, 48, true);
        source.set_activity(1, 0, true);
        source.set_onset(0, 48, true);

        tmgm::native::save_dataset(path, source);
        const auto loaded = tmgm::native::load_dataset(path);
        require(loaded.header.frame_count == 2, "wrong frame count");
        require(loaded.header.feature_count == 65, "wrong feature count");
        require(loaded.header.note_count == 49, "wrong note count");
        require(loaded.header.feature_fingerprint_sha256 ==
                    source.header.feature_fingerprint_sha256,
                "feature-semantics fingerprint was lost");
        require(loaded.feature(0, 0), "feature 0/0 was lost");
        require(loaded.feature(0, 64), "feature 0/64 was lost");
        require(loaded.feature(1, 17), "feature 1/17 was lost");
        require(!loaded.feature(1, 18), "unexpected feature bit");
        require(loaded.activity(0, 48), "activity 0/48 was lost");
        require(loaded.activity(1, 0), "activity 1/0 was lost");
        require(loaded.onset(0, 48), "onset 0/48 was lost");
        require(loaded.onset_indices == source.onset_indices, "onset indices were lost");

        fs::copy_file(path, corrupt_path, fs::copy_options::overwrite_existing);
        {
            std::fstream file(corrupt_path, std::ios::binary | std::ios::in | std::ios::out);
            require(static_cast<bool>(file), "could not open corrupt fixture");
            file.seekg(tmgm::native::kDatasetHeaderBytes);
            char byte = 0;
            file.read(&byte, 1);
            file.clear();
            file.seekp(tmgm::native::kDatasetHeaderBytes);
            byte ^= 1;
            file.write(&byte, 1);
        }

        bool rejected_corruption = false;
        try {
            static_cast<void>(tmgm::native::load_dataset(corrupt_path));
        } catch (const std::runtime_error&) {
            rejected_corruption = true;
        }
        require(rejected_corruption, "SHA-256 corruption was not rejected");

        tmgm::native::save_dataset(path, source);
        {
            std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
            require(static_cast<bool>(file), "could not open fingerprint fixture");
            file.seekp(176);
            const std::array<char, 32> zeros{};
            file.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
        }
        require_throws(
            [&] { static_cast<void>(tmgm::native::load_dataset(path)); },
            "v2 dataset accepted an all-zero feature-semantics fingerprint");

        fs::remove(path);
        fs::remove(corrupt_path);
        std::cout << "TMGMDAT loader tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        fs::remove(path);
        fs::remove(corrupt_path);
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
