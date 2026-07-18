#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace tmgm::native {

constexpr std::uint32_t kDatasetHeaderBytes = 256;
constexpr std::uint32_t kDatasetWordBits = 64;

struct NativeDatasetHeader {
    std::uint64_t frame_count = 0;
    std::uint32_t feature_count = 0;
    std::uint32_t feature_words_per_row = 0;
    std::uint32_t note_count = 0;
    std::uint32_t label_words_per_row = 0;
    std::int32_t midi_min = 0;
    std::int32_t midi_max = 0;
    std::uint32_t sample_rate = 0;
    std::uint32_t hop_size = 0;
    std::uint64_t onset_index_count = 0;
    std::uint64_t features_offset = 0;
    std::uint64_t features_bytes = 0;
    std::uint64_t activity_offset = 0;
    std::uint64_t activity_bytes = 0;
    std::uint64_t onset_offset = 0;
    std::uint64_t onset_bytes = 0;
    std::uint64_t onset_indices_offset = 0;
    std::uint64_t onset_indices_bytes = 0;
    std::uint64_t seed = 0;
    std::array<std::uint8_t, 32> payload_sha256{};
    // Zero identifies a legacy v1 dataset with no semantic feature contract.
    // Non-zero v2 fingerprints bind frontend formula/order, context, and
    // binarizer identity independently of feature_count.
    std::array<std::uint8_t, 32> feature_fingerprint_sha256{};
};

// Exactly mirrors src/tmgm_rt/native_dataset.py. Each matrix is row-major and
// each binary column c occupies bit c % 64 of uint64 word c / 64.
struct NativeDataset {
    NativeDatasetHeader header;
    std::vector<std::uint64_t> feature_words;
    std::vector<std::uint64_t> activity_words;
    std::vector<std::uint64_t> onset_words;
    std::vector<std::uint32_t> onset_indices;

    [[nodiscard]] bool feature(std::uint64_t frame, std::uint32_t column) const;
    [[nodiscard]] bool activity(std::uint64_t frame, std::uint32_t note) const;
    [[nodiscard]] bool onset(std::uint64_t frame, std::uint32_t note) const;
    void set_feature(std::uint64_t frame, std::uint32_t column, bool value);
    void set_activity(std::uint64_t frame, std::uint32_t note, bool value);
    void set_onset(std::uint64_t frame, std::uint32_t note, bool value);
    void validate() const;
};

[[nodiscard]] NativeDataset load_dataset(
    const std::filesystem::path& path,
    bool verify_checksum = true);
void save_dataset(const std::filesystem::path& path, const NativeDataset& dataset);
[[nodiscard]] std::array<std::uint8_t, 32> calculate_payload_sha256(
    const NativeDataset& dataset);
[[nodiscard]] std::string sha256_hex(const std::array<std::uint8_t, 32>& digest);

}  // namespace tmgm::native
