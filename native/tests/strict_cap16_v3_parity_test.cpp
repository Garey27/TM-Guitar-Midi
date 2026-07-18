#include "tmgm/dataset.hpp"
#include "tmgm/strict_cap16_v3.hpp"

#include <array>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace allocation_probe {
std::atomic<std::uint64_t> count{0U};
}

void* operator new(const std::size_t size) {
    if (void* pointer = std::malloc(size == 0U ? 1U : size)) {
        allocation_probe::count.fetch_add(1U, std::memory_order_relaxed);
        return pointer;
    }
    throw std::bad_alloc();
}

void* operator new[](const std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* pointer) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

#if defined(__cpp_aligned_new)
void* operator new(const std::size_t size, const std::align_val_t alignment) {
    const auto alignment_bytes = static_cast<std::size_t>(alignment);
    void* pointer = nullptr;
#if defined(_MSC_VER)
    pointer = _aligned_malloc(size == 0U ? 1U : size, alignment_bytes);
#else
    if (posix_memalign(
            &pointer, alignment_bytes, size == 0U ? 1U : size) != 0) {
        pointer = nullptr;
    }
#endif
    if (pointer == nullptr) {
        throw std::bad_alloc();
    }
    allocation_probe::count.fetch_add(1U, std::memory_order_relaxed);
    return pointer;
}

void* operator new[](
    const std::size_t size,
    const std::align_val_t alignment) {
    return ::operator new(size, alignment);
}

void operator delete(void* pointer, std::align_val_t) noexcept {
#if defined(_MSC_VER)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}

void operator delete[](void* pointer, std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}

void operator delete(
    void* pointer,
    std::size_t,
    std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}

void operator delete[](
    void* pointer,
    std::size_t,
    std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}
#endif

namespace {

constexpr std::size_t kOutputs =
    tmgm::native::kStrictCap16V3OutputCount;

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

[[nodiscard]] std::vector<std::int32_t> read_scores(
    const std::filesystem::path& path,
    const std::uint64_t frame_count) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open score TSV: " + path.string());
    }
    std::vector<std::int32_t> scores;
    scores.reserve(static_cast<std::size_t>(frame_count) * kOutputs);
    std::string line;
    bool columns_seen = false;
    std::uint64_t expected_frame = 0U;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            throw std::runtime_error("blank line in score TSV");
        }
        if (line.front() == '#') {
            if (columns_seen) {
                throw std::runtime_error("metadata follows score columns");
            }
            continue;
        }
        if (!columns_seen) {
            columns_seen = true;
            continue;
        }
        if (expected_frame >= frame_count) {
            throw std::runtime_error("score TSV has too many rows");
        }
        std::size_t begin = 0U;
        const auto next_field = [&]() -> std::pair<const char*, const char*> {
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
            throw std::runtime_error("unexpected score frame index");
        }
        for (std::size_t output = 0U; output < kOutputs; ++output) {
            const auto field = next_field();
            scores.push_back(parse_i32(
                field.first, field.second, "score"));
        }
        ++expected_frame;
    }
    if (!columns_seen || expected_frame != frame_count) {
        throw std::runtime_error("score TSV has too few rows");
    }
    return scores;
}

void pack_frame_u32(
    const tmgm::native::NativeDataset& dataset,
    const std::uint64_t frame,
    std::vector<std::uint32_t>& destination) noexcept {
    const auto* source = dataset.feature_words.data() +
        static_cast<std::size_t>(frame) *
            dataset.header.feature_words_per_row;
    for (std::size_t word = 0U; word < destination.size(); ++word) {
        const auto source_word = source[word / 2U];
        destination[word] = static_cast<std::uint32_t>(
            word % 2U == 0U ? source_word : source_word >> 32U);
    }
}

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(value));
    return bits;
}

template <std::size_t MemberCount>
[[nodiscard]] float reference_mean(
    const std::array<std::vector<std::int32_t>, MemberCount>& raw,
    const std::array<std::int32_t, MemberCount>& thresholds,
    const std::array<float, MemberCount>& scales,
    const std::size_t score_index) noexcept {
    float sum = 0.0F;
    for (std::size_t member = 0U; member < MemberCount; ++member) {
        const float centered = static_cast<float>(raw[member][score_index]) -
            static_cast<float>(thresholds[member]);
        const float normalized = centered / scales[member];
        sum = sum + normalized;
    }
    return sum / static_cast<float>(MemberCount);
}

}  // namespace

int main(const int argc, char** argv) {
    namespace fs = std::filesystem;
    if (argc != 2) {
        std::cerr <<
            "usage: strict_cap16_v3_parity_test <strict-cap16-v3-package>\n";
        return 2;
    }
    try {
        const fs::path package = argv[1];
        const auto& manifest = tmgm::native::strict_cap16_v3_manifest();
        require(std::string(manifest.version) == "strict-cap16-v3",
                "native frozen manifest version differs");
        require(manifest.activity.ensemble_threshold == -169,
                "frozen activity threshold differs");
        require(manifest.onset.ensemble_threshold == -492,
                "frozen onset threshold differs");
        require(manifest.banks[1].packed_row ==
                    tmgm::native::StrictCap16V3PackedRowId::hcontrast &&
                manifest.banks[2].packed_row ==
                    tmgm::native::StrictCap16V3PackedRowId::hcontrast,
                "hcontrast logical banks do not share one packed row");

        auto coordinator =
            tmgm::native::StrictCap16V3Coordinator::load(package);
        require(coordinator.output_count() == kOutputs,
                "coordinator output count differs");

        const auto feature_root = package / "features";
        std::array<tmgm::native::NativeDataset,
                   tmgm::native::kStrictCap16V3PackedRowCount> datasets{
            tmgm::native::load_dataset(feature_root / "plain/2222.tmgd"),
            tmgm::native::load_dataset(feature_root / "hcontrast-d2/2222.tmgd"),
            tmgm::native::load_dataset(feature_root / "hprofile-d3/2222.tmgd"),
            tmgm::native::load_dataset(feature_root / "cattack-d3/2222.tmgd"),
        };
        const auto frame_count = datasets.front().header.frame_count;
        require(frame_count > 0U, "production parity track is empty");
        for (std::size_t row = 0U; row < datasets.size(); ++row) {
            require(datasets[row].header.frame_count == frame_count,
                    "cross-bank production frame count differs");
            require(datasets[row].header.note_count == kOutputs,
                    "cross-bank production output count differs");
            require(datasets[row].header.sample_rate == manifest.sample_rate &&
                        datasets[row].header.hop_size == manifest.hop_size,
                    "cross-bank production timebase differs");
        }
        // The second logical hcontrast dataset is intentionally byte-identical
        // in packed features; one row serves both model bundles at runtime.
        const auto hcontrast_d3 = tmgm::native::load_dataset(
            feature_root / "hcontrast-d3/2222.tmgd");
        require(hcontrast_d3.feature_words == datasets[1].feature_words,
                "hcontrast-d2/d3 packed rows are no longer identical");

        const auto score_root = package / "scores/2222";
        const auto expected_activity = read_scores(
            score_root / "activity-final.tsv", frame_count);
        const auto expected_onset = read_scores(
            score_root / "onset-primary.tsv", frame_count);

        constexpr std::array<const char*,
            tmgm::native::kStrictCap16V3ActivityMemberCount> activity_ids{
            "plain_c256", "plain_c512", "plain_c1024", "hc_c256",
            "hc_c512", "hprofile_c256", "cattack_c256",
        };
        constexpr std::array<std::int32_t,
            tmgm::native::kStrictCap16V3ActivityMemberCount>
            activity_thresholds{73, 136, 289, 68, 133, 75, 67};
        constexpr std::array<float,
            tmgm::native::kStrictCap16V3ActivityMemberCount>
            activity_scales{
                148.26F, 315.7938F, 628.6224F, 148.26F,
                312.8286F, 134.9166F, 149.7426F,
            };
        std::array<std::vector<std::int32_t>,
                   tmgm::native::kStrictCap16V3ActivityMemberCount>
            raw_activity;
        for (std::size_t member = 0U; member < raw_activity.size(); ++member) {
            raw_activity[member] = read_scores(
                score_root / "selected/activity" /
                    (std::string(activity_ids[member]) + ".tsv"),
                frame_count);
        }

        constexpr std::array<const char*,
            tmgm::native::kStrictCap16V3OnsetMemberCount> onset_ids{
            "c256_q1", "c256_q2", "c256_q4", "c256_q8",
            "c256_q4_seed19", "c512_q4", "hprofile_c256",
            "c1024_q4", "cattack_c256", "strict_cap16",
        };
        constexpr std::array<std::int32_t,
            tmgm::native::kStrictCap16V3OnsetMemberCount> onset_thresholds{
            151, 88, 80, 38, 60, 153, 60, 283, 87, 144,
        };
        constexpr std::array<float,
            tmgm::native::kStrictCap16V3OnsetMemberCount> onset_scales{
            85.9908F, 78.5778F, 63.7518F, 60.7866F, 60.7866F,
            131.9514F, 88.956F, 268.3506F, 78.5778F, 140.847F,
        };
        std::array<std::vector<std::int32_t>,
                   tmgm::native::kStrictCap16V3OnsetMemberCount>
            raw_onset;
        for (std::size_t member = 0U; member < raw_onset.size(); ++member) {
            raw_onset[member] = read_scores(
                score_root / "selected/onset" /
                    (std::string(onset_ids[member]) + ".tsv"),
                frame_count);
        }

        std::array<std::vector<std::uint32_t>,
                   tmgm::native::kStrictCap16V3PackedRowCount> packed;
        for (std::size_t row = 0U; row < packed.size(); ++row) {
            packed[row].resize(coordinator.required_packed_word_count(
                static_cast<tmgm::native::StrictCap16V3PackedRowId>(row)));
        }
        tmgm::native::StrictCap16V3FrameInput input;
        for (std::size_t row = 0U; row < packed.size(); ++row) {
            input.rows[row] = {packed[row].data(), packed[row].size()};
        }
        std::array<float, kOutputs> normalized_activity{};
        std::array<std::int32_t, kOutputs> quantized_activity{};
        std::array<std::uint8_t, kOutputs> activity_predictions{};
        std::array<float, kOutputs> normalized_onset{};
        std::array<std::int32_t, kOutputs> quantized_onset{};
        std::array<std::uint8_t, kOutputs> onset_predictions{};
        const tmgm::native::StrictCap16V3FrameOutputBuffers outputs{
            normalized_activity.data(), normalized_activity.size(),
            quantized_activity.data(), quantized_activity.size(),
            activity_predictions.data(), activity_predictions.size(),
            normalized_onset.data(), normalized_onset.size(),
            quantized_onset.data(), quantized_onset.size(),
            onset_predictions.data(), onset_predictions.size(),
        };

        auto bad_input = input;
        --bad_input.rows[0].word_count;
        require(coordinator.predict_frame(bad_input, outputs) ==
                    tmgm::native::StrictCap16V3PredictStatus::
                        wrong_packed_row_word_count,
                "coordinator accepted a wrong packed row size");

        const auto allocations_before =
            allocation_probe::count.load(std::memory_order_relaxed);
        for (std::uint64_t frame = 0U; frame < frame_count; ++frame) {
            for (std::size_t row = 0U; row < packed.size(); ++row) {
                pack_frame_u32(datasets[row], frame, packed[row]);
            }
            require(coordinator.predict_frame(input, outputs) ==
                        tmgm::native::StrictCap16V3PredictStatus::success,
                    "cross-bank per-frame prediction failed");
            const auto frame_base = static_cast<std::size_t>(frame) * kOutputs;
            for (std::size_t output = 0U; output < kOutputs; ++output) {
                const auto score_index = frame_base + output;
                const auto activity_reference = reference_mean(
                    raw_activity,
                    activity_thresholds,
                    activity_scales,
                    score_index);
                const auto onset_reference = reference_mean(
                    raw_onset,
                    onset_thresholds,
                    onset_scales,
                    score_index);
                require(float_bits(normalized_activity[output]) ==
                            float_bits(activity_reference),
                        "activity float32 global mean differs");
                require(float_bits(normalized_onset[output]) ==
                            float_bits(onset_reference),
                        "onset float32 global mean differs");
                require(quantized_activity[output] ==
                            expected_activity[score_index],
                        "activity quantized score differs from frozen TSV");
                require(quantized_onset[output] == expected_onset[score_index],
                        "onset quantized score differs from frozen TSV");
                require(activity_predictions[output] ==
                            static_cast<std::uint8_t>(
                                expected_activity[score_index] >= -169),
                        "activity frozen-threshold decision differs");
                require(onset_predictions[output] ==
                            static_cast<std::uint8_t>(
                                expected_onset[score_index] >= -492),
                        "onset frozen-threshold decision differs");
            }
        }
        const auto allocations_after =
            allocation_probe::count.load(std::memory_order_relaxed);
        require(allocations_after == allocations_before,
                "per-frame cross-bank path allocated memory");

        std::cout << "strict-cap16-v3 2222 parity passed: "
                  << frame_count
                  << " frames, exact float32/quantized/decision parity, "
                  << (allocations_after - allocations_before)
                  << " callback allocations\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
