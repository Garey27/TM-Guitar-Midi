#include "tmgm/dataset.hpp"
#include "tmgm/strict_cap16_v3_frontend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <stdexcept>
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
    void* pointer = nullptr;
#if defined(_MSC_VER)
    pointer = _aligned_malloc(
        size == 0U ? 1U : size, static_cast<std::size_t>(alignment));
#else
    if (posix_memalign(
            &pointer, static_cast<std::size_t>(alignment),
            size == 0U ? 1U : size) != 0) {
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

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] std::vector<float> read_float_audio(
    const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open resampled frontend fixture");
    }
    const auto end = stream.tellg();
    if (end <= 0 || static_cast<std::uint64_t>(end) % sizeof(float) != 0U) {
        throw std::runtime_error("resampled frontend fixture has invalid size");
    }
    std::vector<float> samples(
        static_cast<std::size_t>(end) / sizeof(float));
    stream.seekg(0, std::ios::beg);
    stream.read(
        reinterpret_cast<char*>(samples.data()),
        static_cast<std::streamsize>(samples.size() * sizeof(float)));
    if (!stream) {
        throw std::runtime_error("cannot read complete resampled fixture");
    }
    return samples;
}

[[nodiscard]] std::uint32_t bit_count(std::uint32_t value) noexcept {
    std::uint32_t result = 0U;
    while (value != 0U) {
        value &= value - 1U;
        ++result;
    }
    return result;
}

struct ParityContext {
    std::array<const tmgm::native::NativeDataset*,
               tmgm::native::kStrictCap16V3PackedRowCount> datasets{};
    std::array<std::uint64_t,
               tmgm::native::kStrictCap16V3PackedRowCount> mismatched_words{};
    std::array<std::uint64_t,
               tmgm::native::kStrictCap16V3PackedRowCount> mismatched_bits{};
    std::array<std::uint64_t,
               tmgm::native::kStrictCap16V3PackedRowCount> first_mismatch_frame{
        UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    std::array<std::size_t,
               tmgm::native::kStrictCap16V3PackedRowCount> first_mismatch_word{};
    std::array<std::uint32_t,
               tmgm::native::kStrictCap16V3PackedRowCount> first_mismatch_difference{};
    std::uint64_t frames = 0U;
    std::uint64_t hash = 1469598103934665603ULL;
    bool sequence_error = false;
    bool row_geometry_error = false;
};

void compare_frame(
    void* user,
    const std::uint64_t frame_index,
    const tmgm::native::StrictCap16V3FrameInput& rows) noexcept {
    auto& context = *static_cast<ParityContext*>(user);
    if (frame_index != context.frames) {
        context.sequence_error = true;
    }
    for (std::size_t bank = 0U; bank < context.datasets.size(); ++bank) {
        const auto& dataset = *context.datasets[bank];
        if (frame_index >= dataset.header.frame_count) {
            context.sequence_error = true;
            continue;
        }
        const auto expected_word_count =
            (static_cast<std::size_t>(dataset.header.feature_count) + 31U) /
            32U;
        if (rows.rows[bank].words == nullptr ||
            rows.rows[bank].word_count != expected_word_count) {
            context.row_geometry_error = true;
            continue;
        }
        const auto* expected = dataset.feature_words.data() +
            static_cast<std::size_t>(frame_index) *
                dataset.header.feature_words_per_row;
        for (std::size_t word = 0U; word < expected_word_count; ++word) {
            const auto expected_u64 = expected[word / 2U];
            const auto expected_u32 = static_cast<std::uint32_t>(
                word % 2U == 0U ? expected_u64 : expected_u64 >> 32U);
            const auto actual = rows.rows[bank].words[word];
            const auto difference = actual ^ expected_u32;
            if (difference != 0U) {
                if (context.first_mismatch_frame[bank] == UINT64_MAX) {
                    context.first_mismatch_frame[bank] = frame_index;
                    context.first_mismatch_word[bank] = word;
                    context.first_mismatch_difference[bank] = difference;
                }
                ++context.mismatched_words[bank];
                context.mismatched_bits[bank] += bit_count(difference);
            }
            context.hash ^= actual;
            context.hash *= 1099511628211ULL;
        }
    }
    ++context.frames;
}

[[nodiscard]] std::uint64_t total(
    const std::array<std::uint64_t,
                     tmgm::native::kStrictCap16V3PackedRowCount>& values) {
    std::uint64_t result = 0U;
    for (const auto value : values) {
        result += value;
    }
    return result;
}

}  // namespace

int main(const int argc, char** argv) {
    namespace fs = std::filesystem;
    if (argc != 2) {
        std::cerr <<
            "usage: strict_cap16_v3_frontend_parity_test <package-root>\n";
        return 2;
    }
    try {
        const fs::path package = argv[1];
        const auto frontend_root = package / "native-frontend";
        auto samples = read_float_audio(
            frontend_root / "2222-mono-22050.f32le");
        std::array<tmgm::native::NativeDataset,
                   tmgm::native::kStrictCap16V3PackedRowCount> datasets{
            tmgm::native::load_dataset(package / "features/plain/2222.tmgd"),
            tmgm::native::load_dataset(
                package / "features/hcontrast-d2/2222.tmgd"),
            tmgm::native::load_dataset(
                package / "features/hprofile-d3/2222.tmgd"),
            tmgm::native::load_dataset(
                package / "features/cattack-d3/2222.tmgd"),
        };
        const auto expected_frames = datasets.front().header.frame_count;
        require(expected_frames == 935U, "2222 frame fixture count differs");
        for (const auto& dataset : datasets) {
            require(dataset.header.frame_count == expected_frames,
                    "frontend bank frame count differs");
            require(dataset.header.sample_rate == 22050U &&
                        dataset.header.hop_size == 256U,
                    "frontend bank timebase differs");
        }

        auto frontend = tmgm::native::StrictCap16V3StreamingFrontend::load(
            frontend_root / "strict-cap16-v3.tmgmfront");
        ParityContext contiguous;
        for (std::size_t bank = 0U; bank < datasets.size(); ++bank) {
            contiguous.datasets[bank] = &datasets[bank];
        }
        require(frontend.process_block(
                    nullptr, 1U, compare_frame, &contiguous) ==
                    tmgm::native::StrictCap16V3FrontendStatus::null_audio,
                "frontend accepted null non-empty audio");
        require(frontend.process_block(
                    samples.data(), samples.size(), nullptr, &contiguous) ==
                    tmgm::native::StrictCap16V3FrontendStatus::
                        null_frame_callback,
                "frontend accepted null frame callback");

        const auto allocations_before =
            allocation_probe::count.load(std::memory_order_relaxed);
        require(frontend.process_block(
                    samples.data(), samples.size(), compare_frame, &contiguous) ==
                    tmgm::native::StrictCap16V3FrontendStatus::success,
                "contiguous frontend processing failed");
        const auto allocations_after_contiguous =
            allocation_probe::count.load(std::memory_order_relaxed);

        frontend.reset();
        ParityContext partitioned;
        for (std::size_t bank = 0U; bank < datasets.size(); ++bank) {
            partitioned.datasets[bank] = &datasets[bank];
        }
        constexpr std::array<std::size_t, 11> block_sizes{
            1U, 17U, 255U, 2U, 511U, 64U, 256U, 3U, 1024U, 127U, 409U,
        };
        std::size_t offset = 0U;
        std::size_t block = 0U;
        while (offset < samples.size()) {
            const auto count = std::min(
                block_sizes[block % block_sizes.size()],
                samples.size() - offset);
            require(frontend.process_block(
                        samples.data() + offset,
                        count,
                        compare_frame,
                        &partitioned) ==
                        tmgm::native::StrictCap16V3FrontendStatus::success,
                    "partitioned frontend processing failed");
            offset += count;
            ++block;
        }
        const auto allocations_after_partitioned =
            allocation_probe::count.load(std::memory_order_relaxed);

        constexpr std::array<const char*, 4> bank_names{
            "plain", "hcontrast", "hprofile", "cattack"};
        for (std::size_t bank = 0U; bank < bank_names.size(); ++bank) {
            std::cout << bank_names[bank]
                      << " mismatched_words="
                      << contiguous.mismatched_words[bank]
                      << " mismatched_bits="
                      << contiguous.mismatched_bits[bank]
                      << " first_frame="
                      << contiguous.first_mismatch_frame[bank]
                      << " first_word="
                      << contiguous.first_mismatch_word[bank]
                      << " xor="
                      << contiguous.first_mismatch_difference[bank] << '\n';
        }
        require(contiguous.frames == expected_frames &&
                    partitioned.frames == expected_frames,
                "frontend emitted wrong frame count");
        require(!contiguous.sequence_error && !partitioned.sequence_error &&
                    !contiguous.row_geometry_error &&
                    !partitioned.row_geometry_error,
                "frontend callback sequence/geometry differs");
        require(contiguous.hash == partitioned.hash,
                "frontend output depends on input block partition");
        require(contiguous.mismatched_words == partitioned.mismatched_words &&
                    contiguous.mismatched_bits == partitioned.mismatched_bits,
                "partitioned mismatch profile differs");
        require(total(contiguous.mismatched_bits) == 0U,
                "native frontend packed bits differ from Python TMGD");
        require(allocations_after_contiguous == allocations_before &&
                    allocations_after_partitioned == allocations_before,
                "frontend audio/frame path allocated memory");
        require(frontend.consumed_sample_count() == samples.size() &&
                    frontend.emitted_frame_count() == expected_frames,
                "frontend counters differ");

        std::cout << "strict-cap16-v3 frontend parity passed: "
                  << expected_frames
                  << " frames, four exact packed rows, block invariant, 0 allocations\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
