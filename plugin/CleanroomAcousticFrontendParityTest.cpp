#include "CleanroomAcousticFrontend.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>

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

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete(void* pointer, const std::size_t) noexcept {
    ::operator delete(pointer);
}
void operator delete[](void* pointer, const std::size_t) noexcept {
    ::operator delete(pointer);
}

#if defined(__cpp_aligned_new)
void* operator new(const std::size_t size, const std::align_val_t alignment) {
#if defined(_MSC_VER)
    void* pointer = _aligned_malloc(
        size == 0U ? 1U : size, static_cast<std::size_t>(alignment));
#else
    void* pointer = nullptr;
    if (posix_memalign(&pointer, static_cast<std::size_t>(alignment),
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
    const std::size_t size, const std::align_val_t alignment) {
    return ::operator new(size, alignment);
}

void operator delete(void* pointer, const std::align_val_t) noexcept {
#if defined(_MSC_VER)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}
void operator delete[](void* pointer, const std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}
void operator delete(
    void* pointer, const std::size_t, const std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}
void operator delete[](
    void* pointer, const std::size_t, const std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}
#endif

namespace {

constexpr std::size_t kFixtureFrameCount = 24U;
constexpr std::size_t kFixtureSampleCount =
    1U + (kFixtureFrameCount - 1U) *
        tmgm::preview::kCleanroomAcousticHopSize;
using Frame = tmgm::preview::CleanroomAcousticFrame;

void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

[[nodiscard]] float fixture_sample(
    const std::size_t index, std::uint32_t& state) noexcept {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    const auto word = static_cast<std::int32_t>((state >> 16U) & 0xffffU) -
        32768;
    float sample = static_cast<float>(word) / 32768.0F * 0.12F;
    if (index == 0U) sample += 0.50F;
    if (index == 1024U) sample -= 0.375F;
    if (index == 3072U) sample += 0.25F;
    return sample;
}

[[nodiscard]] std::array<Frame, kFixtureFrameCount> render_fixture(
    tmgm::preview::CleanroomAcousticFrontend& frontend) {
    std::array<Frame, kFixtureFrameCount> frames{};
    std::size_t frame_count = 0U;
    std::uint32_t random_state = 0x12345678U;
    Frame frame;
    for (std::size_t sample = 0U; sample < kFixtureSampleCount; ++sample) {
        if (frontend.push_sample(
                fixture_sample(sample, random_state), frame)) {
            require(frame_count < frames.size(), "frontend emitted extra frame");
            frames[frame_count++] = frame;
        }
    }
    require(frontend.healthy(), "frontend reported PocketFFT failure");
    require(frame_count == frames.size(), "frontend emitted wrong frame count");
    return frames;
}

[[nodiscard]] std::array<Frame, kFixtureFrameCount> read_fixture(
    const char* path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open cleanroom acoustic fixture");
    }
    std::array<Frame, kFixtureFrameCount> result{};
    std::string line;
    std::size_t row = 0U;
    while (std::getline(stream, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        if (row >= result.size()) {
            throw std::runtime_error("cleanroom fixture has extra rows");
        }
        const char* begin = line.data();
        const char* end = begin + line.size();
        const auto next_field = [&]() -> std::string_view {
            const char* field_end = begin;
            while (field_end != end && *field_end != '\t') ++field_end;
            const std::string_view field(begin,
                static_cast<std::size_t>(field_end - begin));
            begin = field_end == end ? end : field_end + 1;
            return field;
        };
        const auto parse_u64 = [](const std::string_view text) {
            std::uint64_t value = 0U;
            const auto parsed = std::from_chars(
                text.data(), text.data() + text.size(), value);
            if (parsed.ec != std::errc{} ||
                parsed.ptr != text.data() + text.size()) {
                throw std::runtime_error("invalid integer in acoustic fixture");
            }
            return value;
        };
        result[row].frame_index = parse_u64(next_field());
        result[row].sample_index = parse_u64(next_field());
        for (float& value : result[row].attack_energy) {
            const auto field = next_field();
            std::string owned(field);
            char* parsed_end = nullptr;
            value = std::strtof(owned.c_str(), &parsed_end);
            if (parsed_end != owned.c_str() + owned.size() ||
                !std::isfinite(value)) {
                throw std::runtime_error("invalid float in acoustic fixture");
            }
        }
        if (begin != end) {
            throw std::runtime_error("cleanroom fixture row has extra fields");
        }
        ++row;
    }
    require(row == result.size(), "cleanroom fixture has too few rows");
    return result;
}

[[nodiscard]] bool near_reference(
    const float actual, const float expected) noexcept {
    constexpr float absolute_tolerance = 2.0e-6F;
    constexpr float relative_tolerance = 2.0e-4F;
    return std::abs(actual - expected) <=
        absolute_tolerance + relative_tolerance * std::abs(expected);
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "usage: CleanroomAcousticFrontendParityTest <fixture.tsv>\n";
            return 2;
        }
        const auto expected = read_fixture(argv[1]);
        tmgm::preview::CleanroomAcousticFrontend frontend;

        const auto allocations_before =
            allocation_probe::count.load(std::memory_order_relaxed);
        const auto actual = render_fixture(frontend);
        const auto allocations_after =
            allocation_probe::count.load(std::memory_order_relaxed);
        require(allocations_after == allocations_before,
                "cleanroom acoustic sample/frame path allocated memory");

        float maximum_absolute_error = 0.0F;
        for (std::size_t row = 0U; row < actual.size(); ++row) {
            require(actual[row].frame_index == expected[row].frame_index,
                    "frame index differs from Python fixture");
            require(actual[row].sample_index == expected[row].sample_index,
                    "sample grid differs from Python fixture");
            for (std::size_t pitch = 0U;
                 pitch < tmgm::preview::kCleanroomAcousticPitchCount; ++pitch) {
                const float value = actual[row].attack_energy[pitch];
                const float reference = expected[row].attack_energy[pitch];
                require(std::isfinite(value) && value >= 0.0F,
                        "attack output is invalid");
                maximum_absolute_error = std::max(
                    maximum_absolute_error, std::abs(value - reference));
                if (!near_reference(value, reference)) {
                    std::cerr << "attack mismatch frame=" << row
                              << " midi="
                              << (tmgm::preview::kCleanroomAcousticMidiMin +
                                  static_cast<int>(pitch))
                              << " actual=" << value
                              << " expected=" << reference << '\n';
                    throw std::runtime_error(
                        "attack energy differs from Python frontend");
                }
            }
        }

        frontend.reset();
        const auto repeated = render_fixture(frontend);
        for (std::size_t row = 0U; row < actual.size(); ++row) {
            for (std::size_t pitch = 0U;
                 pitch < tmgm::preview::kCleanroomAcousticPitchCount; ++pitch) {
                require(std::bit_cast<std::uint32_t>(
                            actual[row].attack_energy[pitch]) ==
                        std::bit_cast<std::uint32_t>(
                            repeated[row].attack_energy[pitch]),
                        "reset did not reproduce exact native output");
            }
        }

        std::cout << "cleanroom acoustic parity passed: "
                  << actual.size() << " frames, max abs error "
                  << maximum_absolute_error << ", 0 callback allocations\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "cleanroom acoustic parity failure: "
                  << exception.what() << '\n';
        return 1;
    }
}
