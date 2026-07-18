#include "tmgm/causal_resampler.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
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

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

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

constexpr double kPi = 3.141592653589793238462643383279502884;

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] std::uint64_t expected_output_count(
    const std::uint64_t input_count,
    const std::uint32_t input_rate) {
    return (input_count * tmgm::native::kResamplerOutputSampleRate +
            input_rate - 1U) /
        input_rate;
}

[[nodiscard]] std::uint64_t hash_samples(
    const std::vector<float>& samples) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const float sample : samples) {
        std::uint32_t bits = 0U;
        std::memcpy(&bits, &sample, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }
    return hash;
}

[[nodiscard]] std::vector<float> process_partitioned(
    tmgm::native::CausalMonoResampler22050& resampler,
    const std::vector<float>& input,
    const std::vector<std::size_t>& block_sizes) {
    require(!block_sizes.empty(), "empty resampler partition pattern");
    std::vector<float> result;
    result.reserve(static_cast<std::size_t>(expected_output_count(
        input.size(), resampler.input_sample_rate())));
    std::size_t offset = 0U;
    std::size_t block = 0U;
    while (offset < input.size()) {
        const auto count = std::min(
            block_sizes[block % block_sizes.size()], input.size() - offset);
        require(count != 0U, "zero resampler partition");
        tmgm::native::ResampledMonoBlock output;
        const auto allocations_before =
            allocation_probe::count.load(std::memory_order_relaxed);
        const auto status = resampler.process_block(
            input.data() + offset, count, output);
        const auto allocations_after =
            allocation_probe::count.load(std::memory_order_relaxed);
        require(status == tmgm::native::CausalResamplerStatus::success,
                "resampler processing failed");
        require(allocations_before == allocations_after,
                "resampler realtime path allocated memory");
        require(output.sample_count <= count,
                "downsampler emitted more samples than input");
        result.insert(
            result.end(), output.samples, output.samples + output.sample_count);
        offset += count;
        ++block;
    }
    return result;
}

[[nodiscard]] std::vector<float> make_probe(
    const std::uint32_t sample_rate,
    const std::size_t sample_count) {
    std::vector<float> input(sample_count, 0.0F);
    for (std::size_t index = 0U; index < input.size(); ++index) {
        const double time = static_cast<double>(index) / sample_rate;
        const double value =
            0.37 * std::sin(2.0 * kPi * 997.0 * time) +
            0.19 * std::sin(2.0 * kPi * 7311.0 * time) +
            (index % 997U == 0U ? 0.23 : 0.0);
        input[index] = static_cast<float>(value);
    }
    return input;
}

[[nodiscard]] double steady_sine_gain(
    const std::uint32_t input_rate,
    const double frequency) {
    const std::size_t input_count = static_cast<std::size_t>(input_rate) * 2U;
    constexpr double amplitude = 0.5;
    std::vector<float> input(input_count);
    for (std::size_t index = 0U; index < input.size(); ++index) {
        input[index] = static_cast<float>(amplitude * std::sin(
            2.0 * kPi * frequency * static_cast<double>(index) / input_rate));
    }
    auto resampler =
        tmgm::native::CausalMonoResampler22050::prepare(input_rate, 509U);
    const auto output = process_partitioned(
        resampler, input, {1U, 509U, 17U, 256U, 3U, 127U});
    constexpr std::size_t analysis_count =
        tmgm::native::kResamplerOutputSampleRate;
    require(output.size() >= analysis_count,
            "tone resampler output is too short");
    const auto first = output.size() - analysis_count;
    double square_sum = 0.0;
    for (std::size_t index = first; index < output.size(); ++index) {
        const double value = output[index];
        square_sum += value * value;
    }
    const double rms = std::sqrt(square_sum / analysis_count);
    return rms * std::sqrt(2.0) / amplitude;
}

void test_rate(const std::uint32_t input_rate) {
    constexpr std::size_t probe_count = 4097U;
    const auto input = make_probe(input_rate, probe_count);

    auto contiguous =
        tmgm::native::CausalMonoResampler22050::prepare(
            input_rate, input.size());
    const auto contiguous_output = process_partitioned(
        contiguous, input, {input.size()});
    require(contiguous_output.size() ==
                expected_output_count(input.size(), input_rate),
            "contiguous output count differs");

    auto irregular =
        tmgm::native::CausalMonoResampler22050::prepare(input_rate, 257U);
    const auto irregular_output = process_partitioned(
        irregular, input,
        {1U, 2U, 17U, 255U, 3U, 64U, 257U, 5U, 128U});
    require(irregular_output == contiguous_output,
            "resampler output depends on irregular partitions");

    auto single_sample =
        tmgm::native::CausalMonoResampler22050::prepare(input_rate, 1U);
    const auto single_output = process_partitioned(
        single_sample, input, {1U});
    require(single_output == contiguous_output,
            "resampler output depends on one-sample partitions");
    require(single_sample.consumed_input_sample_count() == input.size() &&
                single_sample.produced_output_sample_count() ==
                    contiguous_output.size(),
            "resampler counters differ");

    irregular.reset();
    const auto reset_output = process_partitioned(
        irregular, input, {257U, 1U, 31U, 2U, 129U});
    require(reset_output == contiguous_output,
            "resampler reset did not restore phase/history");

    std::vector<float> impulse(700U, 0.0F);
    impulse[0] = 1.0F;
    auto impulse_resampler =
        tmgm::native::CausalMonoResampler22050::prepare(
            input_rate, impulse.size());
    const auto impulse_output = process_partitioned(
        impulse_resampler, impulse, {impulse.size()});
    const auto peak = static_cast<std::size_t>(std::distance(
        impulse_output.begin(),
        std::max_element(
            impulse_output.begin(), impulse_output.end(),
            [](const float left, const float right) {
                return std::abs(left) < std::abs(right);
            })));
    const auto expected_peak = static_cast<std::size_t>(std::llround(
        impulse_resampler.latency_output_samples()));
    require(peak == expected_peak,
            "resampler impulse/group-delay phase differs");
    require(impulse_output.size() ==
                expected_output_count(impulse.size(), input_rate),
            "impulse output count differs");

    std::vector<float> dc(input_rate / 4U, 1.0F);
    auto dc_resampler =
        tmgm::native::CausalMonoResampler22050::prepare(input_rate, 251U);
    const auto dc_output = process_partitioned(
        dc_resampler, dc, {1U, 251U, 7U, 128U});
    require(dc_output.size() > 2048U, "DC output is too short");
    double dc_mean = 0.0;
    for (auto iterator = dc_output.end() - 2048;
         iterator != dc_output.end(); ++iterator) {
        dc_mean += *iterator;
    }
    dc_mean /= 2048.0;
    require(std::abs(dc_mean - 1.0) < 2.0e-6,
            "resampler DC gain differs");

    const double low_gain = steady_sine_gain(input_rate, 1000.0);
    const double edge_gain = steady_sine_gain(input_rate, 10000.0);
    const double stop_gain = steady_sine_gain(input_rate, 12000.0);
    require(std::abs(low_gain - 1.0) < 1.0e-4,
            "resampler low-passband gain differs");
    require(std::abs(edge_gain - 1.0) < 2.0e-3,
            "resampler passband-edge gain differs");
    require(stop_gain < 2.0e-4,
            "resampler stopband/alias attenuation is insufficient");

    std::vector<float> exceptional(1024U, 0.25F);
    exceptional[3] = std::numeric_limits<float>::quiet_NaN();
    exceptional[17] = std::numeric_limits<float>::infinity();
    exceptional[33] = -std::numeric_limits<float>::infinity();
    exceptional[65] = std::numeric_limits<float>::max();
    auto finite_resampler =
        tmgm::native::CausalMonoResampler22050::prepare(
            input_rate, exceptional.size());
    const auto finite_output = process_partitioned(
        finite_resampler, exceptional, {exceptional.size()});
    require(std::all_of(
                finite_output.begin(), finite_output.end(),
                [](const float value) { return std::isfinite(value); }),
            "resampler emitted a non-finite sample");

    std::cout << "rate=" << input_rate
              << " frames=" << contiguous_output.size()
              << " hash=" << hash_samples(contiguous_output)
              << " delay_out=" << contiguous.latency_output_samples()
              << " gain_1k=" << low_gain
              << " gain_10k=" << edge_gain
              << " stop_12k_db="
              << 20.0 * std::log10(std::max(stop_gain, 1.0e-30))
              << '\n';
}

}  // namespace

int main() {
    try {
        require(tmgm::native::CausalMonoResampler22050::
                    supports_input_sample_rate(44100U) &&
                    tmgm::native::CausalMonoResampler22050::
                    supports_input_sample_rate(48000U) &&
                    !tmgm::native::CausalMonoResampler22050::
                    supports_input_sample_rate(96000U),
                "resampler supported-rate contract differs");
        bool rejected_zero = false;
        try {
            static_cast<void>(
                tmgm::native::CausalMonoResampler22050::prepare(48000U, 0U));
        } catch (const std::invalid_argument&) {
            rejected_zero = true;
        }
        require(rejected_zero, "resampler accepted zero maximum block size");
        bool rejected_rate = false;
        try {
            static_cast<void>(
                tmgm::native::CausalMonoResampler22050::prepare(96000U, 64U));
        } catch (const std::invalid_argument&) {
            rejected_rate = true;
        }
        require(rejected_rate, "resampler accepted unsupported input rate");

        auto validation =
            tmgm::native::CausalMonoResampler22050::prepare(48000U, 16U);
        tmgm::native::ResampledMonoBlock output;
        require(validation.process_block(nullptr, 1U, output) ==
                    tmgm::native::CausalResamplerStatus::null_input &&
                    validation.consumed_input_sample_count() == 0U,
                "resampler null input changed state");
        std::array<float, 17> oversized{};
        require(validation.process_block(
                    oversized.data(), oversized.size(), output) ==
                    tmgm::native::CausalResamplerStatus::input_block_too_large &&
                    validation.consumed_input_sample_count() == 0U,
                "resampler oversized input changed state");
        require(validation.process_block(nullptr, 0U, output) ==
                    tmgm::native::CausalResamplerStatus::success &&
                    output.sample_count == 0U,
                "resampler rejected empty block");

        test_rate(44100U);
        test_rate(48000U);
        std::cout << "causal 44.1/48 kHz -> 22.05 kHz resampler tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
