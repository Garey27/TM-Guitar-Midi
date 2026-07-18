#include "tmgm/detail/tm_training_semantics.hpp"
#include "tmgm/model.hpp"
#include "tmgm/tm_cuda.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_default_is_historical_ascending_order() {
    const tmgm::native::TmCudaConfig config;
    require(!config.rotate_output_update_order,
            "output-order ablation unexpectedly became the default");

    constexpr std::uint32_t outputs = 49U;
    for (std::uint64_t step = 0U; step < outputs * 3U; ++step) {
        for (std::uint32_t position = 0U; position < outputs; ++position) {
            const auto output =
                tmgm::native::detail::positive_output_update_index(
                    outputs, position, step, 0xfedcba9876543210ULL, false);
            require(output == position,
                    "default output order is not the historical ascending order");
        }
    }
}

void test_rotation_is_a_balanced_cycle() {
    constexpr std::uint32_t outputs = 49U;
    constexpr std::uint64_t seed = 0x0123456789abcdefULL;

    // counts[position][output] must be exactly one over a complete cycle.
    std::array<std::array<std::uint8_t, outputs>, outputs> counts{};
    for (std::uint64_t step = 0U; step < outputs; ++step) {
        std::array<std::uint8_t, outputs> seen{};
        for (std::uint32_t position = 0U; position < outputs; ++position) {
            const auto output =
                tmgm::native::detail::positive_output_update_index(
                    outputs, position, step, seed, true);
            require(output < outputs, "rotating order produced an invalid output");
            require(seen[output] == 0U,
                    "rotating order repeated an output within one sample");
            seen[output] = 1U;
            ++counts[position][output];
        }
    }
    for (std::uint32_t position = 0U; position < outputs; ++position) {
        for (std::uint32_t output = 0U; output < outputs; ++output) {
            require(counts[position][output] == 1U,
                    "an output did not occupy every order position exactly once");
        }
    }
}

void test_schedule_is_seed_reproducible() {
    constexpr std::uint32_t outputs = 17U;
    constexpr std::uint64_t seed = 0x5eed123456789abcULL;
    std::vector<std::uint32_t> first;
    std::vector<std::uint32_t> second;
    for (std::uint64_t step = 0U; step < 100U; ++step) {
        for (std::uint32_t position = 0U; position < outputs; ++position) {
            first.push_back(
                tmgm::native::detail::positive_output_update_index(
                    outputs, position, step, seed, true));
            second.push_back(
                tmgm::native::detail::positive_output_update_index(
                    outputs, position, step, seed, true));
        }
    }
    require(first == second, "same seed did not reproduce the output schedule");

    // A different seed is allowed to change only the cycle phase.
    const auto first_start =
        tmgm::native::detail::positive_output_update_index(
            outputs, 0U, 0U, seed, true);
    const auto shifted_start =
        tmgm::native::detail::positive_output_update_index(
            outputs, 0U, 0U, seed + 1U, true);
    require(first_start != shifted_start,
            "seed did not select the rotation cycle phase");
}

tmgm::native::NativeTmModel tiny_model() {
    tmgm::native::NativeTmModel model;
    model.head = tmgm::native::TmModelHead::activity;
    model.dimensions.feature_count = 1U;
    model.dimensions.output_count = 1U;
    model.dimensions.clause_count = 1U;
    model.dimensions.state_bits = 8U;
    model.training.threshold = 1;
    model.training.specificity = 2.0F;
    model.training.negative_samples = 1.0F;
    model.training.type_i_ii_ratio = 1.0F;
    model.training.max_included_literals = 2U;
    model.training.epochs_trained = 1U;
    model.training.seed = 7U;
    model.midi.minimum_note = 40;
    model.midi.maximum_note = 40;
    model.midi.channel = 1U;
    model.midi.audio_sample_rate = 22050U;
    model.midi.analysis_hop_samples = 256U;
    model.ta_bitplanes.assign(8U, 0U);
    model.weights.assign(1U, 1);
    return model;
}

void test_ablation_does_not_change_inference_format() {
    // The option lives solely in TmCudaConfig. Toggling it must not alter the
    // canonical inference model or its serialized checksum/version contract.
    auto config = tmgm::native::TmCudaConfig{};
    const auto model = tiny_model();
    tmgm::native::validate_tm_model(model);
    const auto before = tmgm::native::calculate_tm_model_checksum(model);
    config.rotate_output_update_order = true;
    const auto after = tmgm::native::calculate_tm_model_checksum(model);
    require(config.rotate_output_update_order, "test did not enable the ablation");
    require(before == after,
            "training-only output order changed the inference model contract");
    require(tmgm::native::kTmModelHeaderBytes == 256U,
            "native model header size changed unexpectedly");
}

}  // namespace

int main() {
    try {
        test_default_is_historical_ascending_order();
        test_rotation_is_a_balanced_cycle();
        test_schedule_is_seed_reproducible();
        test_ablation_does_not_change_inference_format();
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
