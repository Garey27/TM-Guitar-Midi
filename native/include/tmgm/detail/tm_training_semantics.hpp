#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define TMGM_TRAINING_HOST_DEVICE __host__ __device__
#else
#define TMGM_TRAINING_HOST_DEVICE
#endif

namespace tmgm::native::detail {

// Produces a stride that is coprime with count. Together with an arbitrary
// start this gives an allocation-free affine permutation of [0, count).
TMGM_TRAINING_HOST_DEVICE inline std::uint32_t coprime_permutation_stride(
    const std::uint32_t count,
    const std::uint32_t candidate) noexcept {
    if (count <= 1U) {
        return 1U;
    }

    auto stride = candidate % count;
    if (stride == 0U) {
        stride = 1U;
    }
    for (;;) {
        auto left = stride;
        auto right = count;
        while (right != 0U) {
            const auto remainder = left % right;
            left = right;
            right = remainder;
        }
        if (left == 1U) {
            return stride;
        }
        ++stride;
        if (stride == count) {
            stride = 1U;
        }
    }
}

TMGM_TRAINING_HOST_DEVICE inline std::uint32_t affine_permuted_index(
    const std::uint32_t count,
    const std::uint32_t position,
    const std::uint32_t start,
    const std::uint32_t coprime_stride) noexcept {
    if (count == 0U) {
        return 0U;
    }
    const auto index = static_cast<std::uint64_t>(start % count) +
        static_cast<std::uint64_t>(position) * coprime_stride;
    return static_cast<std::uint32_t>(index % count);
}

TMGM_TRAINING_HOST_DEVICE inline std::uint32_t advance_affine_permutation(
    const std::uint32_t count,
    const std::uint32_t current,
    const std::uint32_t coprime_stride) noexcept {
    if (count <= 1U) {
        return 0U;
    }
    // Both operands are below count. This equivalent of
    // (current + stride) % count avoids integer overflow and an expensive
    // 64-bit remainder in the CUDA training hot loop.
    const auto distance_to_wrap = count - coprime_stride;
    return current >= distance_to_wrap
        ? current - distance_to_wrap
        : current + coprime_stride;
}

// Returns the output visited at a given position in the positive-feedback
// phase.  The fixed branch is intentionally just the historical ascending
// order; keeping it free of RNG/hash work makes the default training path
// bit-compatible.  The opt-in branch is a cyclic Latin-square schedule:
// over any `count` consecutive update steps every output occupies every order
// position exactly once.  The seed only selects the phase of that cycle.
TMGM_TRAINING_HOST_DEVICE inline std::uint32_t positive_output_update_index(
    const std::uint32_t count,
    const std::uint32_t position,
    const std::uint64_t update_step,
    const std::uint64_t seed,
    const bool rotate_start) noexcept {
    if (!rotate_start || count == 0U) {
        return position;
    }
    const auto start = static_cast<std::uint32_t>(
        ((update_step % count) + (seed % count)) % count);
    return affine_permuted_index(count, position, start, 1U);
}

TMGM_TRAINING_HOST_DEVICE inline bool positive_weight_update_allowed(
    const std::uint32_t nonnegative_weights,
    const std::uint32_t maximum_positive_weights) noexcept {
    return nonnegative_weights < maximum_positive_weights;
}

// The most important onset negative is not a random other MIDI pitch: it is
// the same pitch while the note is already active but no new onset occurred.
// Keeping the policy in a host/device helper makes the exact truth table easy
// to test independently from CUDA RNG details.
TMGM_TRAINING_HOST_DEVICE inline bool is_onset_sustain_hard_negative(
    const bool onset_target,
    const bool activity_target,
    const bool enabled) noexcept {
    return enabled && !onset_target && activity_target;
}

TMGM_TRAINING_HOST_DEVICE inline std::int32_t saturating_unit_weight_update(
    const std::int32_t value,
    const bool increment) noexcept {
    constexpr std::int32_t maximum = 2147483647;
    constexpr std::int32_t minimum = -2147483647 - 1;
    if (increment) {
        return value == maximum
            ? value
            : static_cast<std::int32_t>(value + 1);
    }
    return value == minimum
        ? value
        : static_cast<std::int32_t>(value - 1);
}

}  // namespace tmgm::native::detail

#undef TMGM_TRAINING_HOST_DEVICE
