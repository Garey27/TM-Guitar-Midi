#include "tmgm/tm_cuda.hpp"

#include "tmgm/detail/tm_training_semantics.hpp"
#include "tmgm/tm_validation.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cg = cooperative_groups;

namespace tmgm::native {
namespace {

constexpr std::uint32_t kWarpSize = 32;
constexpr std::uint32_t kThreadsPerBlock = 256;

void check_cuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(const std::size_t count) : count_(count) {
        if (count_ != 0) {
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
                "cudaMalloc");
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    [[nodiscard]] T* get() noexcept { return data_; }
    [[nodiscard]] const T* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

    void copy_from(const T* source, const std::size_t count) {
        if (count > count_) {
            throw std::out_of_range("device copy exceeds allocation");
        }
        check_cuda(
            cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice),
            "cudaMemcpy host to device");
    }

    void copy_to(T* destination, const std::size_t count) const {
        if (count > count_) {
            throw std::out_of_range("device copy exceeds allocation");
        }
        check_cuda(
            cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost),
            "cudaMemcpy device to host");
    }

    void copy_from_device(const DeviceBuffer<T>& source, const std::size_t count) {
        if (count > count_ || count > source.count_) {
            throw std::out_of_range("device-to-device copy exceeds allocation");
        }
        check_cuda(
            cudaMemcpy(data_, source.data_, count * sizeof(T), cudaMemcpyDeviceToDevice),
            "cudaMemcpy device to device");
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

__device__ __forceinline__ std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

__device__ __forceinline__ std::uint64_t random_word(
    const std::uint64_t seed,
    const std::uint64_t step,
    const std::uint32_t output,
    const std::uint32_t clause,
    const std::uint32_t chunk,
    const std::uint32_t lane_or_bit,
    const std::uint32_t stream) {
    std::uint64_t key = seed;
    key ^= mix64(step + 0x632be59bd9b4e019ull);
    key ^= mix64(static_cast<std::uint64_t>(output) + 0x8cb92baa3f3d8dd7ull);
    key ^= mix64(static_cast<std::uint64_t>(clause) + 0x9e3779b185ebca87ull);
    key ^= mix64(static_cast<std::uint64_t>(chunk) + 0xc2b2ae3d27d4eb4full);
    key ^= mix64(static_cast<std::uint64_t>(lane_or_bit) + 0x165667b19e3779f9ull);
    key ^= mix64(static_cast<std::uint64_t>(stream) + 0x85ebca77c2b2ae63ull);
    return mix64(key);
}

__device__ __forceinline__ float random_unit(
    const std::uint64_t seed,
    const std::uint64_t step,
    const std::uint32_t output,
    const std::uint32_t clause,
    const std::uint32_t chunk,
    const std::uint32_t lane_or_bit,
    const std::uint32_t stream) {
    const auto bits = static_cast<std::uint32_t>(
        random_word(seed, step, output, clause, chunk, lane_or_bit, stream) >> 40u);
    return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ std::uint32_t valid_word_mask(
    const std::uint32_t chunk,
    const std::uint32_t chunk_count,
    const std::uint32_t literal_count) {
    if (chunk + 1u != chunk_count || (literal_count & 31u) == 0u) {
        return 0xffffffffu;
    }
    return (std::uint32_t{1} << (literal_count & 31u)) - 1u;
}

__device__ __forceinline__ bool packed_label(
    const std::uint64_t* labels,
    const std::uint32_t words_per_row,
    const std::uint32_t row,
    const std::uint32_t output) {
    const auto word = labels[
        static_cast<std::size_t>(row) * words_per_row + output / 64u];
    return ((word >> (output & 63u)) & 1ull) != 0ull;
}

__global__ void encode_literal_rows(
    const std::uint64_t* feature_words,
    const std::uint64_t row_count,
    const std::uint32_t feature_count,
    const std::uint32_t feature_words_per_row,
    const std::uint32_t literal_count,
    const std::uint32_t literal_chunks,
    std::uint32_t* encoded_literals) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = row_count * literal_chunks;
    if (index >= total) {
        return;
    }
    const auto row = index / literal_chunks;
    const auto chunk = static_cast<std::uint32_t>(index % literal_chunks);
    std::uint32_t packed = 0u;
    for (std::uint32_t bit = 0; bit < 32u; ++bit) {
        const auto literal = chunk * 32u + bit;
        if (literal >= literal_count) {
            break;
        }
        const bool negated = literal >= feature_count;
        const auto feature = negated ? literal - feature_count : literal;
        const auto source_word = feature_words[
            row * feature_words_per_row + feature / 64u];
        bool value = ((source_word >> (feature & 63u)) & 1ull) != 0ull;
        if (negated) {
            value = !value;
        }
        if (value) {
            packed |= std::uint32_t{1} << bit;
        }
    }
    encoded_literals[index] = packed;
}

__global__ void initialize_states(
    std::uint32_t* states,
    const std::uint32_t clauses,
    const std::uint32_t state_bits,
    const std::uint32_t chunks,
    const std::uint32_t literal_count) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = static_cast<std::uint64_t>(clauses) * state_bits * chunks;
    if (index >= total) {
        return;
    }
    const auto chunk = static_cast<std::uint32_t>(index % chunks);
    const auto plane = static_cast<std::uint32_t>((index / chunks) % state_bits);
    states[index] = plane + 1u == state_bits
        ? 0u
        : valid_word_mask(chunk, chunks, literal_count);
}

__device__ __forceinline__ std::uint32_t warp_or(std::uint32_t value) {
    constexpr std::uint32_t mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value |= __shfl_down_sync(mask, value, offset);
    }
    return __shfl_sync(mask, value, 0);
}

__device__ __forceinline__ std::uint32_t warp_sum(std::uint32_t value) {
    constexpr std::uint32_t mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return __shfl_sync(mask, value, 0);
}

struct ClauseStats {
    bool fires;
    bool has_include;
    std::uint32_t included_literals;
};

__device__ __forceinline__ ClauseStats evaluate_clause_warp(
    const std::uint32_t* states,
    const std::uint32_t* x,
    const std::uint32_t clause,
    const std::uint32_t state_bits,
    const std::uint32_t chunks,
    const std::uint32_t literal_count,
    const std::uint32_t lane) {
    std::uint32_t bad = 0u;
    std::uint32_t any = 0u;
    std::uint32_t included = 0u;
    const auto action_base =
        (static_cast<std::size_t>(clause) * state_bits + state_bits - 1u) * chunks;
    for (std::uint32_t chunk = lane; chunk < chunks; chunk += kWarpSize) {
        const auto mask = valid_word_mask(chunk, chunks, literal_count);
        const auto action = states[action_base + chunk] & mask;
        bad |= action & ~x[chunk] & mask;
        any |= action;
        included += static_cast<std::uint32_t>(__popc(action));
    }
    bad = warp_or(bad);
    any = warp_or(any);
    included = warp_sum(included);
    return {bad == 0u, any != 0u, included};
}

__device__ __forceinline__ void increment_state_word(
    std::uint32_t* states,
    const std::uint32_t clause,
    const std::uint32_t chunk,
    const std::uint32_t state_bits,
    const std::uint32_t chunks,
    std::uint32_t active) {
    std::uint32_t carry = active;
    for (std::uint32_t bit = 0; bit < state_bits; ++bit) {
        const auto index =
            (static_cast<std::size_t>(clause) * state_bits + bit) * chunks + chunk;
        const auto value = states[index];
        const auto next = value & carry;
        states[index] = value ^ carry;
        carry = next;
    }
    if (carry != 0u) {
        for (std::uint32_t bit = 0; bit < state_bits; ++bit) {
            const auto index =
                (static_cast<std::size_t>(clause) * state_bits + bit) * chunks + chunk;
            states[index] |= carry;
        }
    }
}

__device__ __forceinline__ void decrement_state_word(
    std::uint32_t* states,
    const std::uint32_t clause,
    const std::uint32_t chunk,
    const std::uint32_t state_bits,
    const std::uint32_t chunks,
    std::uint32_t active) {
    std::uint32_t carry = active;
    for (std::uint32_t bit = 0; bit < state_bits; ++bit) {
        const auto index =
            (static_cast<std::size_t>(clause) * state_bits + bit) * chunks + chunk;
        const auto value = states[index];
        const auto next = (~value) & carry;
        states[index] = value ^ carry;
        carry = next;
    }
    if (carry != 0u) {
        for (std::uint32_t bit = 0; bit < state_bits; ++bit) {
            const auto index =
                (static_cast<std::size_t>(clause) * state_bits + bit) * chunks + chunk;
            states[index] &= ~carry;
        }
    }
}

__device__ __forceinline__ std::uint32_t lowest_set_bits(
    std::uint32_t mask,
    std::uint32_t count) {
    std::uint32_t selected = 0u;
    while (mask != 0u && count != 0u) {
        const auto lowest = mask & (0u - mask);
        selected |= lowest;
        mask &= mask - 1u;
        --count;
    }
    return selected;
}

// Limit only increments that cross the Exclude/Include action boundary. All
// other TA state changes retain their historical behavior. Calls are made by
// a complete warp for one consecutive 32-word batch, so the prefix scan admits
// candidates in deterministic ascending literal order without atomics.
__device__ __forceinline__ std::uint32_t cap_literal_action_transitions_warp(
    const std::uint32_t* states,
    const std::uint32_t clause,
    const std::uint32_t chunk,
    const bool chunk_is_valid,
    const std::uint32_t state_bits,
    const std::uint32_t chunks,
    const std::uint32_t increment_mask,
    std::uint32_t& remaining_literal_slots,
    const std::uint32_t lane) {
    std::uint32_t enters_include = 0u;
    if (chunk_is_valid) {
        const auto action_index =
            (static_cast<std::size_t>(clause) * state_bits + state_bits - 1u) *
                chunks +
            chunk;
        enters_include = increment_mask & ~states[action_index];
        for (std::uint32_t bit = 0u; bit + 1u < state_bits; ++bit) {
            const auto index =
                (static_cast<std::size_t>(clause) * state_bits + bit) * chunks +
                chunk;
            enters_include &= states[index];
        }
    }

    const auto candidate_count =
        static_cast<std::uint32_t>(__popc(enters_include));
    auto inclusive_count = candidate_count;
    constexpr std::uint32_t warp_mask = 0xffffffffu;
    for (std::uint32_t offset = 1u; offset < kWarpSize; offset <<= 1u) {
        const auto preceding =
            __shfl_up_sync(warp_mask, inclusive_count, static_cast<int>(offset));
        if (lane >= offset) {
            inclusive_count += preceding;
        }
    }
    const auto preceding_count = inclusive_count - candidate_count;
    const auto available = remaining_literal_slots > preceding_count
        ? remaining_literal_slots - preceding_count
        : 0u;
    const auto admitted_count =
        available < candidate_count ? available : candidate_count;
    const auto admitted = lowest_set_bits(enters_include, admitted_count);
    const auto admitted_in_batch = warp_sum(
        static_cast<std::uint32_t>(__popc(admitted)));
    remaining_literal_slots = admitted_in_batch < remaining_literal_slots
        ? remaining_literal_slots - admitted_in_batch
        : 0u;
    return (increment_mask & ~enters_include) | admitted;
}

__device__ __forceinline__ std::uint32_t bernoulli_literal_mask(
    const float probability,
    const std::uint64_t seed,
    const std::uint64_t step,
    const std::uint32_t output,
    const std::uint32_t clause,
    const std::uint32_t chunk,
    const std::uint32_t stream) {
    std::uint32_t result = 0u;
#pragma unroll
    for (std::uint32_t bit = 0; bit < 32u; ++bit) {
        if (random_unit(seed, step, output, clause, chunk, bit, stream) < probability) {
            result |= std::uint32_t{1} << bit;
        }
    }
    return result;
}

__global__ void train_chunk_kernel(
    const std::uint32_t* encoded_x,
    const std::uint64_t* labels,
    const std::uint64_t* activity_labels,
    const std::uint32_t label_words_per_row,
    const std::uint32_t* order,
    const std::uint32_t order_offset,
    const std::uint32_t sample_count,
    const std::uint64_t step_base,
    const std::uint32_t output_count,
    const std::uint32_t clauses,
    const std::uint32_t chunks,
    const std::uint32_t literal_count,
    const std::uint32_t state_bits,
    const std::int32_t threshold,
    const float specificity,
    const float negative_samples,
    const std::uint32_t onset_sustain_hard_negatives,
    const float onset_sustain_hard_negative_probability,
    const std::uint32_t onset_sustain_hard_negative_weight_only,
    const std::uint32_t max_included_literals,
    const std::uint32_t rotate_output_update_order,
    const std::uint64_t seed,
    std::uint32_t* states,
    std::int32_t* weights,
    std::uint32_t* clause_outputs,
    std::int32_t* class_sums,
    std::uint32_t* nonnegative_weight_counts) {
    auto grid = cg::this_grid();
    const auto global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const auto lane = threadIdx.x & (kWarpSize - 1u);
    const auto global_warp = global_thread / kWarpSize;
    const auto warp_count = gridDim.x * blockDim.x / kWarpSize;

    for (std::uint32_t local_sample = 0; local_sample < sample_count; ++local_sample) {
        const auto row = order[order_offset + local_sample];
        const auto step = step_base + local_sample;
        const auto* x = encoded_x + static_cast<std::size_t>(row) * chunks;

        for (std::uint32_t clause = global_warp;
             clause < clauses;
             clause += warp_count) {
            const auto stats = evaluate_clause_warp(
                states, x, clause, state_bits, chunks, literal_count, lane);
            if (lane == 0u) {
                // Training deliberately treats an empty clause as true.
                clause_outputs[clause] = stats.fires ? 1u : 0u;
            }
        }
        grid.sync();

        if (global_thread < output_count) {
            std::int64_t sum = 0;
            std::uint32_t nonnegative_weights = 0u;
            const auto output = global_thread;
            for (std::uint32_t clause = 0; clause < clauses; ++clause) {
                const auto weight =
                    weights[static_cast<std::size_t>(output) * clauses + clause];
                sum += static_cast<std::int64_t>(clause_outputs[clause]) * weight;
                nonnegative_weights += weight >= 0 ? 1u : 0u;
            }
            sum = max(
                -static_cast<std::int64_t>(threshold),
                min(static_cast<std::int64_t>(threshold), sum));
            class_sums[output] = static_cast<std::int32_t>(sum);
            nonnegative_weight_counts[output] = nonnegative_weights;
        }
        grid.sync();

        std::uint32_t negative_start = 0u;
        std::uint32_t negative_stride = 1u;
        if (lane == 0u) {
            negative_start = output_count == 0u
                ? 0u
                : static_cast<std::uint32_t>(
                      random_word(seed, step, 0u, 0u, 0u, 0u, 2u) % output_count);
            negative_stride = detail::coprime_permutation_stride(
                output_count,
                static_cast<std::uint32_t>(
                    random_word(seed, step, 0u, 0u, 0u, 0u, 3u)));
        }
        negative_start = __shfl_sync(0xffffffffu, negative_start, 0);
        negative_stride = __shfl_sync(0xffffffffu, negative_stride, 0);

        for (std::uint32_t clause = global_warp;
             clause < clauses;
             clause += warp_count) {
            const auto original_clause_output = clause_outputs[clause] != 0u;

            // TMU processes every positive output first, then sampled negatives.
            for (std::uint32_t phase = 0; phase < 2u; ++phase) {
                auto negative_output = negative_start;
                for (std::uint32_t position = 0; position < output_count; ++position) {
                    const auto output = phase == 0u
                        ? detail::positive_output_update_index(
                              output_count,
                              position,
                              step,
                              seed,
                              rotate_output_update_order != 0u)
                        : negative_output;
                    if (phase != 0u) {
                        negative_output = detail::advance_affine_permutation(
                            output_count, negative_output, negative_stride);
                    }
                    const bool target = packed_label(
                        labels, label_words_per_row, row, output);
                    if ((phase == 0u) != target) {
                        continue;
                    }
                    bool sustain_hard_negative = false;
                    if (!target) {
                        const bool sustain_candidate =
                            onset_sustain_hard_negatives != 0u &&
                            detail::is_onset_sustain_hard_negative(
                                target,
                                packed_label(
                                    activity_labels,
                                    label_words_per_row,
                                    row,
                                    output),
                                true);
                        sustain_hard_negative =
                            sustain_candidate &&
                            random_unit(
                                seed, step, output, 0u, 0u, 0u, 4u) <
                                onset_sustain_hard_negative_probability;
                        const auto negative_probability = output_count <= 1u
                            ? 1.0f
                            : min(1.0f, negative_samples /
                                static_cast<float>(output_count - 1u));
                        if (!sustain_hard_negative &&
                            random_unit(seed, step, output, 0u, 0u, 0u, 1u) >=
                                negative_probability) {
                            continue;
                        }
                    }

                    const auto signed_target = target ? 1.0f : -1.0f;
                    const auto update_probability =
                        (static_cast<float>(threshold) -
                         signed_target * static_cast<float>(class_sums[output])) /
                        (2.0f * static_cast<float>(threshold));

                    std::int32_t weight = lane == 0u
                        ? weights[static_cast<std::size_t>(output) * clauses + clause]
                        : 0;
                    weight = __shfl_sync(0xffffffffu, weight, 0);
                    const bool type_i = (weight >= 0) == target;
                    const auto feedback_stream = type_i ? 10u : 20u;
                    const bool apply_feedback = random_unit(
                        seed,
                        step,
                        output,
                        clause,
                        0u,
                        0u,
                        feedback_stream) < update_probability;

                    if (apply_feedback &&
                        !(sustain_hard_negative &&
                          onset_sustain_hard_negative_weight_only != 0u)) {
                        const auto current = evaluate_clause_warp(
                            states, x, clause, state_bits, chunks, literal_count, lane);
                        if (type_i) {
                            const bool reinforce =
                                current.fires &&
                                current.included_literals < max_included_literals;
                            if (reinforce) {
                                auto remaining_literal_slots =
                                    max_included_literals -
                                    current.included_literals;
                                for (std::uint32_t batch = 0u;
                                     batch < chunks;
                                     batch += kWarpSize) {
                                    const auto chunk = batch + lane;
                                    const auto chunk_is_valid = chunk < chunks;
                                    const auto valid = chunk_is_valid
                                        ? valid_word_mask(
                                              chunk, chunks, literal_count)
                                        : 0u;
                                    const auto value = chunk_is_valid
                                        ? x[chunk] & valid
                                        : 0u;
                                    const auto random_mask = chunk_is_valid
                                        ? bernoulli_literal_mask(
                                              1.0f / specificity,
                                              seed,
                                              step,
                                              output,
                                              clause,
                                              chunk,
                                              30u)
                                        : 0u;
                                    auto increment =
                                        cap_literal_action_transitions_warp(
                                            states,
                                            clause,
                                            chunk,
                                            chunk_is_valid,
                                            state_bits,
                                            chunks,
                                            value,
                                            remaining_literal_slots,
                                            lane);
                                    if (!chunk_is_valid) {
                                        continue;
                                    }
                                    // boost_true_positive_feedback=1 in the TMU model.
                                    increment_state_word(
                                        states,
                                        clause,
                                        chunk,
                                        state_bits,
                                        chunks,
                                        increment);
                                    decrement_state_word(
                                        states,
                                        clause,
                                        chunk,
                                        state_bits,
                                        chunks,
                                        (~value) & valid & random_mask);
                                }
                            } else {
                                for (std::uint32_t chunk = lane;
                                     chunk < chunks;
                                     chunk += kWarpSize) {
                                    const auto valid =
                                        valid_word_mask(chunk, chunks, literal_count);
                                    const auto random_mask = bernoulli_literal_mask(
                                        1.0f / specificity,
                                        seed,
                                        step,
                                        output,
                                        clause,
                                        chunk,
                                        30u);
                                    decrement_state_word(
                                        states,
                                        clause,
                                        chunk,
                                        state_bits,
                                        chunks,
                                        valid & random_mask);
                                }
                            }
                        } else if (current.fires) {
                            auto remaining_literal_slots =
                                current.included_literals < max_included_literals
                                ? max_included_literals - current.included_literals
                                : 0u;
                            for (std::uint32_t batch = 0u;
                                 batch < chunks;
                                 batch += kWarpSize) {
                                const auto chunk = batch + lane;
                                const auto chunk_is_valid = chunk < chunks;
                                const auto valid = chunk_is_valid
                                    ? valid_word_mask(chunk, chunks, literal_count)
                                    : 0u;
                                auto increment = chunk_is_valid
                                    ? (~x[chunk]) & valid
                                    : 0u;
                                increment = cap_literal_action_transitions_warp(
                                    states,
                                    clause,
                                    chunk,
                                    chunk_is_valid,
                                    state_bits,
                                    chunks,
                                    increment,
                                    remaining_literal_slots,
                                    lane);
                                if (!chunk_is_valid) {
                                    continue;
                                }
                                increment_state_word(
                                    states,
                                    clause,
                                    chunk,
                                    state_bits,
                                    chunks,
                                    increment);
                            }
                        }
                    }
                    __syncwarp();

                    if (lane == 0u && original_clause_output &&
                        random_unit(seed, step, output, clause, 0u, 0u, 40u) <
                            update_probability) {
                        auto& stored =
                            weights[static_cast<std::size_t>(output) * clauses + clause];
                        if (!target || detail::positive_weight_update_allowed(
                                           nonnegative_weight_counts[output], clauses)) {
                            stored = detail::saturating_unit_weight_update(stored, target);
                        }
                    }
                    __syncwarp();
                }
            }
        }
        grid.sync();
    }
}

__global__ void predict_clauses_kernel(
    const std::uint32_t* encoded_x,
    const std::uint64_t row_count,
    const std::uint32_t clauses,
    const std::uint32_t chunks,
    const std::uint32_t literal_count,
    const std::uint32_t state_bits,
    const std::uint32_t* states,
    std::uint8_t* clause_outputs) {
    const auto global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const auto lane = threadIdx.x & (kWarpSize - 1u);
    const auto global_warp = global_thread / kWarpSize;
    const auto warp_count = gridDim.x * blockDim.x / kWarpSize;
    const auto pair_count = row_count * clauses;
    for (std::uint64_t pair = global_warp; pair < pair_count; pair += warp_count) {
        const auto row = pair / clauses;
        const auto clause = static_cast<std::uint32_t>(pair % clauses);
        const auto* x = encoded_x + row * chunks;
        const auto stats = evaluate_clause_warp(
            states, x, clause, state_bits, chunks, literal_count, lane);
        if (lane == 0u) {
            // Prediction follows TMU and suppresses all-exclude clauses.
            clause_outputs[pair] = stats.fires && stats.has_include ? 1u : 0u;
        }
    }
}

__global__ void predict_scores_kernel(
    const std::uint8_t* clause_outputs,
    const std::int32_t* weights,
    const std::uint64_t row_count,
    const std::uint32_t output_count,
    const std::uint32_t clauses,
    std::int32_t* scores) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = row_count * output_count;
    if (index >= total) {
        return;
    }
    const auto row = index / output_count;
    const auto output = static_cast<std::uint32_t>(index % output_count);
    std::int64_t sum = 0;
    for (std::uint32_t clause = 0; clause < clauses; ++clause) {
        sum += static_cast<std::int64_t>(
            clause_outputs[row * clauses + clause]) *
            weights[static_cast<std::size_t>(output) * clauses + clause];
    }
    sum = max(
        static_cast<std::int64_t>(-2147483647) - 1,
        min(static_cast<std::int64_t>(2147483647), sum));
    scores[index] = static_cast<std::int32_t>(sum);
}

struct HostMetrics {
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double predicted_mean = 0.0;
    double target_mean = 0.0;
    std::int32_t threshold = 0;
};

[[nodiscard]] bool host_label(
    const std::vector<std::uint64_t>& labels,
    const NativeDatasetHeader& header,
    const std::uint64_t row,
    const std::uint32_t output) {
    const auto word = labels[
        row * header.label_words_per_row + output / 64u];
    return ((word >> (output & 63u)) & 1ull) != 0ull;
}

[[nodiscard]] HostMetrics calibrate_metrics(
    const std::vector<std::int32_t>& scores,
    const std::vector<std::uint64_t>& labels,
    const NativeDatasetHeader& header,
    const double maximum_ratio) {
    const auto cell_count = scores.size();
    std::uint64_t positives = 0;
    auto minimum = std::numeric_limits<std::int32_t>::max();
    auto maximum = std::numeric_limits<std::int32_t>::min();
    for (std::uint64_t row = 0; row < header.frame_count; ++row) {
        for (std::uint32_t output = 0; output < header.note_count; ++output) {
            positives += host_label(labels, header, row, output) ? 1u : 0u;
            const auto score = scores[row * header.note_count + output];
            minimum = std::min(minimum, score);
            maximum = std::max(maximum, score);
        }
    }
    const auto target_mean = static_cast<double>(positives) /
        static_cast<double>(header.frame_count);
    const auto score_range =
        static_cast<std::uint64_t>(
            static_cast<std::int64_t>(maximum) - static_cast<std::int64_t>(minimum)) +
        1u;
    if (score_range > 10'000'000u) {
        throw std::runtime_error("TM score range is unexpectedly large");
    }
    std::vector<std::uint64_t> true_histogram(
        static_cast<std::size_t>(score_range), 0u);
    std::vector<std::uint64_t> false_histogram(
        static_cast<std::size_t>(score_range), 0u);
    for (std::size_t index = 0; index < cell_count; ++index) {
        const auto row = index / header.note_count;
        const auto output = static_cast<std::uint32_t>(index % header.note_count);
        const auto bucket = static_cast<std::size_t>(
            static_cast<std::int64_t>(scores[index]) - minimum);
        if (host_label(labels, header, row, output)) {
            ++true_histogram[bucket];
        } else {
            ++false_histogram[bucket];
        }
    }

    HostMetrics best;
    best.f1 = -1.0;
    HostMetrics best_unconstrained;
    best_unconstrained.f1 = -1.0;

    std::uint64_t tp = 0;
    std::uint64_t fp = 0;
    auto consider = [&](const std::int32_t threshold) {
        const auto fn = positives - tp;
        const auto predicted = tp + fp;
        const auto precision = static_cast<double>(tp) /
            static_cast<double>(std::max<std::uint64_t>(tp + fp, 1u));
        const auto recall = static_cast<double>(tp) /
            static_cast<double>(std::max<std::uint64_t>(tp + fn, 1u));
        const auto f1 = 2.0 * precision * recall /
            std::max(precision + recall, 1.0e-12);
        const auto predicted_mean = static_cast<double>(predicted) /
            static_cast<double>(header.frame_count);
        HostMetrics candidate{
            precision,
            recall,
            f1,
            predicted_mean,
            target_mean,
            threshold,
        };
        if (candidate.f1 > best_unconstrained.f1 ||
            (candidate.f1 == best_unconstrained.f1 && threshold > best_unconstrained.threshold)) {
            best_unconstrained = candidate;
        }
        const bool allowed =
            predicted_mean >= 0.5 * target_mean &&
            predicted_mean <= maximum_ratio * target_mean;
        if (allowed &&
            (candidate.f1 > best.f1 ||
             (candidate.f1 == best.f1 && threshold > best.threshold))) {
            best = candidate;
        }
    };

    if (maximum < std::numeric_limits<std::int32_t>::max()) {
        consider(maximum + 1);
    }
    for (std::int64_t threshold = maximum; threshold >= minimum; --threshold) {
        const auto bucket = static_cast<std::size_t>(threshold - minimum);
        tp += true_histogram[bucket];
        fp += false_histogram[bucket];
        consider(static_cast<std::int32_t>(threshold));
    }
    return best.f1 >= 0.0 ? best : best_unconstrained;
}

void launch_prediction(
    const DeviceBuffer<std::uint32_t>& encoded_x,
    const DeviceBuffer<std::uint32_t>& states,
    const DeviceBuffer<std::int32_t>& weights,
    const std::uint64_t rows,
    const std::uint32_t outputs,
    const std::uint32_t clauses,
    const std::uint32_t chunks,
    const std::uint32_t literals,
    const std::uint32_t state_bits,
    DeviceBuffer<std::uint8_t>& clause_outputs,
    DeviceBuffer<std::int32_t>& scores) {
    const auto pair_count = rows * clauses;
    const auto required_blocks = static_cast<std::uint32_t>(
        (pair_count + (kThreadsPerBlock / kWarpSize) - 1u) /
        (kThreadsPerBlock / kWarpSize));
    const auto blocks = std::min<std::uint32_t>(required_blocks, 65535u);
    predict_clauses_kernel<<<blocks, kThreadsPerBlock>>>(
        encoded_x.get(),
        rows,
        clauses,
        chunks,
        literals,
        state_bits,
        states.get(),
        clause_outputs.get());
    check_cuda(cudaGetLastError(), "launch predict clauses");

    const auto score_count = rows * outputs;
    const auto score_blocks = static_cast<std::uint32_t>(
        (score_count + kThreadsPerBlock - 1u) / kThreadsPerBlock);
    predict_scores_kernel<<<score_blocks, kThreadsPerBlock>>>(
        clause_outputs.get(),
        weights.get(),
        rows,
        outputs,
        clauses,
        scores.get());
    check_cuda(cudaGetLastError(), "launch predict scores");
}

}  // namespace

bool cuda_tm_supported() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        return false;
    }
    int cooperative = 0;
    if (cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, 0) !=
        cudaSuccess) {
        return false;
    }
    return cooperative != 0;
}

TmCudaResult train_tm_cuda(
    const NativeDataset& dataset,
    const TargetHead head,
    const TmCudaConfig& config,
    const NativeDataset* const validation_dataset,
    const bool allow_legacy_feature_contract) {
    dataset.validate();
    const auto training_contract_is_legacy = std::all_of(
        dataset.header.feature_fingerprint_sha256.begin(),
        dataset.header.feature_fingerprint_sha256.end(),
        [](const std::uint8_t value) { return value == 0U; });
    if (training_contract_is_legacy && !allow_legacy_feature_contract) {
        throw std::invalid_argument(
            "training dataset has no feature-semantics fingerprint; pass an "
            "explicit legacy opt-in only for audit use");
    }
    if (validation_dataset != nullptr) {
        validate_tm_training_validation_compatibility(
            dataset, *validation_dataset);
        const auto validation_contract_is_legacy = std::all_of(
            validation_dataset->header.feature_fingerprint_sha256.begin(),
            validation_dataset->header.feature_fingerprint_sha256.end(),
            [](const std::uint8_t value) { return value == 0U; });
        if (validation_contract_is_legacy && !allow_legacy_feature_contract) {
            throw std::invalid_argument(
                "validation dataset has no feature-semantics fingerprint; "
                "pass an explicit legacy opt-in only for audit use");
        }
        if (!training_contract_is_legacy && !validation_contract_is_legacy &&
            dataset.header.feature_fingerprint_sha256 !=
                validation_dataset->header.feature_fingerprint_sha256) {
            throw std::invalid_argument(
                "training and validation feature-semantics fingerprints differ");
        }
    }
    validate_tm_validation_patience(
        config.validation_patience, validation_dataset != nullptr);
    if (!cuda_tm_supported()) {
        throw std::runtime_error("CUDA cooperative launch is unavailable");
    }
    if (config.clauses == 0u || config.threshold <= 0 ||
        !std::isfinite(config.specificity) || config.specificity <= 1.0f ||
        !std::isfinite(config.negative_samples) || config.negative_samples < 0.0f ||
        config.epochs == 0u ||
        config.state_bits < 2u || config.state_bits > 16u) {
        throw std::invalid_argument("invalid TM CUDA configuration");
    }
    if (config.samples_per_launch == 0u ||
        config.samples_per_launch > kTmCudaMaxSamplesPerLaunch) {
        throw std::invalid_argument(
            "samples_per_launch must be in [1, 512] to limit WDDM kernel duration");
    }
    if (config.onset_sustain_hard_negatives && head != TargetHead::onset) {
        throw std::invalid_argument(
            "onset sustain hard negatives require the onset target head");
    }
    if (!std::isfinite(config.onset_sustain_hard_negative_probability) ||
        config.onset_sustain_hard_negative_probability < 0.0f ||
        config.onset_sustain_hard_negative_probability > 1.0f ||
        (config.onset_sustain_hard_negatives &&
         config.onset_sustain_hard_negative_probability <= 0.0f)) {
        throw std::invalid_argument(
            "onset sustain hard-negative probability must be in (0, 1]");
    }
    if (config.onset_sustain_hard_negative_weight_only &&
        !config.onset_sustain_hard_negatives) {
        throw std::invalid_argument(
            "weight-only sustain hard negatives require hard negatives to be enabled");
    }
    if (dataset.header.frame_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("native trainer currently uses uint32 row indices");
    }
    if (validation_dataset != nullptr &&
        validation_dataset->header.frame_count >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "native validator currently uses uint32 row indices");
    }

    if (dataset.header.feature_count >
        std::numeric_limits<std::uint32_t>::max() / 2u) {
        throw std::invalid_argument("feature count is too large for negated literals");
    }
    const auto rows = dataset.header.frame_count;
    const auto outputs = dataset.header.note_count;
    const auto features = dataset.header.feature_count;
    const auto literals = features * 2u;
    const auto chunks = (literals + 31u) / 32u;
    if (config.max_included_literals == 0u ||
        config.max_included_literals > literals) {
        throw std::invalid_argument("max included literals is outside literal range");
    }
    const auto& host_labels =
        head == TargetHead::activity ? dataset.activity_words : dataset.onset_words;

    std::vector<std::uint32_t> base_order;
    if (head == TargetHead::activity) {
        base_order.resize(static_cast<std::size_t>(rows));
        for (std::uint32_t row = 0; row < rows; ++row) {
            base_order[row] = row;
        }
    } else {
        base_order = dataset.onset_indices;
    }
    if (base_order.empty()) {
        throw std::invalid_argument("training order is empty");
    }

    DeviceBuffer<std::uint64_t> device_features(dataset.feature_words.size());
    DeviceBuffer<std::uint64_t> device_labels(host_labels.size());
    const auto use_onset_sustain_hard_negatives =
        head == TargetHead::onset && config.onset_sustain_hard_negatives;
    DeviceBuffer<std::uint64_t> device_activity_labels(
        use_onset_sustain_hard_negatives ? dataset.activity_words.size() : 0U);
    DeviceBuffer<std::uint32_t> encoded_x(
        static_cast<std::size_t>(rows) * chunks);
    DeviceBuffer<std::uint32_t> order(base_order.size());
    DeviceBuffer<std::uint32_t> states(
        static_cast<std::size_t>(config.clauses) * config.state_bits * chunks);
    DeviceBuffer<std::int32_t> weights(
        static_cast<std::size_t>(outputs) * config.clauses);
    DeviceBuffer<std::uint32_t> train_clause_outputs(config.clauses);
    DeviceBuffer<std::int32_t> class_sums(outputs);
    DeviceBuffer<std::uint32_t> nonnegative_weight_counts(outputs);
    DeviceBuffer<std::uint8_t> prediction_clause_outputs(
        static_cast<std::size_t>(rows) * config.clauses);
    DeviceBuffer<std::int32_t> prediction_scores(
        static_cast<std::size_t>(rows) * outputs);

    DeviceBuffer<std::uint64_t> validation_device_features;
    DeviceBuffer<std::uint32_t> validation_encoded_x;
    DeviceBuffer<std::uint8_t> validation_clause_outputs;
    DeviceBuffer<std::int32_t> validation_prediction_scores;
    std::vector<std::int32_t> validation_host_scores;
    if (validation_dataset != nullptr) {
        const auto validation_rows = validation_dataset->header.frame_count;
        validation_device_features = DeviceBuffer<std::uint64_t>(
            validation_dataset->feature_words.size());
        validation_encoded_x = DeviceBuffer<std::uint32_t>(
            static_cast<std::size_t>(validation_rows) * chunks);
        validation_clause_outputs = DeviceBuffer<std::uint8_t>(
            static_cast<std::size_t>(validation_rows) * config.clauses);
        validation_prediction_scores = DeviceBuffer<std::int32_t>(
            static_cast<std::size_t>(validation_rows) * outputs);
        validation_host_scores.resize(
            static_cast<std::size_t>(validation_rows) * outputs);
    }

    // Allocated once and updated with D2D copies only when held-out F1 improves.
    // This avoids a host checkpoint round-trip (and epoch-by-epoch allocation)
    // while preserving the exact TA/weight state selected by validation.
    DeviceBuffer<std::uint32_t> best_states(
        validation_dataset != nullptr ? states.size() : 0U);
    DeviceBuffer<std::int32_t> best_weights(
        validation_dataset != nullptr ? weights.size() : 0U);

    device_features.copy_from(dataset.feature_words.data(), dataset.feature_words.size());
    device_labels.copy_from(host_labels.data(), host_labels.size());
    if (use_onset_sustain_hard_negatives) {
        device_activity_labels.copy_from(
            dataset.activity_words.data(), dataset.activity_words.size());
    }
    if (validation_dataset != nullptr) {
        validation_device_features.copy_from(
            validation_dataset->feature_words.data(),
            validation_dataset->feature_words.size());
    }

    const auto encode_total = rows * chunks;
    const auto encode_blocks = static_cast<std::uint32_t>(
        (encode_total + kThreadsPerBlock - 1u) / kThreadsPerBlock);
    encode_literal_rows<<<encode_blocks, kThreadsPerBlock>>>(
        device_features.get(),
        rows,
        features,
        dataset.header.feature_words_per_row,
        literals,
        chunks,
        encoded_x.get());
    check_cuda(cudaGetLastError(), "launch literal encoder");

    if (validation_dataset != nullptr) {
        const auto validation_encode_total =
            validation_dataset->header.frame_count * chunks;
        const auto validation_encode_blocks = static_cast<std::uint32_t>(
            (validation_encode_total + kThreadsPerBlock - 1U) /
            kThreadsPerBlock);
        encode_literal_rows<<<validation_encode_blocks, kThreadsPerBlock>>>(
            validation_device_features.get(),
            validation_dataset->header.frame_count,
            features,
            validation_dataset->header.feature_words_per_row,
            literals,
            chunks,
            validation_encoded_x.get());
        check_cuda(cudaGetLastError(), "launch validation literal encoder");
    }

    const auto state_total = states.size();
    const auto state_blocks = static_cast<std::uint32_t>(
        (state_total + kThreadsPerBlock - 1u) / kThreadsPerBlock);
    initialize_states<<<state_blocks, kThreadsPerBlock>>>(
        states.get(),
        config.clauses,
        config.state_bits,
        chunks,
        literals);
    check_cuda(cudaGetLastError(), "launch state initializer");

    std::mt19937 weight_rng(static_cast<std::uint32_t>(config.seed));
    std::uniform_int_distribution<int> binary_choice(0, 1);
    std::vector<std::int32_t> template_weights(config.clauses);
    for (auto& weight : template_weights) {
        weight = binary_choice(weight_rng) == 0 ? -1 : 1;
    }
    std::vector<std::int32_t> host_weights(
        static_cast<std::size_t>(outputs) * config.clauses);
    for (std::uint32_t output = 0; output < outputs; ++output) {
        std::copy(
            template_weights.begin(),
            template_weights.end(),
            host_weights.begin() + static_cast<std::size_t>(output) * config.clauses);
    }
    weights.copy_from(host_weights.data(), host_weights.size());
    check_cuda(cudaDeviceSynchronize(), "initialize native TM CUDA state");

    int active_blocks_per_sm = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            train_chunk_kernel,
            kThreadsPerBlock,
            0),
        "query cooperative kernel occupancy");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0), "query CUDA device properties");
    const auto maximum_cooperative_blocks =
        static_cast<std::uint32_t>(active_blocks_per_sm * properties.multiProcessorCount);
    const auto desired_blocks =
        (config.clauses + kThreadsPerBlock / kWarpSize - 1u) /
        (kThreadsPerBlock / kWarpSize);
    const auto cooperative_blocks = std::min(desired_blocks, maximum_cooperative_blocks);
    if (cooperative_blocks == 0u) {
        throw std::runtime_error("cooperative TM kernel has zero occupancy");
    }

    TmCudaResult result;
    result.frame_count = rows;
    result.output_count = outputs;
    result.scores.resize(static_cast<std::size_t>(rows) * outputs);
    result.predictions.resize(result.scores.size());
    result.epochs.reserve(config.epochs);

    std::vector<std::uint32_t> shuffled_order = base_order;
    std::mt19937_64 shuffle_rng(config.seed ^ 0xd1b54a32d192ed03ull);
    const auto maximum_ratio = head == TargetHead::activity ? 1.5 : 4.0;
    const auto model_seed = config.seed ^
        (head == TargetHead::activity ? 0ull : 0xa0761d6478bd642full);
    bool has_best_validation = false;
    TmValidationCandidate best_validation{};
    TmValidationEarlyStopping early_stopping;
    early_stopping.patience = config.validation_patience;

    for (std::uint32_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(shuffled_order.begin(), shuffled_order.end(), shuffle_rng);
        order.copy_from(shuffled_order.data(), shuffled_order.size());
        const auto started = std::chrono::steady_clock::now();

        for (std::uint32_t offset = 0;
             offset < shuffled_order.size();
             offset += config.samples_per_launch) {
            const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(
                config.samples_per_launch,
                shuffled_order.size() - offset));
            const auto step_base =
                static_cast<std::uint64_t>(epoch) * shuffled_order.size() + offset;
            auto label_words = dataset.header.label_words_per_row;
            auto output_count = outputs;
            auto clause_count = config.clauses;
            auto chunk_count = chunks;
            auto literal_count = literals;
            auto state_bit_count = config.state_bits;
            auto threshold = config.threshold;
            auto specificity = config.specificity;
            auto negative_samples = config.negative_samples;
            auto sustain_hard_negative_flag =
                use_onset_sustain_hard_negatives ? 1U : 0U;
            auto sustain_hard_negative_probability =
                config.onset_sustain_hard_negative_probability;
            auto sustain_hard_negative_weight_only =
                config.onset_sustain_hard_negative_weight_only ? 1U : 0U;
            auto max_literals = config.max_included_literals;
            auto rotate_output_update_order =
                config.rotate_output_update_order ? 1U : 0U;
            auto random_seed = model_seed;

            // CUDA's cooperative launcher takes addresses of argument values.
            auto* encoded_pointer = encoded_x.get();
            auto* label_pointer = device_labels.get();
            auto* activity_label_pointer = device_activity_labels.get();
            auto* order_pointer = order.get();
            auto* state_pointer = states.get();
            auto* weight_pointer = weights.get();
            auto* clause_pointer = train_clause_outputs.get();
            auto* sums_pointer = class_sums.get();
            auto* nonnegative_counts_pointer = nonnegative_weight_counts.get();
            void* kernel_arguments[] = {
                &encoded_pointer,
                &label_pointer,
                &activity_label_pointer,
                &label_words,
                &order_pointer,
                &offset,
                const_cast<std::uint32_t*>(&count),
                const_cast<std::uint64_t*>(&step_base),
                &output_count,
                &clause_count,
                &chunk_count,
                &literal_count,
                &state_bit_count,
                &threshold,
                &specificity,
                &negative_samples,
                &sustain_hard_negative_flag,
                &sustain_hard_negative_probability,
                &sustain_hard_negative_weight_only,
                &max_literals,
                &rotate_output_update_order,
                &random_seed,
                &state_pointer,
                &weight_pointer,
                &clause_pointer,
                &sums_pointer,
                &nonnegative_counts_pointer,
            };
            check_cuda(
                cudaLaunchCooperativeKernel(
                    reinterpret_cast<void*>(train_chunk_kernel),
                    dim3(cooperative_blocks),
                    dim3(kThreadsPerBlock),
                    kernel_arguments,
                    0,
                    nullptr),
                "launch cooperative TM training chunk");
        }
        check_cuda(cudaDeviceSynchronize(), "train native TM epoch");
        const auto trained = std::chrono::steady_clock::now();

        launch_prediction(
            encoded_x,
            states,
            weights,
            rows,
            outputs,
            config.clauses,
            chunks,
            literals,
            config.state_bits,
            prediction_clause_outputs,
            prediction_scores);
        prediction_scores.copy_to(result.scores.data(), result.scores.size());
        const auto metrics = calibrate_metrics(
            result.scores, host_labels, dataset.header, maximum_ratio);

        HostMetrics validation_metrics;
        if (validation_dataset != nullptr) {
            launch_prediction(
                validation_encoded_x,
                states,
                weights,
                validation_dataset->header.frame_count,
                outputs,
                config.clauses,
                chunks,
                literals,
                config.state_bits,
                validation_clause_outputs,
                validation_prediction_scores);
            validation_prediction_scores.copy_to(
                validation_host_scores.data(), validation_host_scores.size());
            const auto& validation_labels = head == TargetHead::activity
                ? validation_dataset->activity_words
                : validation_dataset->onset_words;
            validation_metrics = calibrate_metrics(
                validation_host_scores,
                validation_labels,
                validation_dataset->header,
                maximum_ratio);
        }
        const auto ended = std::chrono::steady_clock::now();
        TmEpochReport report;
        report.epoch = epoch + 1u;
        report.seconds = std::chrono::duration<double>(ended - started).count();
        report.train_seconds =
            std::chrono::duration<double>(trained - started).count();
        report.precision = metrics.precision;
        report.recall = metrics.recall;
        report.f1 = metrics.f1;
        report.predicted_mean_polyphony = metrics.predicted_mean;
        report.target_mean_polyphony = metrics.target_mean;
        report.score_threshold = metrics.threshold;
        bool should_stop_after_epoch = false;
        if (validation_dataset != nullptr) {
            report.has_validation = true;
            report.validation_precision = validation_metrics.precision;
            report.validation_recall = validation_metrics.recall;
            report.validation_f1 = validation_metrics.f1;
            report.validation_predicted_mean_polyphony =
                validation_metrics.predicted_mean;
            report.validation_target_mean_polyphony =
                validation_metrics.target_mean;
            report.validation_score_threshold = validation_metrics.threshold;

            const TmValidationCandidate candidate{
                report.epoch,
                report.validation_f1,
                report.validation_precision,
                report.validation_recall,
                report.validation_score_threshold,
            };
            if (!has_best_validation ||
                is_better_tm_validation_candidate(candidate, best_validation)) {
                best_states.copy_from_device(states, states.size());
                best_weights.copy_from_device(weights, weights.size());
                best_validation = candidate;
                has_best_validation = true;
                result.best_validation_epoch = report.epoch;
            }
            should_stop_after_epoch = update_tm_validation_early_stopping(
                early_stopping, report.validation_f1);
        }
        result.epochs.push_back(report);
        if (config.verbose) {
            if (validation_dataset == nullptr) {
                // Preserve the original machine-readable log without held-out
                // validation so existing experiment parsers keep working.
                std::cout
                    << "{\"epoch\":" << report.epoch
                    << ",\"seconds\":" << report.seconds
                    << ",\"train_seconds\":" << report.train_seconds
                    << ",\"precision\":" << report.precision
                    << ",\"recall\":" << report.recall
                    << ",\"f1\":" << report.f1
                    << ",\"score_threshold\":" << report.score_threshold
                    << ",\"predicted_mean_polyphony\":"
                    << report.predicted_mean_polyphony
                    << ",\"target_mean_polyphony\":"
                    << report.target_mean_polyphony
                    << "}" << std::endl;
            } else {
                std::cout
                    << "{\"epoch\":" << report.epoch
                    << ",\"seconds\":" << report.seconds
                    << ",\"train_seconds\":" << report.train_seconds
                    << ",\"train_precision\":" << report.precision
                    << ",\"train_recall\":" << report.recall
                    << ",\"train_f1\":" << report.f1
                    << ",\"train_score_threshold\":"
                    << report.score_threshold
                    << ",\"train_predicted_mean_polyphony\":"
                    << report.predicted_mean_polyphony
                    << ",\"train_target_mean_polyphony\":"
                    << report.target_mean_polyphony
                    << ",\"validation_precision\":"
                    << report.validation_precision
                    << ",\"validation_recall\":"
                    << report.validation_recall
                    << ",\"validation_f1\":" << report.validation_f1
                    << ",\"validation_score_threshold\":"
                    << report.validation_score_threshold
                    << ",\"validation_predicted_mean_polyphony\":"
                    << report.validation_predicted_mean_polyphony
                    << ",\"validation_target_mean_polyphony\":"
                    << report.validation_target_mean_polyphony
                    << ",\"best_validation_epoch\":"
                    << result.best_validation_epoch;
                if (config.validation_patience != 0U) {
                    std::cout
                        << ",\"early_stopped\":"
                        << (should_stop_after_epoch ? "true" : "false")
                        << ",\"epochs_executed\":" << report.epoch;
                }
                std::cout << "}" << std::endl;
            }
        }
        if (should_stop_after_epoch) {
            result.early_stopped = true;
            break;
        }
    }
    result.epochs_executed = static_cast<std::uint32_t>(result.epochs.size());

    std::int32_t final_threshold = result.epochs.back().score_threshold;
    std::uint32_t checkpoint_epoch = config.epochs;
    if (validation_dataset != nullptr) {
        if (!has_best_validation) {
            throw std::runtime_error("held-out validation did not select a checkpoint");
        }
        states.copy_from_device(best_states, states.size());
        weights.copy_from_device(best_weights, weights.size());
        check_cuda(cudaDeviceSynchronize(), "restore best validation checkpoint");
        final_threshold = best_validation.score_threshold;
        checkpoint_epoch = best_validation.epoch;

        // Return training-set scores from the same restored state that is saved
        // below.  Its binary predictions intentionally use the held-out
        // threshold, never a threshold refitted on training data.
        launch_prediction(
            encoded_x,
            states,
            weights,
            rows,
            outputs,
            config.clauses,
            chunks,
            literals,
            config.state_bits,
            prediction_clause_outputs,
            prediction_scores);
        prediction_scores.copy_to(result.scores.data(), result.scores.size());
    }
    for (std::size_t index = 0; index < result.scores.size(); ++index) {
        result.predictions[index] =
            result.scores[index] >= final_threshold ? 1u : 0u;
    }

    result.model.head = head == TargetHead::activity
        ? TmModelHead::activity
        : TmModelHead::onset;
    result.model.dimensions.feature_count = features;
    result.model.dimensions.output_count = outputs;
    result.model.dimensions.clause_count = config.clauses;
    result.model.dimensions.state_bits = config.state_bits;
    result.model.training.threshold = config.threshold;
    result.model.training.specificity = config.specificity;
    result.model.training.negative_samples = config.negative_samples;
    result.model.training.type_i_ii_ratio = 1.0f;
    result.model.training.max_included_literals = config.max_included_literals;
    result.model.training.epochs_trained = checkpoint_epoch;
    result.model.training.seed = config.seed;
    result.model.training.feature_negation = true;
    result.model.training.boost_true_positive_feedback = true;
    result.model.training.onset_sustain_hard_negatives =
        use_onset_sustain_hard_negatives;
    result.model.training.onset_sustain_hard_negative_probability =
        use_onset_sustain_hard_negatives
        ? config.onset_sustain_hard_negative_probability
        : 0.0f;
    result.model.training.onset_sustain_hard_negative_weight_only =
        use_onset_sustain_hard_negatives &&
        config.onset_sustain_hard_negative_weight_only;
    result.model.midi.minimum_note = dataset.header.midi_min;
    result.model.midi.maximum_note = dataset.header.midi_max;
    result.model.midi.channel = 1u;
    result.model.midi.audio_sample_rate = dataset.header.sample_rate;
    result.model.midi.analysis_hop_samples = dataset.header.hop_size;
    result.model.score_threshold = final_threshold;
    result.model.feature_fingerprint_sha256 =
        dataset.header.feature_fingerprint_sha256;
    result.model.ta_bitplanes.resize(states.size());
    result.model.weights.resize(weights.size());
    states.copy_to(result.model.ta_bitplanes.data(), result.model.ta_bitplanes.size());
    weights.copy_to(result.model.weights.data(), result.model.weights.size());
    validate_tm_model(result.model);
    return result;
}

TmCudaPrediction predict_tm_cuda(
    const NativeDataset& dataset,
    const NativeTmModel& model,
    const bool allow_legacy_feature_contract) {
    dataset.validate();
    validate_tm_model(model);
    validate_tm_dataset_compatibility(
        dataset, model, allow_legacy_feature_contract);
    if (!cuda_tm_supported()) {
        throw std::runtime_error("CUDA cooperative launch is unavailable");
    }

    if (model.dimensions.feature_count >
        std::numeric_limits<std::uint32_t>::max() / 2u) {
        throw std::invalid_argument("feature count is too large for negated literals");
    }
    const auto rows = dataset.header.frame_count;
    const auto features = model.dimensions.feature_count;
    const auto outputs = model.dimensions.output_count;
    const auto clauses = model.dimensions.clause_count;
    const auto state_bits = model.dimensions.state_bits;
    const auto literals = features * 2u;
    const auto chunks = (literals + 31u) / 32u;

    DeviceBuffer<std::uint64_t> device_features(dataset.feature_words.size());
    DeviceBuffer<std::uint32_t> encoded_x(
        static_cast<std::size_t>(rows) * chunks);
    DeviceBuffer<std::uint32_t> states(model.ta_bitplanes.size());
    DeviceBuffer<std::int32_t> weights(model.weights.size());
    DeviceBuffer<std::uint8_t> clause_outputs(
        static_cast<std::size_t>(rows) * clauses);
    DeviceBuffer<std::int32_t> scores(
        static_cast<std::size_t>(rows) * outputs);

    device_features.copy_from(dataset.feature_words.data(), dataset.feature_words.size());
    states.copy_from(model.ta_bitplanes.data(), model.ta_bitplanes.size());
    weights.copy_from(model.weights.data(), model.weights.size());
    const auto encode_total = rows * chunks;
    const auto encode_blocks = static_cast<std::uint32_t>(
        (encode_total + kThreadsPerBlock - 1u) / kThreadsPerBlock);
    encode_literal_rows<<<encode_blocks, kThreadsPerBlock>>>(
        device_features.get(),
        rows,
        features,
        dataset.header.feature_words_per_row,
        literals,
        chunks,
        encoded_x.get());
    check_cuda(cudaGetLastError(), "launch prediction literal encoder");

    launch_prediction(
        encoded_x,
        states,
        weights,
        rows,
        outputs,
        clauses,
        chunks,
        literals,
        state_bits,
        clause_outputs,
        scores);

    TmCudaPrediction result;
    result.frame_count = rows;
    result.output_count = outputs;
    result.scores.resize(static_cast<std::size_t>(rows) * outputs);
    result.predictions.resize(result.scores.size());
    scores.copy_to(result.scores.data(), result.scores.size());
    for (std::size_t index = 0; index < result.scores.size(); ++index) {
        result.predictions[index] =
            result.scores[index] >= model.score_threshold ? 1u : 0u;
    }
    return result;
}

}  // namespace tmgm::native
