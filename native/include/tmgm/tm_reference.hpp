#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tmgm::native {

// Deterministic CPU reference for the shared-clause multi-output classifier
// used by TMU's experimental TMCoalesceMultiOuputClassifier. This is intended
// as a correctness oracle for the CUDA trainer, not as the fast production
// implementation.
struct TMReferenceConfig {
    std::uint32_t feature_count = 0;
    std::uint32_t output_count = 0;
    std::uint32_t clause_count = 0;
    std::int32_t threshold = 0;
    double specificity = 0.0;
    double type_i_ii_ratio = 1.0;
    double negative_sampling_q = 1.0;
    std::uint32_t max_included_literals = 0;  // Zero means all literals.
    std::uint32_t max_positive_clauses = 0;  // Zero means all clauses.
    std::uint64_t seed = 0;
    bool feature_negation = true;
    bool boost_true_positive_feedback = true;
};

class SharedMultiOutputTMReference {
public:
    static constexpr std::uint32_t kStateBits = 8;
    static constexpr std::uint8_t kInitialTAState = 127;

    explicit SharedMultiOutputTMReference(TMReferenceConfig config);

    [[nodiscard]] const TMReferenceConfig& config() const noexcept;
    [[nodiscard]] std::uint32_t literal_count() const noexcept;
    [[nodiscard]] std::uint32_t literal_word_count() const noexcept;

    // TMU literal layout: positive features first, followed by their negations.
    [[nodiscard]] std::vector<std::uint32_t> encode_literals(
        const std::vector<std::uint8_t>& features) const;

    // In training mode an empty clause is true. At inference an empty clause is
    // false, matching calculate_clause_outputs_update/predict in TMU.
    [[nodiscard]] bool clause_output(
        std::uint32_t clause,
        const std::vector<std::uint32_t>& literal_words,
        bool inference) const;
    [[nodiscard]] std::vector<std::uint8_t> clause_outputs(
        const std::vector<std::uint32_t>& literal_words,
        bool inference) const;

    [[nodiscard]] std::vector<std::int32_t> scores(
        const std::vector<std::uint32_t>& literal_words,
        bool clip_to_threshold = false) const;
    [[nodiscard]] std::vector<std::uint8_t> predict(
        const std::vector<std::uint32_t>& literal_words) const;
    [[nodiscard]] std::vector<std::uint8_t> predict_features(
        const std::vector<std::uint8_t>& features) const;

    // The epoch and sample index are part of every random key. Randomness is
    // counter-based/stateless, so an identical update schedule is exactly
    // reproducible and can be implemented identically in CUDA.
    void fit_sample(
        const std::vector<std::uint32_t>& literal_words,
        const std::vector<std::uint8_t>& targets,
        std::uint64_t epoch,
        std::uint64_t sample_index);
    void fit_sample_features(
        const std::vector<std::uint8_t>& features,
        const std::vector<std::uint8_t>& targets,
        std::uint64_t epoch,
        std::uint64_t sample_index);
    void fit_epoch(
        const std::vector<std::vector<std::uint8_t>>& features,
        const std::vector<std::vector<std::uint8_t>>& targets,
        std::uint64_t epoch,
        bool shuffle = true);

    [[nodiscard]] std::uint8_t ta_state(
        std::uint32_t clause,
        std::uint32_t literal) const;
    void set_ta_state(
        std::uint32_t clause,
        std::uint32_t literal,
        std::uint8_t state);
    [[nodiscard]] std::int32_t weight(
        std::uint32_t output,
        std::uint32_t clause) const;
    void set_weight(
        std::uint32_t output,
        std::uint32_t clause,
        std::int32_t weight);

    // CUDA parity tests compare these arrays directly. TA layout is
    // [clause][bit-plane][literal-word], weights are [output][clause].
    [[nodiscard]] const std::vector<std::uint32_t>& ta_bitplanes() const noexcept;
    [[nodiscard]] const std::vector<std::int32_t>& weights() const noexcept;

private:
    enum class RandomDomain : std::uint64_t;

    [[nodiscard]] std::size_t ta_index(
        std::uint32_t clause,
        std::uint32_t bit,
        std::uint32_t word) const noexcept;
    [[nodiscard]] std::size_t weight_index(
        std::uint32_t output,
        std::uint32_t clause) const noexcept;
    [[nodiscard]] std::uint32_t valid_literal_mask(std::uint32_t word) const noexcept;
    [[nodiscard]] std::uint32_t feedback_literal_mask(std::uint32_t word) const noexcept;
    void validate_literal_words(const std::vector<std::uint32_t>& literal_words) const;

    [[nodiscard]] std::uint64_t random_u64(
        RandomDomain domain,
        std::uint64_t epoch,
        std::uint64_t sample,
        std::uint64_t output,
        std::uint64_t clause,
        std::uint64_t item = 0) const noexcept;
    [[nodiscard]] bool bernoulli(
        double probability,
        RandomDomain domain,
        std::uint64_t epoch,
        std::uint64_t sample,
        std::uint64_t output,
        std::uint64_t clause,
        std::uint64_t item = 0) const noexcept;

    [[nodiscard]] std::uint32_t included_literal_count(std::uint32_t clause) const noexcept;
    [[nodiscard]] std::uint32_t cap_literal_action_transitions(
        std::uint32_t clause,
        std::uint32_t word,
        std::uint32_t increment_mask,
        std::uint32_t& remaining_literal_slots) const noexcept;
    void increment_ta_word(std::uint32_t clause, std::uint32_t word, std::uint32_t mask);
    void decrement_ta_word(std::uint32_t clause, std::uint32_t word, std::uint32_t mask);
    void type_i_feedback(
        std::uint32_t output,
        std::uint32_t clause,
        double update_probability,
        const std::vector<std::uint32_t>& literal_words,
        std::uint64_t epoch,
        std::uint64_t sample);
    void type_ii_feedback(
        std::uint32_t output,
        std::uint32_t clause,
        double update_probability,
        const std::vector<std::uint32_t>& literal_words,
        std::uint64_t epoch,
        std::uint64_t sample);
    void update_output(
        std::uint32_t output,
        bool target,
        double update_probability,
        const std::vector<std::uint8_t>& original_clause_outputs,
        const std::vector<std::uint32_t>& literal_words,
        std::uint64_t epoch,
        std::uint64_t sample);

    TMReferenceConfig config_;
    std::uint32_t literal_count_ = 0;
    std::uint32_t literal_word_count_ = 0;
    double type_i_probability_scale_ = 1.0;
    double type_ii_probability_scale_ = 1.0;
    std::vector<std::uint32_t> ta_bitplanes_;
    std::vector<std::int32_t> weights_;
};

}  // namespace tmgm::native
