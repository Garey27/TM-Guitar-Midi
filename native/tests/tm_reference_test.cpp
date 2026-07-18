#include "tmgm/tm_reference.hpp"
#include "tmgm/detail/tm_training_semantics.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

tmgm::native::TMReferenceConfig base_config() {
    tmgm::native::TMReferenceConfig config;
    config.feature_count = 3;
    config.output_count = 2;
    config.clause_count = 8;
    config.threshold = 4;
    config.specificity = 3.0;
    config.negative_sampling_q = 2.0;
    config.seed = 20260718U;
    return config;
}

void test_literal_encoding_and_empty_clause_semantics() {
    auto config = base_config();
    tmgm::native::SharedMultiOutputTMReference model(config);

    const auto literals = model.encode_literals({1U, 0U, 1U});
    require(model.literal_count() == 6U, "wrong literal count");
    require(model.literal_word_count() == 1U, "wrong literal word count");
    // x0, x2, and !x1 are literals 0, 2, and 4.
    require(literals[0] == 0x15U, "TMU literal layout was encoded incorrectly");
    require(model.ta_state(0U, 0U) == 127U, "TA did not initialize at state 127");
    require(model.clause_output(0U, literals, false),
            "an empty training clause must be true");
    require(!model.clause_output(0U, literals, true),
            "an empty inference clause must be false");

    model.set_ta_state(0U, 0U, 128U);
    require(model.clause_output(0U, literals, true),
            "included true literal did not fire");
    require(!model.clause_output(0U, model.encode_literals({0U, 0U, 1U}), true),
            "included false literal unexpectedly fired");
}

void test_exact_type_i_type_ii_and_weight_updates() {
    tmgm::native::TMReferenceConfig config;
    config.feature_count = 1;
    config.output_count = 1;
    config.clause_count = 4;
    config.threshold = 2;
    config.specificity = 1.0;
    config.negative_sampling_q = 1.0;
    config.seed = 9U;
    tmgm::native::SharedMultiOutputTMReference model(config);

    // All empty clauses fire during training. The class sum is -2, hence a
    // positive target has update probability exactly one.
    model.set_weight(0U, 0U, 1);
    model.set_weight(0U, 1U, -1);
    model.set_weight(0U, 2U, -1);
    model.set_weight(0U, 3U, -1);
    model.fit_sample_features({1U}, {1U}, 0U, 0U);

    // Positive-weight clause gets Type-Ia: true literal increments and false
    // literal decrements. Negative clauses get Type-II on their false literal.
    require(model.ta_state(0U, 0U) == 128U, "Type-Ia true literal did not increment");
    require(model.ta_state(0U, 1U) == 126U, "Type-Ia false literal did not decrement");
    require(model.ta_state(1U, 0U) == 127U, "Type-II changed a true literal");
    require(model.ta_state(1U, 1U) == 128U, "Type-II false literal did not increment");

    // Weight update uses the original clause outputs, after TA feedback.
    require(model.weight(0U, 0U) == 2, "positive weight update was wrong");
    require(model.weight(0U, 1U) == 0, "negative weight update was wrong");
    require(model.weight(0U, 2U) == 0, "negative weight update was wrong");
    require(model.weight(0U, 3U) == 0, "negative weight update was wrong");

    // The bit-sliced counter operations must saturate at both endpoints.
    tmgm::native::SharedMultiOutputTMReference saturated(config);
    saturated.set_weight(0U, 0U, 1);
    saturated.set_weight(0U, 1U, -1);
    saturated.set_weight(0U, 2U, -1);
    saturated.set_weight(0U, 3U, -1);
    saturated.set_ta_state(0U, 0U, 255U);
    saturated.set_ta_state(0U, 1U, 0U);
    saturated.fit_sample_features({1U}, {1U}, 0U, 0U);
    require(saturated.ta_state(0U, 0U) == 255U, "TA increment overflowed");
    require(saturated.ta_state(0U, 1U) == 0U, "TA decrement underflowed");
}

std::uint32_t included_literal_count(
    const tmgm::native::SharedMultiOutputTMReference& model,
    const std::uint32_t clause) {
    std::uint32_t count = 0U;
    for (std::uint32_t literal = 0U; literal < model.literal_count(); ++literal) {
        count += model.ta_state(clause, literal) >= 128U ? 1U : 0U;
    }
    return count;
}

void test_max_included_literals_is_a_strict_cap() {
    tmgm::native::TMReferenceConfig config;
    config.feature_count = 40U;
    config.output_count = 1U;
    config.clause_count = 4U;
    config.threshold = 2;
    config.specificity = 1.0;
    config.negative_sampling_q = 1.0;
    config.max_included_literals = 3U;
    config.seed = 91U;
    tmgm::native::SharedMultiOutputTMReference model(config);

    // The clipped class sum is -threshold, making feedback probability one.
    // Clause 0 receives Type-I and the remaining clauses receive Type-II. From
    // initial state 127, the historical bit-sliced update moved forty literals
    // across the action boundary in either path despite max=3.
    model.set_weight(0U, 0U, 1);
    model.set_weight(0U, 1U, -1);
    model.set_weight(0U, 2U, -1);
    model.set_weight(0U, 3U, -1);
    model.fit_sample_features(
        std::vector<std::uint8_t>(config.feature_count, 1U),
        {1U},
        0U,
        0U);

    for (std::uint32_t clause = 0U; clause < config.clause_count; ++clause) {
        require(included_literal_count(model, clause) ==
                    config.max_included_literals,
                "Type-I/II feedback exceeded the strict literal cap");
    }
    require(model.ta_state(0U, 0U) == 128U,
            "Type-I cap did not admit the lowest eligible literal");
    require(model.ta_state(1U, config.feature_count) == 128U,
            "Type-II cap did not admit the lowest eligible literal");
}

void test_stateless_reproducibility_and_tiny_learning() {
    tmgm::native::TMReferenceConfig config;
    config.feature_count = 2;
    config.output_count = 2;
    config.clause_count = 64;
    config.threshold = 15;
    config.specificity = 3.9;
    config.negative_sampling_q = 2.0;
    config.seed = 0x5eedU;

    const std::vector<std::vector<std::uint8_t>> features = {
        {0U, 0U}, {0U, 1U}, {1U, 0U}, {1U, 1U}};
    // Output 0 copies x0. Output 1 is XOR.
    const std::vector<std::vector<std::uint8_t>> targets = {
        {0U, 0U}, {0U, 1U}, {1U, 1U}, {1U, 0U}};

    tmgm::native::SharedMultiOutputTMReference left(config);
    tmgm::native::SharedMultiOutputTMReference right(config);
    for (std::uint64_t epoch = 0; epoch < 200U; ++epoch) {
        left.fit_epoch(features, targets, epoch, true);
        right.fit_epoch(features, targets, epoch, true);
    }
    require(left.ta_bitplanes() == right.ta_bitplanes(),
            "counter-based training was not reproducible (TA state)");
    require(left.weights() == right.weights(),
            "counter-based training was not reproducible (weights)");

    for (std::size_t row = 0; row < features.size(); ++row) {
        require(left.predict_features(features[row]) == targets[row],
                "reference TM failed to learn the tiny deterministic corpus");
    }
}

void test_allocation_free_negative_permutation_and_weight_guards() {
    using tmgm::native::detail::affine_permuted_index;
    using tmgm::native::detail::advance_affine_permutation;
    using tmgm::native::detail::coprime_permutation_stride;
    using tmgm::native::detail::is_onset_sustain_hard_negative;
    using tmgm::native::detail::positive_weight_update_allowed;
    using tmgm::native::detail::saturating_unit_weight_update;

    for (std::uint32_t count = 1U; count <= 128U; ++count) {
        for (std::uint32_t candidate = 0U; candidate < count * 2U; ++candidate) {
            const auto stride = coprime_permutation_stride(count, candidate);
            std::vector<std::uint8_t> seen(count, 0U);
            auto incremental = (candidate * 17U + 3U) % count;
            for (std::uint32_t position = 0U; position < count; ++position) {
                const auto output = affine_permuted_index(
                    count, position, candidate * 17U + 3U, stride);
                require(output == incremental,
                        "incremental affine permutation disagrees with definition");
                require(output < count, "affine permutation produced an invalid output");
                require(seen[output] == 0U, "affine permutation repeated an output");
                seen[output] = 1U;
                incremental = advance_affine_permutation(count, incremental, stride);
            }
            for (const auto visited : seen) {
                require(visited != 0U, "affine permutation skipped an output");
            }
        }
    }

    require(positive_weight_update_allowed(255U, 256U),
            "positive weight gate closed too early");
    require(!positive_weight_update_allowed(256U, 256U),
            "positive weight gate exceeded the TMU maximum");
    require(saturating_unit_weight_update(
                std::numeric_limits<std::int32_t>::max(), true) ==
                std::numeric_limits<std::int32_t>::max(),
            "positive weight overflow did not saturate");
    require(saturating_unit_weight_update(
                std::numeric_limits<std::int32_t>::min(), false) ==
                std::numeric_limits<std::int32_t>::min(),
            "negative weight overflow did not saturate");
    require(saturating_unit_weight_update(7, true) == 8,
            "ordinary positive weight update was wrong");
    require(saturating_unit_weight_update(-7, false) == -8,
            "ordinary negative weight update was wrong");

    require(is_onset_sustain_hard_negative(false, true, true),
            "active sustain was not selected as an onset hard negative");
    require(!is_onset_sustain_hard_negative(true, true, true),
            "a true onset was incorrectly selected as a hard negative");
    require(!is_onset_sustain_hard_negative(false, false, true),
            "silence was incorrectly selected as a sustain hard negative");
    require(!is_onset_sustain_hard_negative(false, true, false),
            "disabled hard-negative policy still selected an output");
}

}  // namespace

int main() {
    try {
        test_literal_encoding_and_empty_clause_semantics();
        test_exact_type_i_type_ii_and_weight_updates();
        test_max_included_literals_is_a_strict_cap();
        test_stateless_reproducibility_and_tiny_learning();
        test_allocation_free_negative_permutation_and_weight_guards();
        std::cout << "TM shared-clause CPU reference tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
