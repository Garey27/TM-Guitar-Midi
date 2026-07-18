#include "tmgm/tm_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace tmgm::native {
namespace {

constexpr std::uint64_t kHashIncrement = 0x9e3779b97f4a7c15ULL;

std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += kHashIncrement;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint32_t popcount32(std::uint32_t value) noexcept {
    std::uint32_t count = 0;
    while (value != 0U) {
        value &= value - 1U;
        ++count;
    }
    return count;
}

double clamp_probability(double probability) noexcept {
    return std::max(0.0, std::min(1.0, probability));
}

}  // namespace

enum class SharedMultiOutputTMReference::RandomDomain : std::uint64_t {
    WeightInitialization = 0x166aa5b21b84a395ULL,
    TypeIClause = 0x77e35d809f28f294ULL,
    TypeISpecificity = 0xe12f9875a4c7d916ULL,
    TypeIIClause = 0xa6c896683b8ef41dULL,
    WeightUpdate = 0xc457d9a43f1298b7ULL,
    NegativeSelection = 0x81f672ae9053c4dbULL,
    NegativeOrder = 0x3bd81cfa21e67509ULL,
    EpochOrder = 0xf5a61d907c43b82eULL,
};

SharedMultiOutputTMReference::SharedMultiOutputTMReference(TMReferenceConfig config)
    : config_(std::move(config)) {
    if (config_.feature_count == 0U || config_.output_count == 0U ||
        config_.clause_count == 0U) {
        throw std::invalid_argument("feature, output, and clause counts must be positive");
    }
    if (config_.feature_count > std::numeric_limits<std::uint32_t>::max() / 2U) {
        throw std::invalid_argument("feature count is too large");
    }
    if (config_.threshold <= 0) {
        throw std::invalid_argument("threshold must be positive");
    }
    if (!std::isfinite(config_.specificity) || config_.specificity <= 0.0) {
        throw std::invalid_argument("specificity must be finite and positive");
    }
    if (!std::isfinite(config_.type_i_ii_ratio) || config_.type_i_ii_ratio <= 0.0) {
        throw std::invalid_argument("type-I/type-II ratio must be finite and positive");
    }
    if (!std::isfinite(config_.negative_sampling_q) || config_.negative_sampling_q < 0.0) {
        throw std::invalid_argument("negative sampling q must be finite and non-negative");
    }

    // BaseClauseBank always reserves both positive and negated literals. When
    // feature_negation is false TMU merely deactivates feedback for the latter.
    literal_count_ = config_.feature_count * 2U;
    literal_word_count_ = (literal_count_ + 31U) / 32U;
    if (config_.max_included_literals == 0U) {
        config_.max_included_literals = literal_count_;
    } else if (config_.max_included_literals > literal_count_) {
        throw std::invalid_argument("max included literals exceeds literal count");
    }
    if (config_.max_positive_clauses == 0U) {
        config_.max_positive_clauses = config_.clause_count;
    } else if (config_.max_positive_clauses > config_.clause_count) {
        throw std::invalid_argument("max positive clauses exceeds clause count");
    }

    if (config_.type_i_ii_ratio >= 1.0) {
        type_i_probability_scale_ = 1.0;
        type_ii_probability_scale_ = 1.0 / config_.type_i_ii_ratio;
    } else {
        type_i_probability_scale_ = config_.type_i_ii_ratio;
        type_ii_probability_scale_ = 1.0;
    }

    const auto ta_words = static_cast<std::size_t>(config_.clause_count) *
                          kStateBits * literal_word_count_;
    ta_bitplanes_.assign(ta_words, 0U);
    // Scalar state 127: lower seven bit-planes set, include/MSB clear.
    for (std::uint32_t clause = 0; clause < config_.clause_count; ++clause) {
        for (std::uint32_t bit = 0; bit + 1U < kStateBits; ++bit) {
            for (std::uint32_t word = 0; word < literal_word_count_; ++word) {
                ta_bitplanes_[ta_index(clause, bit, word)] = 0xffffffffU;
            }
        }
    }

    weights_.resize(
        static_cast<std::size_t>(config_.output_count) * config_.clause_count);
    // TMU supplies the same random +/-1 initialization vector to every output
    // WeightBank (each WeightBank then copies it).
    for (std::uint32_t clause = 0; clause < config_.clause_count; ++clause) {
        const auto initial = (random_u64(
                                  RandomDomain::WeightInitialization,
                                  0U,
                                  0U,
                                  0U,
                                  clause) &
                              1U) != 0U
                                 ? 1
                                 : -1;
        for (std::uint32_t output = 0; output < config_.output_count; ++output) {
            weights_[weight_index(output, clause)] = initial;
        }
    }
}

const TMReferenceConfig& SharedMultiOutputTMReference::config() const noexcept {
    return config_;
}

std::uint32_t SharedMultiOutputTMReference::literal_count() const noexcept {
    return literal_count_;
}

std::uint32_t SharedMultiOutputTMReference::literal_word_count() const noexcept {
    return literal_word_count_;
}

std::vector<std::uint32_t> SharedMultiOutputTMReference::encode_literals(
    const std::vector<std::uint8_t>& features) const {
    if (features.size() != config_.feature_count) {
        throw std::invalid_argument("feature row has the wrong width");
    }
    std::vector<std::uint32_t> encoded(literal_word_count_, 0U);
    for (std::uint32_t feature = 0; feature < config_.feature_count; ++feature) {
        const auto value = features[feature];
        if (value > 1U) {
            throw std::invalid_argument("features must be binary");
        }
        const auto literal = value != 0U ? feature : feature + config_.feature_count;
        encoded[literal / 32U] |= 1U << (literal % 32U);
    }
    return encoded;
}

bool SharedMultiOutputTMReference::clause_output(
    std::uint32_t clause,
    const std::vector<std::uint32_t>& literal_words,
    bool inference) const {
    if (clause >= config_.clause_count) {
        throw std::out_of_range("clause index is out of range");
    }
    validate_literal_words(literal_words);
    bool has_included_literal = false;
    for (std::uint32_t word = 0; word < literal_word_count_; ++word) {
        const auto included =
            ta_bitplanes_[ta_index(clause, kStateBits - 1U, word)] &
            valid_literal_mask(word);
        has_included_literal = has_included_literal || included != 0U;
        if ((included & ~literal_words[word]) != 0U) {
            return false;
        }
    }
    return !inference || has_included_literal;
}

std::vector<std::uint8_t> SharedMultiOutputTMReference::clause_outputs(
    const std::vector<std::uint32_t>& literal_words,
    bool inference) const {
    validate_literal_words(literal_words);
    std::vector<std::uint8_t> result(config_.clause_count, 0U);
    for (std::uint32_t clause = 0; clause < config_.clause_count; ++clause) {
        result[clause] = clause_output(clause, literal_words, inference) ? 1U : 0U;
    }
    return result;
}

std::vector<std::int32_t> SharedMultiOutputTMReference::scores(
    const std::vector<std::uint32_t>& literal_words,
    bool clip_to_threshold) const {
    const auto outputs = clause_outputs(literal_words, true);
    std::vector<std::int32_t> result(config_.output_count, 0);
    for (std::uint32_t output = 0; output < config_.output_count; ++output) {
        std::int64_t sum = 0;
        for (std::uint32_t clause = 0; clause < config_.clause_count; ++clause) {
            if (outputs[clause] != 0U) {
                sum += weights_[weight_index(output, clause)];
            }
        }
        if (clip_to_threshold) {
            sum = std::max<std::int64_t>(-config_.threshold,
                                         std::min<std::int64_t>(config_.threshold, sum));
        } else {
            sum = std::max<std::int64_t>(std::numeric_limits<std::int32_t>::min(),
                                         std::min<std::int64_t>(
                                             std::numeric_limits<std::int32_t>::max(), sum));
        }
        result[output] = static_cast<std::int32_t>(sum);
    }
    return result;
}

std::vector<std::uint8_t> SharedMultiOutputTMReference::predict(
    const std::vector<std::uint32_t>& literal_words) const {
    const auto class_scores = scores(literal_words, false);
    std::vector<std::uint8_t> result(class_scores.size(), 0U);
    for (std::size_t output = 0; output < class_scores.size(); ++output) {
        result[output] = class_scores[output] >= 0 ? 1U : 0U;
    }
    return result;
}

std::vector<std::uint8_t> SharedMultiOutputTMReference::predict_features(
    const std::vector<std::uint8_t>& features) const {
    return predict(encode_literals(features));
}

void SharedMultiOutputTMReference::fit_sample(
    const std::vector<std::uint32_t>& literal_words,
    const std::vector<std::uint8_t>& targets,
    std::uint64_t epoch,
    std::uint64_t sample_index) {
    validate_literal_words(literal_words);
    if (targets.size() != config_.output_count) {
        throw std::invalid_argument("target row has the wrong width");
    }
    for (const auto target : targets) {
        if (target > 1U) {
            throw std::invalid_argument("targets must be binary");
        }
    }

    // TMU computes these once before any output feedback for this sample.
    const auto original_clause_outputs = clause_outputs(literal_words, false);
    std::vector<std::int32_t> clipped_scores(config_.output_count, 0);
    std::vector<double> update_probabilities(config_.output_count, 0.0);
    for (std::uint32_t output = 0; output < config_.output_count; ++output) {
        std::int64_t sum = 0;
        for (std::uint32_t clause = 0; clause < config_.clause_count; ++clause) {
            if (original_clause_outputs[clause] != 0U) {
                sum += weights_[weight_index(output, clause)];
            }
        }
        sum = std::max<std::int64_t>(-config_.threshold,
                                     std::min<std::int64_t>(config_.threshold, sum));
        clipped_scores[output] = static_cast<std::int32_t>(sum);
        const auto sign = targets[output] != 0U ? 1.0 : -1.0;
        update_probabilities[output] = clamp_probability(
            (static_cast<double>(config_.threshold) -
             sign * static_cast<double>(clipped_scores[output])) /
            (2.0 * static_cast<double>(config_.threshold)));
    }

    // TMU updates every positive output in ascending index order.
    for (std::uint32_t output = 0; output < config_.output_count; ++output) {
        if (targets[output] != 0U) {
            update_output(output,
                          true,
                          update_probabilities[output],
                          original_clause_outputs,
                          literal_words,
                          epoch,
                          sample_index);
        }
    }

    // It shuffles negatives, then accepts each with q/(C-1). Counter-based
    // random sort keys give the same deterministic ordered operation without a
    // mutable PRNG stream.
    std::vector<std::uint32_t> negatives;
    negatives.reserve(config_.output_count);
    for (std::uint32_t output = 0; output < config_.output_count; ++output) {
        if (targets[output] == 0U) {
            negatives.push_back(output);
        }
    }
    std::stable_sort(
        negatives.begin(), negatives.end(), [&](std::uint32_t left, std::uint32_t right) {
            return random_u64(RandomDomain::NegativeOrder,
                              epoch,
                              sample_index,
                              left,
                              0U) <
                   random_u64(RandomDomain::NegativeOrder,
                              epoch,
                              sample_index,
                              right,
                              0U);
        });
    const auto selection_probability = clamp_probability(
        config_.negative_sampling_q /
        static_cast<double>(std::max<std::uint32_t>(1U, config_.output_count - 1U)));
    for (const auto output : negatives) {
        if (!bernoulli(selection_probability,
                       RandomDomain::NegativeSelection,
                       epoch,
                       sample_index,
                       output,
                       0U)) {
            continue;
        }
        update_output(output,
                      false,
                      update_probabilities[output],
                      original_clause_outputs,
                      literal_words,
                      epoch,
                      sample_index);
    }
}

void SharedMultiOutputTMReference::fit_sample_features(
    const std::vector<std::uint8_t>& features,
    const std::vector<std::uint8_t>& targets,
    std::uint64_t epoch,
    std::uint64_t sample_index) {
    fit_sample(encode_literals(features), targets, epoch, sample_index);
}

void SharedMultiOutputTMReference::fit_epoch(
    const std::vector<std::vector<std::uint8_t>>& features,
    const std::vector<std::vector<std::uint8_t>>& targets,
    std::uint64_t epoch,
    bool shuffle) {
    if (features.size() != targets.size()) {
        throw std::invalid_argument("feature and target row counts differ");
    }
    std::vector<std::size_t> order(features.size());
    std::iota(order.begin(), order.end(), 0U);
    if (shuffle) {
        std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
            return random_u64(RandomDomain::EpochOrder, epoch, left, 0U, 0U) <
                   random_u64(RandomDomain::EpochOrder, epoch, right, 0U, 0U);
        });
    }
    for (const auto row : order) {
        fit_sample_features(features[row], targets[row], epoch, row);
    }
}

std::uint8_t SharedMultiOutputTMReference::ta_state(
    std::uint32_t clause,
    std::uint32_t literal) const {
    if (clause >= config_.clause_count || literal >= literal_count_) {
        throw std::out_of_range("TA index is out of range");
    }
    const auto word = literal / 32U;
    const auto mask = 1U << (literal % 32U);
    std::uint8_t result = 0U;
    for (std::uint32_t bit = 0; bit < kStateBits; ++bit) {
        if ((ta_bitplanes_[ta_index(clause, bit, word)] & mask) != 0U) {
            result = static_cast<std::uint8_t>(result | (1U << bit));
        }
    }
    return result;
}

void SharedMultiOutputTMReference::set_ta_state(
    std::uint32_t clause,
    std::uint32_t literal,
    std::uint8_t state) {
    if (clause >= config_.clause_count || literal >= literal_count_) {
        throw std::out_of_range("TA index is out of range");
    }
    const auto word = literal / 32U;
    const auto mask = 1U << (literal % 32U);
    for (std::uint32_t bit = 0; bit < kStateBits; ++bit) {
        auto& plane = ta_bitplanes_[ta_index(clause, bit, word)];
        if ((state & (1U << bit)) != 0U) {
            plane |= mask;
        } else {
            plane &= ~mask;
        }
    }
}

std::int32_t SharedMultiOutputTMReference::weight(
    std::uint32_t output,
    std::uint32_t clause) const {
    if (output >= config_.output_count || clause >= config_.clause_count) {
        throw std::out_of_range("weight index is out of range");
    }
    return weights_[weight_index(output, clause)];
}

void SharedMultiOutputTMReference::set_weight(
    std::uint32_t output,
    std::uint32_t clause,
    std::int32_t value) {
    if (output >= config_.output_count || clause >= config_.clause_count) {
        throw std::out_of_range("weight index is out of range");
    }
    weights_[weight_index(output, clause)] = value;
}

const std::vector<std::uint32_t>&
SharedMultiOutputTMReference::ta_bitplanes() const noexcept {
    return ta_bitplanes_;
}

const std::vector<std::int32_t>& SharedMultiOutputTMReference::weights() const noexcept {
    return weights_;
}

std::size_t SharedMultiOutputTMReference::ta_index(
    std::uint32_t clause,
    std::uint32_t bit,
    std::uint32_t word) const noexcept {
    return (static_cast<std::size_t>(clause) * kStateBits + bit) *
               literal_word_count_ +
           word;
}

std::size_t SharedMultiOutputTMReference::weight_index(
    std::uint32_t output,
    std::uint32_t clause) const noexcept {
    return static_cast<std::size_t>(output) * config_.clause_count + clause;
}

std::uint32_t SharedMultiOutputTMReference::valid_literal_mask(
    std::uint32_t word) const noexcept {
    if (word + 1U < literal_word_count_) {
        return 0xffffffffU;
    }
    const auto tail = literal_count_ % 32U;
    return tail == 0U ? 0xffffffffU : (1U << tail) - 1U;
}

std::uint32_t SharedMultiOutputTMReference::feedback_literal_mask(
    std::uint32_t word) const noexcept {
    auto mask = valid_literal_mask(word);
    if (config_.feature_negation) {
        return mask;
    }
    const auto first_literal = word * 32U;
    if (first_literal >= config_.feature_count) {
        return 0U;
    }
    const auto remaining = config_.feature_count - first_literal;
    if (remaining < 32U) {
        mask &= (1U << remaining) - 1U;
    }
    return mask;
}

void SharedMultiOutputTMReference::validate_literal_words(
    const std::vector<std::uint32_t>& literal_words) const {
    if (literal_words.size() != literal_word_count_) {
        throw std::invalid_argument("encoded literal row has the wrong width");
    }
    if ((literal_words.back() & ~valid_literal_mask(literal_word_count_ - 1U)) != 0U) {
        throw std::invalid_argument("encoded literal row has non-zero padding bits");
    }
}

std::uint64_t SharedMultiOutputTMReference::random_u64(
    RandomDomain domain,
    std::uint64_t epoch,
    std::uint64_t sample,
    std::uint64_t output,
    std::uint64_t clause,
    std::uint64_t item) const noexcept {
    auto hash = splitmix64(config_.seed ^ static_cast<std::uint64_t>(domain));
    hash = splitmix64(hash ^ splitmix64(epoch + 0x243f6a8885a308d3ULL));
    hash = splitmix64(hash ^ splitmix64(sample + 0x13198a2e03707344ULL));
    hash = splitmix64(hash ^ splitmix64(output + 0xa4093822299f31d0ULL));
    hash = splitmix64(hash ^ splitmix64(clause + 0x082efa98ec4e6c89ULL));
    return splitmix64(hash ^ splitmix64(item + 0x452821e638d01377ULL));
}

bool SharedMultiOutputTMReference::bernoulli(
    double probability,
    RandomDomain domain,
    std::uint64_t epoch,
    std::uint64_t sample,
    std::uint64_t output,
    std::uint64_t clause,
    std::uint64_t item) const noexcept {
    if (probability <= 0.0) {
        return false;
    }
    if (probability >= 1.0) {
        return true;
    }
    const auto random = random_u64(domain, epoch, sample, output, clause, item);
    const auto unit = static_cast<double>(random >> 11U) * 0x1.0p-53;
    return unit < probability;
}

std::uint32_t SharedMultiOutputTMReference::included_literal_count(
    std::uint32_t clause) const noexcept {
    std::uint32_t count = 0U;
    for (std::uint32_t word = 0; word < literal_word_count_; ++word) {
        count += popcount32(
            ta_bitplanes_[ta_index(clause, kStateBits - 1U, word)] &
            valid_literal_mask(word));
    }
    return count;
}

std::uint32_t SharedMultiOutputTMReference::cap_literal_action_transitions(
    const std::uint32_t clause,
    const std::uint32_t word,
    const std::uint32_t increment_mask,
    std::uint32_t& remaining_literal_slots) const noexcept {
    // Only state (2^(N-1) - 1) enters Include on the next increment: every
    // lower bit is one and the action/MSB is zero.
    auto enters_include = increment_mask &
        ~ta_bitplanes_[ta_index(clause, kStateBits - 1U, word)];
    for (std::uint32_t bit = 0; bit + 1U < kStateBits; ++bit) {
        enters_include &= ta_bitplanes_[ta_index(clause, bit, word)];
    }

    auto allowed_to_enter = 0U;
    for (std::uint32_t bit = 0;
         bit < 32U && remaining_literal_slots != 0U;
         ++bit) {
        const auto bit_mask = 1U << bit;
        if ((enters_include & bit_mask) != 0U) {
            allowed_to_enter |= bit_mask;
            --remaining_literal_slots;
        }
    }
    return (increment_mask & ~enters_include) | allowed_to_enter;
}

void SharedMultiOutputTMReference::increment_ta_word(
    std::uint32_t clause,
    std::uint32_t word,
    std::uint32_t mask) {
    auto carry = mask & feedback_literal_mask(word);
    for (std::uint32_t bit = 0; bit < kStateBits; ++bit) {
        auto& plane = ta_bitplanes_[ta_index(clause, bit, word)];
        const auto next = plane & carry;
        plane ^= carry;
        carry = next;
    }
    if (carry != 0U) {
        for (std::uint32_t bit = 0; bit < kStateBits; ++bit) {
            ta_bitplanes_[ta_index(clause, bit, word)] |= carry;
        }
    }
}

void SharedMultiOutputTMReference::decrement_ta_word(
    std::uint32_t clause,
    std::uint32_t word,
    std::uint32_t mask) {
    auto borrow = mask & feedback_literal_mask(word);
    for (std::uint32_t bit = 0; bit < kStateBits; ++bit) {
        auto& plane = ta_bitplanes_[ta_index(clause, bit, word)];
        const auto next = ~plane & borrow;
        plane ^= borrow;
        borrow = next;
    }
    if (borrow != 0U) {
        for (std::uint32_t bit = 0; bit < kStateBits; ++bit) {
            ta_bitplanes_[ta_index(clause, bit, word)] &= ~borrow;
        }
    }
}

void SharedMultiOutputTMReference::type_i_feedback(
    std::uint32_t output,
    std::uint32_t clause,
    double update_probability,
    const std::vector<std::uint32_t>& literal_words,
    std::uint64_t epoch,
    std::uint64_t sample) {
    if (!bernoulli(update_probability * type_i_probability_scale_,
                   RandomDomain::TypeIClause,
                   epoch,
                   sample,
                   output,
                   clause)) {
        return;
    }

    const auto fires = clause_output(clause, literal_words, false);
    const auto included_literals = included_literal_count(clause);
    const auto type_ia =
        fires && included_literals < config_.max_included_literals;
    auto remaining_literal_slots = type_ia
        ? config_.max_included_literals - included_literals
        : 0U;
    const auto specificity_probability = clamp_probability(1.0 / config_.specificity);

    for (std::uint32_t word = 0; word < literal_word_count_; ++word) {
        const auto active = feedback_literal_mask(word);
        std::uint32_t specificity_mask = 0U;
        for (std::uint32_t bit = 0; bit < 32U; ++bit) {
            const auto bit_mask = 1U << bit;
            if ((active & bit_mask) == 0U) {
                continue;
            }
            const auto literal = static_cast<std::uint64_t>(word) * 32U + bit;
            if (bernoulli(specificity_probability,
                          RandomDomain::TypeISpecificity,
                          epoch,
                          sample,
                          output,
                          clause,
                          literal)) {
                specificity_mask |= bit_mask;
            }
        }

        if (type_ia) {
            auto increment = active & literal_words[word];
            if (!config_.boost_true_positive_feedback) {
                increment &= ~specificity_mask;
            }

            // A bit-sliced update may otherwise move many state-127 automata
            // across the action boundary in one operation.
            increment = cap_literal_action_transitions(
                clause, word, increment, remaining_literal_slots);
            increment_ta_word(clause, word, increment);
            decrement_ta_word(
                clause, word, active & ~literal_words[word] & specificity_mask);
        } else {
            // TMU Type-Ib: every active literal receives the 1/s penalty,
            // regardless of the sample value.
            decrement_ta_word(clause, word, active & specificity_mask);
        }
    }
}

void SharedMultiOutputTMReference::type_ii_feedback(
    std::uint32_t output,
    std::uint32_t clause,
    double update_probability,
    const std::vector<std::uint32_t>& literal_words,
    std::uint64_t epoch,
    std::uint64_t sample) {
    if (!bernoulli(update_probability * type_ii_probability_scale_,
                   RandomDomain::TypeIIClause,
                   epoch,
                   sample,
                   output,
                   clause)) {
        return;
    }
    if (!clause_output(clause, literal_words, false)) {
        return;
    }
    const auto included_literals = included_literal_count(clause);
    auto remaining_literal_slots = included_literals < config_.max_included_literals
        ? config_.max_included_literals - included_literals
        : 0U;
    for (std::uint32_t word = 0; word < literal_word_count_; ++word) {
        auto increment = feedback_literal_mask(word) & ~literal_words[word];
        increment = cap_literal_action_transitions(
            clause, word, increment, remaining_literal_slots);
        increment_ta_word(clause, word, increment);
    }
}

void SharedMultiOutputTMReference::update_output(
    std::uint32_t output,
    bool target,
    double update_probability,
    const std::vector<std::uint8_t>& original_clause_outputs,
    const std::vector<std::uint32_t>& literal_words,
    std::uint64_t epoch,
    std::uint64_t sample) {
    // Type I runs over all matching-polarity clauses before Type II, just as
    // the two ClauseBank calls in TMU do.
    for (std::uint32_t clause = 0; clause < config_.clause_count; ++clause) {
        const auto non_negative = weights_[weight_index(output, clause)] >= 0;
        if (non_negative == target) {
            type_i_feedback(
                output, clause, update_probability, literal_words, epoch, sample);
        }
    }
    for (std::uint32_t clause = 0; clause < config_.clause_count; ++clause) {
        const auto non_negative = weights_[weight_index(output, clause)] >= 0;
        if (non_negative != target) {
            type_ii_feedback(
                output, clause, update_probability, literal_words, epoch, sample);
        }
    }

    bool update_positive_weights = true;
    if (target) {
        std::uint32_t positive_clause_count = 0U;
        for (std::uint32_t clause = 0; clause < config_.clause_count; ++clause) {
            if (weights_[weight_index(output, clause)] >= 0) {
                ++positive_clause_count;
            }
        }
        update_positive_weights = positive_clause_count < config_.max_positive_clauses;
    }
    if (target && !update_positive_weights) {
        return;
    }

    // Weight feedback uses the clause outputs captured before all TA feedback.
    for (std::uint32_t clause = 0; clause < config_.clause_count; ++clause) {
        if (original_clause_outputs[clause] == 0U ||
            !bernoulli(update_probability,
                       RandomDomain::WeightUpdate,
                       epoch,
                       sample,
                       output,
                       clause)) {
            continue;
        }
        auto& clause_weight = weights_[weight_index(output, clause)];
        if (target) {
            if (clause_weight < std::numeric_limits<std::int32_t>::max()) {
                ++clause_weight;
            }
        } else if (clause_weight > std::numeric_limits<std::int32_t>::min()) {
            --clause_weight;
        }
    }
}

}  // namespace tmgm::native
