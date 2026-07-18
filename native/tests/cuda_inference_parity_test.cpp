#include "tmgm/dataset.hpp"
#include "tmgm/model.hpp"
#include "tmgm/tm_cuda.hpp"
#include "tmgm/tm_reference.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kSkipped = 77;

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Value>
void require_byte_equal(
    const std::vector<Value>& cpu,
    const std::vector<Value>& cuda,
    const char* payload_name) {
    if (cpu.size() != cuda.size()) {
        std::ostringstream message;
        message << payload_name << " size differs: CPU=" << cpu.size()
                << ", CUDA=" << cuda.size();
        throw std::runtime_error(message.str());
    }
    if (!cpu.empty() &&
        std::memcmp(cpu.data(), cuda.data(), cpu.size() * sizeof(Value)) != 0) {
        for (std::size_t index = 0; index < cpu.size(); ++index) {
            if (cpu[index] != cuda[index]) {
                std::ostringstream message;
                message << payload_name << " differs at flat index " << index
                        << ": CPU=" << static_cast<long long>(cpu[index])
                        << ", CUDA=" << static_cast<long long>(cuda[index]);
                throw std::runtime_error(message.str());
            }
        }
        throw std::runtime_error(std::string(payload_name) + " differs byte-for-byte");
    }
}

tmgm::native::NativeDataset make_dataset() {
    constexpr std::uint64_t frames = 19U;
    constexpr std::uint32_t features = 37U;
    constexpr std::uint32_t outputs = 4U;

    tmgm::native::NativeDataset dataset;
    dataset.header.frame_count = frames;
    dataset.header.feature_count = features;
    dataset.header.feature_words_per_row = 1U;
    dataset.header.note_count = outputs;
    dataset.header.label_words_per_row = 1U;
    dataset.header.midi_min = 40;
    dataset.header.midi_max = 43;
    dataset.header.sample_rate = 22'050U;
    dataset.header.hop_size = 256U;
    dataset.header.onset_index_count = 0U;
    dataset.header.seed = 20'260'718U;
    dataset.header.feature_fingerprint_sha256.fill(0x42U);
    dataset.feature_words.assign(frames, 0U);
    dataset.activity_words.assign(frames, 0U);
    dataset.onset_words.assign(frames, 0U);

    // Rows zero and one are all-false/all-true. The remaining deterministic
    // pattern exercises both the 32-bit literal boundary and feature 36, the
    // final valid bit before packed-row padding.
    for (std::uint32_t feature = 0; feature < features; ++feature) {
        dataset.set_feature(1U, feature, true);
    }
    for (std::uint64_t frame = 2U; frame < frames; ++frame) {
        for (std::uint32_t feature = 0; feature < features; ++feature) {
            const auto value =
                (frame * 13U + feature * 7U + feature * feature) % 11U;
            dataset.set_feature(frame, feature, value < 5U);
        }
    }
    dataset.validate();
    return dataset;
}

struct Fixture {
    tmgm::native::SharedMultiOutputTMReference reference;
    tmgm::native::NativeTmModel model;
};

Fixture make_model() {
    constexpr std::uint32_t features = 37U;
    constexpr std::uint32_t outputs = 4U;
    constexpr std::uint32_t clauses = 11U;

    tmgm::native::TMReferenceConfig config;
    config.feature_count = features;
    config.output_count = outputs;
    config.clause_count = clauses;
    config.threshold = 32;
    config.specificity = 5.0;
    config.negative_sampling_q = 3.0;
    config.max_included_literals = 12U;
    config.seed = 20'260'718U;
    tmgm::native::SharedMultiOutputTMReference reference(config);

    // Literal layout is [x0..x36, !x0..!x36]. Clause 3 is deliberately empty
    // and clause 4 contradictory. Both backends must suppress them at inference.
    const std::vector<std::vector<std::uint32_t>> included_literals = {
        {0U},
        {features + 1U},
        {35U, features + 36U},
        {},
        {4U, features + 4U},
        {2U, 3U},
        {features + 2U, 3U},
        {36U},
        {features + 0U, features + 35U},
        {17U, features + 18U},
        {features + 36U},
    };
    for (std::uint32_t clause = 0; clause < clauses; ++clause) {
        for (const auto literal : included_literals[clause]) {
            reference.set_ta_state(clause, literal, 128U);
        }
    }

    const std::vector<std::vector<std::int32_t>> weights = {
        {7, -2, 3, 101, -5, 4, 11, 1, -9, 6, 2},
        {-3, 8, -4, -101, 2, 12, -7, 5, 10, -6, 1},
        {std::numeric_limits<std::int32_t>::max(),
         std::numeric_limits<std::int32_t>::max(),
         17, 42, -3, 9, 5, 0, -2, 4, 1},
        {std::numeric_limits<std::int32_t>::min(),
         std::numeric_limits<std::int32_t>::min(),
         -17, -42, 3, -9, -5, 0, 2, -4, -1},
    };
    for (std::uint32_t output = 0; output < outputs; ++output) {
        for (std::uint32_t clause = 0; clause < clauses; ++clause) {
            reference.set_weight(output, clause, weights[output][clause]);
        }
    }

    tmgm::native::NativeTmModel model;
    model.head = tmgm::native::TmModelHead::activity;
    model.dimensions.feature_count = features;
    model.dimensions.output_count = outputs;
    model.dimensions.clause_count = clauses;
    model.dimensions.state_bits =
        tmgm::native::SharedMultiOutputTMReference::kStateBits;
    model.training.threshold = config.threshold;
    model.training.specificity = static_cast<float>(config.specificity);
    model.training.negative_samples =
        static_cast<float>(config.negative_sampling_q);
    model.training.type_i_ii_ratio = static_cast<float>(config.type_i_ii_ratio);
    model.training.max_included_literals = config.max_included_literals;
    model.training.seed = config.seed;
    model.training.feature_negation = config.feature_negation;
    model.training.boost_true_positive_feedback =
        config.boost_true_positive_feedback;
    model.midi.minimum_note = 40;
    model.midi.maximum_note = 43;
    model.midi.channel = 1U;
    model.midi.audio_sample_rate = 22'050U;
    model.midi.analysis_hop_samples = 256U;
    model.score_threshold = 5;
    model.feature_fingerprint_sha256.fill(0x42U);
    model.ta_bitplanes = reference.ta_bitplanes();
    model.weights = reference.weights();
    tmgm::native::validate_tm_model(model);
    return {std::move(reference), std::move(model)};
}

void test_cuda_matches_cpu_reference() {
    const auto dataset = make_dataset();
    auto fixture = make_model();

    require(fixture.reference.ta_bitplanes() == fixture.model.ta_bitplanes,
            "fixture TA bitplanes differ before inference");
    require(fixture.reference.weights() == fixture.model.weights,
            "fixture weights differ before inference");

    std::vector<std::int32_t> cpu_scores;
    std::vector<std::uint8_t> cpu_predictions;
    cpu_scores.reserve(
        static_cast<std::size_t>(dataset.header.frame_count) *
        dataset.header.note_count);
    cpu_predictions.reserve(cpu_scores.capacity());
    for (std::uint64_t frame = 0; frame < dataset.header.frame_count; ++frame) {
        std::vector<std::uint8_t> feature_row(dataset.header.feature_count, 0U);
        for (std::uint32_t feature = 0;
             feature < dataset.header.feature_count;
             ++feature) {
            feature_row[feature] = dataset.feature(frame, feature) ? 1U : 0U;
        }
        const auto literals = fixture.reference.encode_literals(feature_row);
        const auto row_scores = fixture.reference.scores(literals, false);
        for (const auto score : row_scores) {
            cpu_scores.push_back(score);
            cpu_predictions.push_back(
                score >= fixture.model.score_threshold ? 1U : 0U);
        }
    }

    const auto cuda = tmgm::native::predict_tm_cuda(dataset, fixture.model);
    require(cuda.frame_count == dataset.header.frame_count,
            "CUDA prediction returned the wrong frame count");
    require(cuda.output_count == dataset.header.note_count,
            "CUDA prediction returned the wrong output count");
    require_byte_equal(cpu_scores, cuda.scores, "scores");
    require_byte_equal(cpu_predictions, cuda.predictions, "predictions");
}

std::uint32_t popcount32(std::uint32_t value) {
    std::uint32_t count = 0U;
    while (value != 0U) {
        value &= value - 1U;
        ++count;
    }
    return count;
}

void test_cuda_training_respects_strict_literal_cap() {
    constexpr std::uint64_t frames = 32U;
    constexpr std::uint32_t features = 1'100U;
    constexpr std::uint32_t clauses = 16U;

    tmgm::native::NativeDataset dataset;
    dataset.header.frame_count = frames;
    dataset.header.feature_count = features;
    dataset.header.feature_words_per_row = (features + 63U) / 64U;
    dataset.header.note_count = 1U;
    dataset.header.label_words_per_row = 1U;
    dataset.header.midi_min = 40;
    dataset.header.midi_max = 40;
    dataset.header.sample_rate = 22'050U;
    dataset.header.hop_size = 256U;
    dataset.header.onset_index_count = 0U;
    dataset.header.seed = 91U;
    dataset.header.feature_fingerprint_sha256.fill(0x91U);
    dataset.feature_words.assign(
        frames * dataset.header.feature_words_per_row, 0U);
    dataset.activity_words.assign(frames, 0U);
    dataset.onset_words.assign(frames, 0U);
    for (std::uint64_t frame = 0U; frame < frames; ++frame) {
        // All-false rows place the Type-I candidates in the negated half,
        // beyond the first 32 literal words. This exercises the warp-prefix
        // capacity accounting across more than one chunk batch.
        dataset.set_activity(frame, 0U, true);
    }
    dataset.validate();

    tmgm::native::TmCudaConfig config;
    config.clauses = clauses;
    config.threshold = 64;
    config.specificity = 2.0F;
    config.negative_samples = 1.0F;
    config.max_included_literals = 3U;
    config.state_bits = 8U;
    config.epochs = 2U;
    config.samples_per_launch = 32U;
    config.seed = 91U;
    config.verbose = false;
    const auto result = tmgm::native::train_tm_cuda(
        dataset, tmgm::native::TargetHead::activity, config);

    const auto literal_count = features * 2U;
    const auto chunks = (literal_count + 31U) / 32U;
    std::uint32_t total_included = 0U;
    for (std::uint32_t clause = 0U; clause < clauses; ++clause) {
        std::uint32_t included = 0U;
        const auto action_base =
            (static_cast<std::size_t>(clause) * config.state_bits +
             config.state_bits - 1U) * chunks;
        for (std::uint32_t chunk = 0U; chunk < chunks; ++chunk) {
            auto valid = 0xffffffffU;
            if (chunk + 1U == chunks && (literal_count & 31U) != 0U) {
                valid = (1U << (literal_count & 31U)) - 1U;
            }
            included += popcount32(
                result.model.ta_bitplanes[action_base + chunk] & valid);
        }
        if (included > config.max_included_literals) {
            std::ostringstream message;
            message << "CUDA Type-I/II feedback exceeded the strict literal cap: "
                    << "clause=" << clause << ", included=" << included
                    << ", cap=" << config.max_included_literals << ", action_words=";
            for (std::uint32_t chunk = 0U; chunk < chunks; ++chunk) {
                message << std::hex
                        << result.model.ta_bitplanes[action_base + chunk] << ',';
            }
            throw std::runtime_error(message.str());
        }
        total_included += included;
    }
    require(total_included != 0U,
            "CUDA literal-cap test was vacuous: no clause received feedback");
}

}  // namespace

int main() {
    if (!tmgm::native::cuda_tm_supported()) {
        std::cout << "CUDA inference parity test skipped: compatible CUDA device unavailable\n";
        return kSkipped;
    }
    try {
        test_cuda_matches_cpu_reference();
        test_cuda_training_respects_strict_literal_cap();
        std::cout << "CUDA/CPU inference scores and predictions are byte-identical\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
