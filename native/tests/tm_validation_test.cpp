#include "tmgm/tm_validation.hpp"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using tmgm::native::NativeDataset;
using tmgm::native::TmValidationCandidate;
using tmgm::native::TmValidationEarlyStopping;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_invalid(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("expected std::invalid_argument");
}

NativeDataset make_dataset(const std::uint64_t frames) {
    NativeDataset result;
    result.header.frame_count = frames;
    result.header.feature_count = 65U;
    result.header.feature_words_per_row = 2U;
    result.header.note_count = 49U;
    result.header.label_words_per_row = 1U;
    result.header.midi_min = 40;
    result.header.midi_max = 88;
    result.header.sample_rate = 22050U;
    result.header.hop_size = 256U;
    result.header.onset_index_count = frames;
    result.feature_words.resize(static_cast<std::size_t>(frames) * 2U);
    result.activity_words.resize(static_cast<std::size_t>(frames));
    result.onset_words.resize(static_cast<std::size_t>(frames));
    result.onset_indices.resize(static_cast<std::size_t>(frames));
    for (std::uint32_t row = 0; row < frames; ++row) {
        result.onset_indices[row] = row;
    }
    return result;
}

void test_schema_accepts_different_rows_and_seed() {
    auto training = make_dataset(3U);
    auto validation = make_dataset(7U);
    training.header.seed = 1U;
    validation.header.seed = 2U;
    tmgm::native::validate_tm_training_validation_compatibility(
        training, validation);
}

void test_schema_rejects_model_input_mismatch() {
    const auto training = make_dataset(3U);
    auto validation = make_dataset(7U);
    validation.header.hop_size = 128U;
    require_invalid([&] {
        tmgm::native::validate_tm_training_validation_compatibility(
            training, validation);
    });

    validation = make_dataset(7U);
    validation.header.feature_count = 64U;
    // Keep its packed storage internally valid so the schema check, rather
    // than NativeDataset::validate(), is what rejects it.
    validation.header.feature_words_per_row = 1U;
    validation.feature_words.resize(7U);
    require_invalid([&] {
        tmgm::native::validate_tm_training_validation_compatibility(
            training, validation);
    });
}

void test_best_checkpoint_policy() {
    const TmValidationCandidate baseline{3U, 0.80, 0.75, 0.86, 4};
    require(
        tmgm::native::is_better_tm_validation_candidate(
            {4U, 0.81, 0.60, 0.99, 1}, baseline),
        "higher validation F1 must win");
    require(
        tmgm::native::is_better_tm_validation_candidate(
            {4U, 0.80, 0.76, 0.84, 3}, baseline),
        "precision must break an exact F1 tie");
    require(
        !tmgm::native::is_better_tm_validation_candidate(
            {4U, 0.80, 0.75, 0.86, 4}, baseline),
        "an identical later epoch must not replace the checkpoint");
    require(
        tmgm::native::is_better_tm_validation_candidate(
            {2U, 0.80, 0.75, 0.86, 4}, baseline),
        "an identical earlier epoch must win deterministically");
}

void test_early_stopping_disabled() {
    TmValidationEarlyStopping state;
    require(
        !tmgm::native::update_tm_validation_early_stopping(state, 0.8),
        "disabled early stopping must accept the first epoch");
    for (const auto f1 : {0.8, 0.7, 0.6, 0.5}) {
        require(
            !tmgm::native::update_tm_validation_early_stopping(state, f1),
            "patience zero must never stop");
    }
    require(!state.early_stopped, "disabled state must not become stopped");
}

void test_patience_requires_validation() {
    tmgm::native::validate_tm_validation_patience(0U, false);
    tmgm::native::validate_tm_validation_patience(4U, true);
    require_invalid([] {
        tmgm::native::validate_tm_validation_patience(4U, false);
    });
}

void test_early_stopping_patience_and_reset() {
    TmValidationEarlyStopping state;
    state.patience = 2U;
    require(
        !tmgm::native::update_tm_validation_early_stopping(state, 0.50),
        "first validation F1 establishes the baseline");
    require(
        !tmgm::native::update_tm_validation_early_stopping(state, 0.49),
        "one stale epoch must not exhaust patience two");
    require(
        !tmgm::native::update_tm_validation_early_stopping(state, 0.51),
        "strict improvement must reset patience");
    require(
        state.epochs_without_f1_improvement == 0U,
        "improvement did not reset the stale epoch counter");
    require(
        !tmgm::native::update_tm_validation_early_stopping(state, 0.51),
        "an exact tie is the first stale epoch");
    require(
        tmgm::native::update_tm_validation_early_stopping(state, 0.50),
        "two consecutive stale epochs must stop");
    require(
        state.early_stopped &&
            state.epochs_without_f1_improvement == 2U,
        "stopped state has inconsistent counters");
    require(
        tmgm::native::update_tm_validation_early_stopping(state, 0.99),
        "a stopped state must remain stopped");
}

void test_early_stopping_rejects_invalid_f1() {
    TmValidationEarlyStopping state;
    state.patience = 1U;
    require_invalid([&] {
        static_cast<void>(
            tmgm::native::update_tm_validation_early_stopping(state, -0.01));
    });
    require_invalid([&] {
        static_cast<void>(
            tmgm::native::update_tm_validation_early_stopping(state, 1.01));
    });
}

}  // namespace

int main() {
    try {
        test_schema_accepts_different_rows_and_seed();
        test_schema_rejects_model_input_mismatch();
        test_best_checkpoint_policy();
        test_early_stopping_disabled();
        test_patience_requires_validation();
        test_early_stopping_patience_and_reset();
        test_early_stopping_rejects_invalid_f1();
        std::cout << "TM validation support tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "TM validation support test failed: " << exception.what() << '\n';
        return 1;
    }
}
