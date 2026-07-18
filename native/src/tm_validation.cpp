#include "tmgm/tm_validation.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace tmgm::native {
namespace {

template <typename T>
void require_equal(const T training, const T validation, const char* field) {
    if (training != validation) {
        throw std::invalid_argument(
            std::string("validation dataset ") + field +
            " does not match the training dataset");
    }
}

void validate_candidate(const TmValidationCandidate& candidate) {
    if (candidate.epoch == 0U || !std::isfinite(candidate.f1) ||
        !std::isfinite(candidate.precision) || !std::isfinite(candidate.recall) ||
        candidate.f1 < 0.0 || candidate.f1 > 1.0 ||
        candidate.precision < 0.0 || candidate.precision > 1.0 ||
        candidate.recall < 0.0 || candidate.recall > 1.0) {
        throw std::invalid_argument("invalid TM validation checkpoint candidate");
    }
}

}  // namespace

void validate_tm_training_validation_compatibility(
    const NativeDataset& training,
    const NativeDataset& validation) {
    training.validate();
    validation.validate();
    require_equal(
        training.header.feature_count,
        validation.header.feature_count,
        "feature_count");
    require_equal(
        training.header.feature_words_per_row,
        validation.header.feature_words_per_row,
        "feature_words_per_row");
    require_equal(
        training.header.note_count,
        validation.header.note_count,
        "note_count");
    require_equal(
        training.header.label_words_per_row,
        validation.header.label_words_per_row,
        "label_words_per_row");
    require_equal(training.header.midi_min, validation.header.midi_min, "midi_min");
    require_equal(training.header.midi_max, validation.header.midi_max, "midi_max");
    require_equal(
        training.header.sample_rate,
        validation.header.sample_rate,
        "sample_rate");
    require_equal(training.header.hop_size, validation.header.hop_size, "hop_size");
}

void validate_tm_validation_patience(
    const std::uint32_t patience,
    const bool has_validation_dataset) {
    if (patience != 0U && !has_validation_dataset) {
        throw std::invalid_argument(
            "validation_patience requires a held-out validation dataset");
    }
}

bool is_better_tm_validation_candidate(
    const TmValidationCandidate& candidate,
    const TmValidationCandidate& current_best) {
    validate_candidate(candidate);
    validate_candidate(current_best);
    if (candidate.f1 != current_best.f1) {
        return candidate.f1 > current_best.f1;
    }
    if (candidate.precision != current_best.precision) {
        return candidate.precision > current_best.precision;
    }
    if (candidate.recall != current_best.recall) {
        return candidate.recall > current_best.recall;
    }
    if (candidate.score_threshold != current_best.score_threshold) {
        return candidate.score_threshold > current_best.score_threshold;
    }
    return candidate.epoch < current_best.epoch;
}

bool update_tm_validation_early_stopping(
    TmValidationEarlyStopping& state,
    const double validation_f1) {
    if (!std::isfinite(validation_f1) || validation_f1 < 0.0 ||
        validation_f1 > 1.0) {
        throw std::invalid_argument("invalid validation F1 for early stopping");
    }
    if (state.early_stopped) {
        return true;
    }
    if (!state.has_best_f1 || validation_f1 > state.best_f1) {
        state.best_f1 = validation_f1;
        state.has_best_f1 = true;
        state.epochs_without_f1_improvement = 0U;
        return false;
    }
    if (state.patience == 0U) {
        return false;
    }
    ++state.epochs_without_f1_improvement;
    if (state.epochs_without_f1_improvement >= state.patience) {
        state.early_stopped = true;
    }
    return state.early_stopped;
}

}  // namespace tmgm::native
