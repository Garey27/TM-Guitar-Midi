#pragma once

#include "tmgm/dataset.hpp"

#include <cstdint>

namespace tmgm::native {

// Host-side description of a validation checkpoint candidate.  Keeping the
// comparison outside the CUDA translation unit makes checkpoint policy easy to
// test and keeps it deterministic across GPU backends.
struct TmValidationCandidate {
    std::uint32_t epoch = 0U;
    double f1 = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    std::int32_t score_threshold = 0;
};

// Host-only state machine used by CUDA training and unit tests. Patience zero
// is deliberately a no-op so existing fixed-epoch experiments are unchanged.
// Exact F1 ties do not reset patience, even if another checkpoint tie-breaker
// (precision/recall/threshold) selects that epoch for persistence.
struct TmValidationEarlyStopping {
    std::uint32_t patience = 0U;
    std::uint32_t epochs_without_f1_improvement = 0U;
    double best_f1 = 0.0;
    bool has_best_f1 = false;
    bool early_stopped = false;
};

// Train and held-out datasets may contain different frame counts and sampling
// orders, but every field that defines the model/input schema must match.
void validate_tm_training_validation_compatibility(
    const NativeDataset& training,
    const NativeDataset& validation);

// A non-zero patience has no meaning without metrics from a held-out dataset.
// Kept host-only so the CLI contract can be regression-tested without CUDA.
void validate_tm_validation_patience(
    std::uint32_t patience,
    bool has_validation_dataset);

// Higher held-out F1 wins.  Exact ties prefer precision, then recall, then the
// more conservative (higher) score threshold, and finally the earlier epoch.
// The final rule prevents replacing a stable checkpoint with an identical one.
[[nodiscard]] bool is_better_tm_validation_candidate(
    const TmValidationCandidate& candidate,
    const TmValidationCandidate& current_best);

// Observe one completed validation epoch. Returns true from the first epoch at
// which the configured patience is exhausted, and stays true afterwards.
[[nodiscard]] bool update_tm_validation_early_stopping(
    TmValidationEarlyStopping& state,
    double validation_f1);

}  // namespace tmgm::native
