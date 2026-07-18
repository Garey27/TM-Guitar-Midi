#pragma once

#include "tmgm/dataset.hpp"
#include "tmgm/model.hpp"

#include <cstdint>
#include <vector>

namespace tmgm::native {

struct TmScoreCalibration {
    std::int32_t threshold = 0;
    std::uint64_t true_positives = 0;
    std::uint64_t false_positives = 0;
    std::uint64_t false_negatives = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double predicted_mean_polyphony = 0.0;
    double target_mean_polyphony = 0.0;
};

// Mirrors the calibration constraint used during native CUDA training. The
// looser onset limit allows short onset targets to absorb small timing errors.
[[nodiscard]] double default_maximum_polyphony_ratio(TmModelHead head);

// Selects one global inclusive score threshold (score >= threshold) on the
// supplied labelled dataset. The model head decides whether activity_words or
// onset_words is the calibration truth. Equal-F1 candidates prefer the higher,
// more conservative threshold.
[[nodiscard]] TmScoreCalibration calibrate_score_threshold(
    const NativeDataset& dataset,
    TmModelHead head,
    const std::vector<std::int32_t>& scores,
    double maximum_polyphony_ratio);

void apply_score_threshold(
    const std::vector<std::int32_t>& scores,
    std::int32_t threshold,
    std::vector<std::uint8_t>& predictions);

}  // namespace tmgm::native
