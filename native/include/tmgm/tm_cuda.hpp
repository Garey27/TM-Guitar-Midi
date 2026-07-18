#pragma once

#include "tmgm/dataset.hpp"
#include "tmgm/model.hpp"

#include <cstdint>
#include <vector>

namespace tmgm::native {

// A cooperative training launch handles this many sequential samples. Large
// launches can exceed Windows' WDDM watchdog even when they are valid CUDA, so
// keep a conservative hard ceiling and split epochs into additional launches.
inline constexpr std::uint32_t kTmCudaMaxSamplesPerLaunch = 512U;

enum class TargetHead {
    activity,
    onset,
};

struct TmCudaConfig {
    std::uint32_t clauses = 256;
    std::int32_t threshold = 128;
    float specificity = 5.0f;
    float negative_samples = 8.0f;
    // Effective only for the onset head. When enabled, activity=1/onset=0 for
    // the same pitch is always selected as a negative training output; the
    // usual TM update probability still controls the actual feedback.
    bool onset_sustain_hard_negatives = false;
    // Selection probability for each activity=1/onset=0 output. A value near
    // 0.1 is useful on natural corpora where sustain frames greatly outnumber
    // attacks; 1.0 reproduces the always-select ablation.
    float onset_sustain_hard_negative_probability = 1.0f;
    // Keep shared clause automata intact and apply sustain hard negatives only
    // to output-specific clause weights. This avoids destructive cross-pitch
    // TA feedback in the coalesced/shared bank.
    bool onset_sustain_hard_negative_weight_only = false;
    std::uint32_t max_included_literals = 64;
    std::uint32_t state_bits = 8;
    std::uint32_t epochs = 30;
    std::uint32_t samples_per_launch = 128;
    // Experimental ablation only. The historical/default positive-output
    // feedback order is ascending MIDI index. When enabled, each training
    // sample advances the cyclic start by one output while preserving all
    // existing counter-based random keys and the inference/model format.
    bool rotate_output_update_order = false;
    // Zero disables early stopping. A positive value requires a held-out
    // validation dataset and stops after this many consecutive epochs without
    // a strictly higher validation F1.
    std::uint32_t validation_patience = 0;
    std::uint64_t seed = 20260718;
    bool verbose = true;
};

struct TmEpochReport {
    std::uint32_t epoch = 0;
    // End-to-end epoch time, including prediction and threshold calibration.
    double seconds = 0.0;
    // CUDA training kernels only; useful for backend profiling.
    double train_seconds = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double predicted_mean_polyphony = 0.0;
    double target_mean_polyphony = 0.0;
    std::int32_t score_threshold = 0;

    bool has_validation = false;
    double validation_precision = 0.0;
    double validation_recall = 0.0;
    double validation_f1 = 0.0;
    double validation_predicted_mean_polyphony = 0.0;
    double validation_target_mean_polyphony = 0.0;
    std::int32_t validation_score_threshold = 0;
};

struct TmCudaResult {
    std::uint64_t frame_count = 0;
    std::uint32_t output_count = 0;
    std::vector<std::int32_t> scores;
    std::vector<std::uint8_t> predictions;
    std::vector<TmEpochReport> epochs;
    NativeTmModel model;
    // Zero when no held-out validation dataset was supplied.
    std::uint32_t best_validation_epoch = 0U;
    // Number of epochs that actually completed. This is smaller than the
    // configured epoch count only when validation early stopping fired.
    std::uint32_t epochs_executed = 0U;
    bool early_stopped = false;
};

struct TmCudaPrediction {
    std::uint64_t frame_count = 0;
    std::uint32_t output_count = 0;
    std::vector<std::int32_t> scores;
    std::vector<std::uint8_t> predictions;
};

[[nodiscard]] bool cuda_tm_supported();

[[nodiscard]] TmCudaResult train_tm_cuda(
    const NativeDataset& dataset,
    TargetHead head,
    const TmCudaConfig& config,
    const NativeDataset* validation_dataset = nullptr,
    bool allow_legacy_feature_contract = false);

[[nodiscard]] TmCudaPrediction predict_tm_cuda(
    const NativeDataset& dataset,
    const NativeTmModel& model,
    bool allow_legacy_feature_contract = false);

}  // namespace tmgm::native
