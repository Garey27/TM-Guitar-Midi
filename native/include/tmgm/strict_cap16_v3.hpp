#pragma once

#include "tmgm/ensemble_bundle.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace tmgm::native {

inline constexpr std::size_t kStrictCap16V3LogicalBankCount = 5U;
inline constexpr std::size_t kStrictCap16V3PackedRowCount = 4U;
inline constexpr std::size_t kStrictCap16V3ActivityMemberCount = 7U;
inline constexpr std::size_t kStrictCap16V3OnsetMemberCount = 10U;
inline constexpr std::uint32_t kStrictCap16V3OutputCount = 49U;

enum class StrictCap16V3PackedRowId : std::uint8_t {
    plain = 0U,
    hcontrast = 1U,
    hprofile = 2U,
    cattack = 3U,
};

// Immutable, auditable description of one logical bundle. hcontrast-d2 and
// hcontrast-d3 deliberately map to the same semantic packed row, but remain
// separate authenticated bundles because they contain different selected
// heads. The embedded fingerprint is the exact fingerprint in TMGMBND; the
// semantic fingerprint is the frontend/binarizer contract required from the
// caller. They differ for legacy bundles by design.
struct StrictCap16V3BankManifest {
    const char* logical_id = nullptr;
    const char* bundle_filename = nullptr;
    StrictCap16V3PackedRowId packed_row = StrictCap16V3PackedRowId::plain;
    std::uint32_t feature_count = 0U;
    std::uint32_t bundle_format_version = 0U;
    const char* embedded_feature_fingerprint_sha256 = nullptr;
    const char* semantic_feature_fingerprint_sha256 = nullptr;
    const char* bundle_checksum_sha256 = nullptr;
};

// A route is part of the frozen global ensemble, rather than a request to
// average a bundle head. global_identifier authenticates the cross-bank order;
// bundle_identifier locates the actual member (some v3 packaging IDs have a
// head prefix). Threshold and scale are repeated here intentionally so prepare
// can reject a locally valid bundle that does not implement the frozen global
// calibration contract.
struct StrictCap16V3MemberManifest {
    const char* global_identifier = nullptr;
    const char* bundle_identifier = nullptr;
    std::uint8_t logical_bank_index = 0U;
    EnsembleMemberHead head = EnsembleMemberHead::activity;
    std::int32_t score_threshold = 0;
    float robust_scale = 1.0F;
};

struct StrictCap16V3HeadManifest {
    std::uint32_t quantization = 1024U;
    std::int32_t ensemble_threshold = 0;
    const char* member_order_sha256 = nullptr;
    const char* source_artifact_sha256 = nullptr;
};

struct StrictCap16V3Manifest {
    const char* schema = nullptr;
    const char* version = nullptr;
    std::uint32_t sample_rate = 0U;
    std::uint32_t hop_size = 0U;
    std::int32_t midi_min = 0;
    std::int32_t midi_max = 0;
    StrictCap16V3HeadManifest activity;
    StrictCap16V3HeadManifest onset;
    std::array<StrictCap16V3BankManifest,
               kStrictCap16V3LogicalBankCount> banks{};
    std::array<StrictCap16V3MemberManifest,
               kStrictCap16V3ActivityMemberCount> activity_members{};
    std::array<StrictCap16V3MemberManifest,
               kStrictCap16V3OnsetMemberCount> onset_members{};
};

// The manifest is compiled from the frozen strict-cap16-v3 selection. It is
// the native source of truth for routing and calibration; dummy opposite heads
// present only for the per-bank bundle format are absent from these routes.
[[nodiscard]] const StrictCap16V3Manifest&
strict_cap16_v3_manifest() noexcept;

struct StrictCap16V3PackedRow {
    const std::uint32_t* words = nullptr;
    std::size_t word_count = 0U;
};

struct StrictCap16V3FrameInput {
    std::array<StrictCap16V3PackedRow,
               kStrictCap16V3PackedRowCount> rows{};
};

// normalized_* are the unquantized float32 global means. quantized_* are the
// exact TMGMSCORES values obtained with float64 scaling by 1024 followed by
// round-to-nearest/ties-to-even. Decisions compare those integer values with
// the frozen thresholds (-169 activity, -492 onset).
struct StrictCap16V3FrameOutputBuffers {
    float* normalized_activity_scores = nullptr;
    std::size_t normalized_activity_score_count = 0U;
    std::int32_t* quantized_activity_scores = nullptr;
    std::size_t quantized_activity_score_count = 0U;
    std::uint8_t* activity_predictions = nullptr;
    std::size_t activity_prediction_count = 0U;
    float* normalized_onset_scores = nullptr;
    std::size_t normalized_onset_score_count = 0U;
    std::int32_t* quantized_onset_scores = nullptr;
    std::size_t quantized_onset_score_count = 0U;
    std::uint8_t* onset_predictions = nullptr;
    std::size_t onset_prediction_count = 0U;
};

enum class StrictCap16V3PredictStatus : std::uint8_t {
    success = 0U,
    unprepared,
    null_packed_row,
    wrong_packed_row_word_count,
    null_output_buffer,
    output_buffer_too_small,
    bank_predictor_failure,
};

// Production-safe coordinator for the frozen cross-bank ensemble. load and
// prepare perform all file access, checksum/contract validation, route lookup,
// and allocation. predict_frame is noexcept and performs no allocation, I/O,
// locking, or model lookup by string. A prepared instance belongs to one
// realtime stream; prepare a separate instance per concurrent stream.
class StrictCap16V3Coordinator {
public:
    ~StrictCap16V3Coordinator();

    StrictCap16V3Coordinator(const StrictCap16V3Coordinator&) = delete;
    StrictCap16V3Coordinator& operator=(
        const StrictCap16V3Coordinator&) = delete;
    StrictCap16V3Coordinator(StrictCap16V3Coordinator&&) noexcept;
    StrictCap16V3Coordinator& operator=(
        StrictCap16V3Coordinator&&) noexcept;

    // package_root is the strict-cap16-v3 package directory containing the
    // bundles/ subdirectory.
    [[nodiscard]] static StrictCap16V3Coordinator load(
        const std::filesystem::path& package_root);

    // Exact logical order is manifest.banks. This form lets a host resolve its
    // own package layout while retaining native checksum verification. Loading
    // and preparation must always happen outside the audio callback.
    [[nodiscard]] static StrictCap16V3Coordinator prepare(
        const std::array<std::filesystem::path,
                         kStrictCap16V3LogicalBankCount>& bundle_paths);

    [[nodiscard]] std::size_t required_packed_word_count(
        StrictCap16V3PackedRowId row) const noexcept;
    [[nodiscard]] std::uint32_t output_count() const noexcept;

    [[nodiscard]] StrictCap16V3PredictStatus predict_frame(
        const StrictCap16V3FrameInput& input,
        const StrictCap16V3FrameOutputBuffers& outputs) noexcept;

private:
    struct Impl;
    explicit StrictCap16V3Coordinator(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tmgm::native
