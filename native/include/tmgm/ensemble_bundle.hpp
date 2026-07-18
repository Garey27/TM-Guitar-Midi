#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace tmgm::native {

inline constexpr std::uint32_t kEnsembleBundleLegacyFormatVersion = 1U;
inline constexpr std::uint32_t kEnsembleBundleFormatVersion = 2U;
inline constexpr std::uint32_t kEnsembleBundleHeaderBytes = 256U;
inline constexpr std::uint32_t kEnsembleMemberDescriptorBytes = 192U;

enum class EnsembleMemberHead : std::uint32_t {
    activity = 1U,
    onset = 2U,
};

enum class EnsembleFusion : std::uint32_t {
    mean = 1U,
};

// Inference-only shared-clause TM. Clause literals use TMU's canonical
// positive-first layout: [0, feature_count) requires a true feature and
// [feature_count, 2 * feature_count) requires a false feature. Weights are
// clause-major [clause][output]. Empty clauses are retained but suppressed at
// inference, matching TMU/native CUDA prediction.
struct SparseTmEnsembleMember {
    std::string identifier;
    EnsembleMemberHead head = EnsembleMemberHead::activity;
    std::int32_t score_threshold = 0;
    float robust_scale = 1.0F;
    std::uint32_t feature_count = 0U;
    std::uint32_t output_count = 0U;
    std::uint32_t clause_count = 0U;
    std::uint32_t literal_count = 0U;
    std::array<std::uint8_t, 32> source_model_sha256{};
    std::vector<std::uint32_t> clause_offsets;
    std::vector<std::uint16_t> literal_ids;
    std::vector<std::int16_t> weights;
};

struct EnsembleHeadConfig {
    EnsembleFusion fusion = EnsembleFusion::mean;
    std::uint32_t quantization = 0U;
    std::int32_t ensemble_threshold = 0;
    std::array<std::uint8_t, 32> member_order_sha256{};
};

struct EnsembleBundle {
    std::uint32_t format_version = kEnsembleBundleFormatVersion;
    std::uint32_t feature_count = 0U;
    std::uint32_t output_count = 0U;
    std::int32_t midi_min = 0;
    std::int32_t midi_max = 0;
    std::uint32_t sample_rate = 0U;
    std::uint32_t hop_size = 0U;
    EnsembleHeadConfig activity;
    EnsembleHeadConfig onset;
    std::array<std::uint8_t, 32> feature_fingerprint_sha256{};
    std::array<std::uint8_t, 32> bundle_checksum_sha256{};
    std::vector<SparseTmEnsembleMember> members;
};

// Throws std::invalid_argument when an in-memory bundle is inconsistent.
void validate_ensemble_bundle(const EnsembleBundle& bundle);

// SHA-256 is over the canonical complete file with the 32-byte checksum field
// zeroed. Source-model and feature hashes are part of that authenticated data.
[[nodiscard]] std::string ensemble_sha256_hex(
    const std::array<std::uint8_t, 32>& digest);

[[nodiscard]] EnsembleBundle load_ensemble_bundle(
    const std::filesystem::path& path,
    bool verify_checksum = true);

}  // namespace tmgm::native
