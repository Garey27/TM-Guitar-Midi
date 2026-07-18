#include "tmgm/strict_cap16_v3.hpp"

#include "tmgm/ensemble_inference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tmgm::native {
namespace {

constexpr std::size_t row_index(const StrictCap16V3PackedRowId row) noexcept {
    return static_cast<std::size_t>(row);
}

[[nodiscard]] std::array<std::uint8_t, 32> parse_sha256(
    const char* text,
    const char* label) {
    if (text == nullptr || std::strlen(text) != 64U) {
        throw std::invalid_argument(std::string(label) + " is not SHA-256 hex");
    }
    const auto nibble = [&](const char value) -> std::uint8_t {
        if (value >= '0' && value <= '9') {
            return static_cast<std::uint8_t>(value - '0');
        }
        if (value >= 'a' && value <= 'f') {
            return static_cast<std::uint8_t>(value - 'a' + 10);
        }
        if (value >= 'A' && value <= 'F') {
            return static_cast<std::uint8_t>(value - 'A' + 10);
        }
        throw std::invalid_argument(
            std::string(label) + " contains non-hex characters");
    };
    std::array<std::uint8_t, 32> digest{};
    for (std::size_t index = 0U; index < digest.size(); ++index) {
        digest[index] = static_cast<std::uint8_t>(
            (nibble(text[index * 2U]) << 4U) |
            nibble(text[index * 2U + 1U]));
    }
    return digest;
}

[[nodiscard]] bool same_float_bits(
    const float left,
    const float right) noexcept {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::uint32_t left_bits = 0U;
    std::uint32_t right_bits = 0U;
    std::memcpy(&left_bits, &left, sizeof(left));
    std::memcpy(&right_bits, &right, sizeof(right));
    return left_bits == right_bits;
}

// Mirrors numpy: np.rint(float32_score.astype(float64) * quantization).
[[nodiscard]] std::int32_t quantize_ties_to_even(
    const float value,
    const std::uint32_t quantization) noexcept {
    const auto scaled =
        static_cast<double>(value) * static_cast<double>(quantization);
    constexpr auto minimum = std::numeric_limits<std::int32_t>::min();
    constexpr auto maximum = std::numeric_limits<std::int32_t>::max() - 1;
    if (scaled <= static_cast<double>(minimum)) {
        return minimum;
    }
    if (scaled >= static_cast<double>(maximum)) {
        return maximum;
    }
    const auto lower_double = std::floor(scaled);
    auto lower = static_cast<std::int64_t>(lower_double);
    const auto fraction = scaled - lower_double;
    if (fraction > 0.5 || (fraction == 0.5 && lower % 2 != 0)) {
        ++lower;
    }
    return static_cast<std::int32_t>(lower);
}

const StrictCap16V3Manifest kManifest{
    "tmgm-native-cross-bank-manifest-v1",
    "strict-cap16-v3",
    22050U,
    256U,
    40,
    88,
    {
        1024U,
        -169,
        "28fac219a417b1f9ec51db7a32dace6e25cfd42e13e1d1dda5b1789e60da8638",
        "4cb2bf985199747020c5d14f7fa63119cb2de662abb2d5eb893c65fef431365d",
    },
    {
        1024U,
        -492,
        "0965407fe001346cb890b9104f6d8034f79f159d7a68d09aa560ce40b4022957",
        "e1c32666f153825afa0ddd9cde3e35f57e788c4b6377cc4854a49dabc6c0ed5e",
    },
    {{
        {
            "plain",
            "plain.tmgmbundle",
            StrictCap16V3PackedRowId::plain,
            7973U,
            kEnsembleBundleLegacyFormatVersion,
            "e85bd175c09d0135d9204e5703f245e86eee4a643112a7b043e4d65183f190c3",
            "8cf43d89312dc5a481280a017a0f23f2b3ee4f3a573ebf992fb78edb2a305fc1",
            "5cf1a456e55f41db05d3be4bd992273d979717b4cc670a7df0006857dd30a566",
        },
        {
            "hcontrast-d2",
            "hcontrast-d2.tmgmbundle",
            StrictCap16V3PackedRowId::hcontrast,
            7973U,
            kEnsembleBundleLegacyFormatVersion,
            "fde35e458c2a9a7794fef243b9fb65e11d20fcc56fd9e28a2b3c9472dcb1536f",
            "bcfbbdd59dcbf30ceb8b184b61edfc768de031131a72e90b346d460ac3adb8e1",
            "a5ecf3fac66459f46f434977de294c04a156159b92a5d16462a0c37850f080c2",
        },
        {
            "hcontrast-d3",
            "hcontrast-d3.tmgmbundle",
            StrictCap16V3PackedRowId::hcontrast,
            7973U,
            kEnsembleBundleLegacyFormatVersion,
            "38ca0b120795a8b659a68d2c18434b861c3e93bb977c856593c2be06af75794e",
            "bcfbbdd59dcbf30ceb8b184b61edfc768de031131a72e90b346d460ac3adb8e1",
            "3d6cfb7d654131cfbc4de354fe81d61c25017e422a2dd4ea4072ef7189e2f6c6",
        },
        {
            "hprofile-d3",
            "hprofile-d3.tmgmbundle",
            StrictCap16V3PackedRowId::hprofile,
            16205U,
            kEnsembleBundleLegacyFormatVersion,
            "a101928dcc916326611f6d9d68a5ea95a4d47318657f2bb172298bfe92d8709d",
            "4ce497eb7ed6fec1b8d292ef65ad28b38115045e985c4c2634bc0219a61a0ddd",
            "6104dc1e703f87e807f8060988f604d1b896b48bc12e026b6fbb9667795b44a9",
        },
        {
            "cattack-d3",
            "cattack-d3.tmgmbundle",
            StrictCap16V3PackedRowId::cattack,
            10374U,
            kEnsembleBundleFormatVersion,
            "59d6e0de04f6121f427564d8d39c11bb5c7ba1731f43ef8b02e34eff59bf8b2c",
            "59d6e0de04f6121f427564d8d39c11bb5c7ba1731f43ef8b02e34eff59bf8b2c",
            "3ebda8892014be60cceddc9673de284d4c57236af0ddb947a660d28c917c6b8d",
        },
    }},
    {{
        {"plain_c256", "plain_c256", 0U,
         EnsembleMemberHead::activity, 73, 148.26F},
        {"plain_c512", "plain_c512", 0U,
         EnsembleMemberHead::activity, 136, 315.7938F},
        {"plain_c1024", "plain_c1024", 0U,
         EnsembleMemberHead::activity, 289, 628.6224F},
        {"hc_c256", "hc_c256", 1U,
         EnsembleMemberHead::activity, 68, 148.26F},
        {"hc_c512", "hc_c512", 1U,
         EnsembleMemberHead::activity, 133, 312.8286F},
        {"hprofile_c256", "activity_hprofile_c256", 3U,
         EnsembleMemberHead::activity, 75, 134.9166F},
        {"cattack_c256", "activity_cattack_c256", 4U,
         EnsembleMemberHead::activity, 67, 149.7426F},
    }},
    {{
        {"c256_q1", "c256_q1", 2U,
         EnsembleMemberHead::onset, 151, 85.9908F},
        {"c256_q2", "c256_q2", 2U,
         EnsembleMemberHead::onset, 88, 78.5778F},
        {"c256_q4", "c256_q4", 2U,
         EnsembleMemberHead::onset, 80, 63.7518F},
        {"c256_q8", "c256_q8", 2U,
         EnsembleMemberHead::onset, 38, 60.7866F},
        {"c256_q4_seed19", "c256_q4_seed19", 2U,
         EnsembleMemberHead::onset, 60, 60.7866F},
        {"c512_q4", "c512_q4", 2U,
         EnsembleMemberHead::onset, 153, 131.9514F},
        {"hprofile_c256", "onset_hprofile_c256", 3U,
         EnsembleMemberHead::onset, 60, 88.956F},
        {"c1024_q4", "c1024_q4", 2U,
         EnsembleMemberHead::onset, 283, 268.3506F},
        {"cattack_c256", "onset_cattack_c256", 4U,
         EnsembleMemberHead::onset, 87, 78.5778F},
        {"strict_cap16", "strict_cap16", 2U,
         EnsembleMemberHead::onset, 144, 140.847F},
    }},
};

}  // namespace

const StrictCap16V3Manifest& strict_cap16_v3_manifest() noexcept {
    return kManifest;
}

struct StrictCap16V3Coordinator::Impl {
    struct BankRuntime {
        EnsembleCpuFramePredictor predictor;
        std::vector<std::int32_t> raw_scores;
        std::vector<std::int32_t> activity_scores;
        std::vector<std::uint8_t> activity_predictions;
        std::vector<std::int32_t> onset_scores;
        std::vector<std::uint8_t> onset_predictions;
        EnsembleFrameOutputBuffers outputs;

        explicit BankRuntime(
            EnsembleBundle bundle,
            const bool allow_legacy)
            : predictor(std::move(bundle), allow_legacy),
              raw_scores(predictor.raw_member_score_count()),
              activity_scores(predictor.output_count()),
              activity_predictions(predictor.output_count()),
              onset_scores(predictor.output_count()),
              onset_predictions(predictor.output_count()),
              outputs{
                  raw_scores.data(), raw_scores.size(),
                  activity_scores.data(), activity_scores.size(),
                  activity_predictions.data(), activity_predictions.size(),
                  onset_scores.data(), onset_scores.size(),
                  onset_predictions.data(), onset_predictions.size(),
              } {}
    };

    struct RuntimeRoute {
        std::uint8_t bank_index = 0U;
        std::size_t member_index = 0U;
    };

    std::vector<BankRuntime> banks;
    std::array<std::size_t, kStrictCap16V3PackedRowCount> word_counts{};
    std::array<RuntimeRoute, kStrictCap16V3ActivityMemberCount>
        activity_routes{};
    std::array<RuntimeRoute, kStrictCap16V3OnsetMemberCount> onset_routes{};

    explicit Impl(
        std::array<EnsembleBundle,
                   kStrictCap16V3LogicalBankCount> bundles) {
        banks.reserve(kStrictCap16V3LogicalBankCount);
        std::array<bool, kStrictCap16V3PackedRowCount> row_seen{};
        for (std::size_t index = 0U; index < bundles.size(); ++index) {
            auto& bundle = bundles[index];
            const auto& expected = kManifest.banks[index];
            validate_ensemble_bundle(bundle);
            if (bundle.format_version != expected.bundle_format_version ||
                bundle.feature_count != expected.feature_count ||
                bundle.output_count != kStrictCap16V3OutputCount ||
                bundle.sample_rate != kManifest.sample_rate ||
                bundle.hop_size != kManifest.hop_size ||
                bundle.midi_min != kManifest.midi_min ||
                bundle.midi_max != kManifest.midi_max) {
                throw std::invalid_argument(
                    std::string("strict-cap16-v3 bundle geometry/format differs: ") +
                    expected.logical_id);
            }
            if (bundle.bundle_checksum_sha256 != parse_sha256(
                    expected.bundle_checksum_sha256, "bundle checksum") ||
                bundle.feature_fingerprint_sha256 != parse_sha256(
                    expected.embedded_feature_fingerprint_sha256,
                    "embedded feature fingerprint")) {
                throw std::invalid_argument(
                    std::string("strict-cap16-v3 bundle authentication failed: ") +
                    expected.logical_id);
            }
            const auto row = row_index(expected.packed_row);
            const auto words =
                (static_cast<std::size_t>(bundle.feature_count) + 31U) / 32U;
            if (row_seen[row] && word_counts[row] != words) {
                throw std::invalid_argument(
                    "strict-cap16-v3 logical banks disagree on packed row size");
            }
            word_counts[row] = words;
            row_seen[row] = true;
            banks.emplace_back(
                std::move(bundle),
                expected.bundle_format_version ==
                    kEnsembleBundleLegacyFormatVersion);
        }
        if (!std::all_of(row_seen.begin(), row_seen.end(),
                         [](const bool value) { return value; })) {
            throw std::invalid_argument(
                "strict-cap16-v3 manifest does not cover all packed rows");
        }

        const auto resolve = [&](const StrictCap16V3MemberManifest& route) {
            if (route.logical_bank_index >= banks.size()) {
                throw std::invalid_argument(
                    "strict-cap16-v3 member route has invalid bank index");
            }
            const auto& members = banks[route.logical_bank_index]
                                      .predictor.prepared_bundle().members;
            const auto iterator = std::find_if(
                members.begin(), members.end(),
                [&](const SparseTmEnsembleMember& member) {
                    return member.identifier == route.bundle_identifier;
                });
            if (iterator == members.end() || iterator->head != route.head ||
                iterator->score_threshold != route.score_threshold ||
                !same_float_bits(iterator->robust_scale, route.robust_scale)) {
                throw std::invalid_argument(
                    std::string("strict-cap16-v3 selected member differs: ") +
                    route.global_identifier);
            }
            return RuntimeRoute{
                route.logical_bank_index,
                static_cast<std::size_t>(iterator - members.begin()),
            };
        };
        for (std::size_t index = 0U;
             index < activity_routes.size(); ++index) {
            activity_routes[index] = resolve(kManifest.activity_members[index]);
        }
        for (std::size_t index = 0U; index < onset_routes.size(); ++index) {
            onset_routes[index] = resolve(kManifest.onset_members[index]);
        }
    }

    template <std::size_t MemberCount>
    void fuse(
        const std::array<RuntimeRoute, MemberCount>& routes,
        const StrictCap16V3HeadManifest& head,
        float* normalized_scores,
        std::int32_t* quantized_scores,
        std::uint8_t* predictions) const noexcept {
        constexpr auto member_count_f32 = static_cast<float>(MemberCount);
        for (std::size_t output = 0U;
             output < kStrictCap16V3OutputCount; ++output) {
            float sum = 0.0F;
            for (std::size_t route_index = 0U;
                 route_index < routes.size(); ++route_index) {
                const auto& route = routes[route_index];
                const auto& bank = banks[route.bank_index];
                const auto& member = bank.predictor.prepared_bundle()
                                         .members[route.member_index];
                const auto raw = bank.raw_scores[
                    route.member_index * kStrictCap16V3OutputCount + output];
                const float centered = static_cast<float>(raw) -
                    static_cast<float>(member.score_threshold);
                const float normalized = centered / member.robust_scale;
                // Fixed global route order and explicit float32 accumulation
                // match np.mean(..., axis=0, dtype=np.float32).
                sum = sum + normalized;
            }
            const float fused = sum / member_count_f32;
            const auto quantized =
                quantize_ties_to_even(fused, head.quantization);
            normalized_scores[output] = fused;
            quantized_scores[output] = quantized;
            predictions[output] =
                quantized >= head.ensemble_threshold ? 1U : 0U;
        }
    }
};

StrictCap16V3Coordinator::StrictCap16V3Coordinator(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

StrictCap16V3Coordinator::~StrictCap16V3Coordinator() = default;
StrictCap16V3Coordinator::StrictCap16V3Coordinator(
    StrictCap16V3Coordinator&&) noexcept = default;
StrictCap16V3Coordinator& StrictCap16V3Coordinator::operator=(
    StrictCap16V3Coordinator&&) noexcept = default;

StrictCap16V3Coordinator StrictCap16V3Coordinator::load(
    const std::filesystem::path& package_root) {
    std::array<std::filesystem::path,
               kStrictCap16V3LogicalBankCount> paths;
    for (std::size_t index = 0U; index < paths.size(); ++index) {
        paths[index] =
            package_root / "bundles" /
            kManifest.banks[index].bundle_filename;
    }
    return prepare(paths);
}

StrictCap16V3Coordinator StrictCap16V3Coordinator::prepare(
    const std::array<std::filesystem::path,
                     kStrictCap16V3LogicalBankCount>& bundle_paths) {
    std::array<EnsembleBundle, kStrictCap16V3LogicalBankCount> bundles;
    for (std::size_t index = 0U; index < bundles.size(); ++index) {
        bundles[index] = load_ensemble_bundle(bundle_paths[index]);
    }
    return StrictCap16V3Coordinator(
        std::make_unique<Impl>(std::move(bundles)));
}

std::size_t StrictCap16V3Coordinator::required_packed_word_count(
    const StrictCap16V3PackedRowId row) const noexcept {
    const auto index = row_index(row);
    if (!impl_ || index >= impl_->word_counts.size()) {
        return 0U;
    }
    return impl_->word_counts[index];
}

std::uint32_t StrictCap16V3Coordinator::output_count() const noexcept {
    return impl_ ? kStrictCap16V3OutputCount : 0U;
}

StrictCap16V3PredictStatus StrictCap16V3Coordinator::predict_frame(
    const StrictCap16V3FrameInput& input,
    const StrictCap16V3FrameOutputBuffers& outputs) noexcept {
    if (!impl_) {
        return StrictCap16V3PredictStatus::unprepared;
    }
    for (std::size_t row = 0U; row < input.rows.size(); ++row) {
        if (input.rows[row].words == nullptr) {
            return StrictCap16V3PredictStatus::null_packed_row;
        }
        if (input.rows[row].word_count != impl_->word_counts[row]) {
            return StrictCap16V3PredictStatus::wrong_packed_row_word_count;
        }
    }
    if (outputs.normalized_activity_scores == nullptr ||
        outputs.quantized_activity_scores == nullptr ||
        outputs.activity_predictions == nullptr ||
        outputs.normalized_onset_scores == nullptr ||
        outputs.quantized_onset_scores == nullptr ||
        outputs.onset_predictions == nullptr) {
        return StrictCap16V3PredictStatus::null_output_buffer;
    }
    if (outputs.normalized_activity_score_count < kStrictCap16V3OutputCount ||
        outputs.quantized_activity_score_count < kStrictCap16V3OutputCount ||
        outputs.activity_prediction_count < kStrictCap16V3OutputCount ||
        outputs.normalized_onset_score_count < kStrictCap16V3OutputCount ||
        outputs.quantized_onset_score_count < kStrictCap16V3OutputCount ||
        outputs.onset_prediction_count < kStrictCap16V3OutputCount) {
        return StrictCap16V3PredictStatus::output_buffer_too_small;
    }

    for (std::size_t bank_index = 0U;
         bank_index < impl_->banks.size(); ++bank_index) {
        auto& bank = impl_->banks[bank_index];
        const auto row = row_index(kManifest.banks[bank_index].packed_row);
        const auto status = bank.predictor.predict_frame(
            input.rows[row].words,
            input.rows[row].word_count,
            bank.outputs);
        if (status != EnsembleFramePredictStatus::success) {
            return StrictCap16V3PredictStatus::bank_predictor_failure;
        }
    }
    impl_->fuse(
        impl_->activity_routes,
        kManifest.activity,
        outputs.normalized_activity_scores,
        outputs.quantized_activity_scores,
        outputs.activity_predictions);
    impl_->fuse(
        impl_->onset_routes,
        kManifest.onset,
        outputs.normalized_onset_scores,
        outputs.quantized_onset_scores,
        outputs.onset_predictions);
    return StrictCap16V3PredictStatus::success;
}

}  // namespace tmgm::native
