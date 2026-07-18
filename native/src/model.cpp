#include "tmgm/model.hpp"

#include "tmgm/dataset.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace tmgm::native {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{'T', 'M', 'G', 'M', 'M', 'O', 'D', 0};
constexpr std::uint32_t kFeatureNegationFlag = 1U << 0U;
constexpr std::uint32_t kBoostTruePositiveFlag = 1U << 1U;
constexpr std::uint32_t kOnsetSustainHardNegativeFlag = 1U << 2U;
constexpr std::uint32_t kOnsetSustainHardNegativeWeightOnlyFlag = 1U << 3U;
constexpr std::uint32_t kKnownFlags = kFeatureNegationFlag |
    kBoostTruePositiveFlag | kOnsetSustainHardNegativeFlag |
    kOnsetSustainHardNegativeWeightOnlyFlag;
constexpr std::size_t kChecksumOffset = 160U;
constexpr std::size_t kChecksumBytes = 32U;
constexpr std::size_t kFeatureFingerprintOffset = 192U;
constexpr std::size_t kFeatureFingerprintBytes = 32U;
constexpr std::size_t kPayloadChunkWords = 16U * 1024U;

[[nodiscard]] std::runtime_error format_error(const std::string& message) {
    return std::runtime_error("invalid TMGMMOD model: " + message);
}

[[nodiscard]] std::invalid_argument model_error(const std::string& message) {
    return std::invalid_argument("invalid native TM model: " + message);
}

[[nodiscard]] std::invalid_argument compatibility_error(
    const std::string& message) {
    return std::invalid_argument(
        "dataset is incompatible with native TM model: " + message);
}

[[nodiscard]] std::uint64_t checked_multiply(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* label) {
    if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw model_error(std::string(label) + " overflows uint64");
    }
    return left * right;
}

[[nodiscard]] std::uint64_t checked_add(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* label) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw model_error(std::string(label) + " overflows uint64");
    }
    return left + right;
}

[[nodiscard]] std::size_t checked_size(const std::uint64_t value, const char* label) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw model_error(std::string(label) + " does not fit into address space");
    }
    return static_cast<std::size_t>(value);
}

template <typename T>
[[nodiscard]] T decode_little_endian(const std::uint8_t* bytes) {
    static_assert(std::is_unsigned_v<T>);
    T value = 0;
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        value |= static_cast<T>(bytes[index]) << (index * 8U);
    }
    return value;
}

[[nodiscard]] std::int32_t decode_i32(const std::uint8_t* bytes) {
    const auto bits = decode_little_endian<std::uint32_t>(bytes);
    std::int32_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

[[nodiscard]] float decode_f32(const std::uint8_t* bytes) {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    const auto bits = decode_little_endian<std::uint32_t>(bytes);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

template <typename T>
void encode_little_endian(std::uint8_t* bytes, T value) {
    static_assert(std::is_unsigned_v<T>);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        bytes[index] = static_cast<std::uint8_t>(value & static_cast<T>(0xffU));
        value >>= 8U;
    }
}

void encode_i32(std::uint8_t* bytes, const std::int32_t value) {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    encode_little_endian<std::uint32_t>(bytes, bits);
}

void encode_f32(std::uint8_t* bytes, const float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    encode_little_endian<std::uint32_t>(bytes, bits);
}

void read_exact(std::ifstream& stream, void* destination, const std::size_t byte_count) {
    if (byte_count == 0U) {
        return;
    }
    stream.read(static_cast<char*>(destination), static_cast<std::streamsize>(byte_count));
    if (!stream || static_cast<std::size_t>(stream.gcount()) != byte_count) {
        throw format_error("file is truncated");
    }
}

void write_exact(std::ofstream& stream, const void* source, const std::size_t byte_count) {
    if (byte_count == 0U) {
        return;
    }
    stream.write(static_cast<const char*>(source), static_cast<std::streamsize>(byte_count));
    if (!stream) {
        throw std::runtime_error("failed to write TMGMMOD model");
    }
}

class Sha256 {
public:
    void update(const void* source, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(source);
        total_bytes_ += static_cast<std::uint64_t>(size);
        while (size != 0U) {
            const auto amount = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, bytes, amount);
            block_size_ += amount;
            bytes += amount;
            size -= amount;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0U;
            }
        }
    }

    [[nodiscard]] std::array<std::uint8_t, 32> finish() {
        const auto bit_length = total_bytes_ * 8U;
        block_[block_size_++] = 0x80U;
        if (block_size_ > 56U) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                      block_.end(),
                      std::uint8_t{0U});
            transform(block_.data());
            block_size_ = 0U;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                  block_.begin() + 56,
                  std::uint8_t{0U});
        for (std::size_t index = 0; index < 8U; ++index) {
            block_[63U - index] =
                static_cast<std::uint8_t>(bit_length >> (index * 8U));
        }
        transform(block_.data());

        std::array<std::uint8_t, 32> digest{};
        for (std::size_t word = 0; word < state_.size(); ++word) {
            for (std::size_t byte = 0; byte < 4U; ++byte) {
                digest[word * 4U + byte] = static_cast<std::uint8_t>(
                    state_[word] >> ((3U - byte) * 8U));
            }
        }
        return digest;
    }

private:
    static constexpr std::array<std::uint32_t, 64> kRoundConstants{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
        0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
        0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
        0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };

    [[nodiscard]] static std::uint32_t rotate_right(
        const std::uint32_t value,
        const unsigned shift) noexcept {
        return (value >> shift) | (value << (32U - shift));
    }

    void transform(const std::uint8_t* block) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16U; ++index) {
            const auto offset = index * 4U;
            words[index] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                           (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
                           (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
                           static_cast<std::uint32_t>(block[offset + 3U]);
        }
        for (std::size_t index = 16U; index < words.size(); ++index) {
            const auto left = words[index - 15U];
            const auto right = words[index - 2U];
            const auto sigma0 =
                rotate_right(left, 7U) ^ rotate_right(left, 18U) ^ (left >> 3U);
            const auto sigma1 =
                rotate_right(right, 17U) ^ rotate_right(right, 19U) ^ (right >> 10U);
            words[index] =
                words[index - 16U] + sigma0 + words[index - 7U] + sigma1;
        }

        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const auto sum1 =
                rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
            const auto choose = (e & f) ^ (~e & g);
            const auto temporary1 =
                h + sum1 + choose + kRoundConstants[index] + words[index];
            const auto sum0 =
                rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{
        0x6a09e667U,
        0xbb67ae85U,
        0x3c6ef372U,
        0xa54ff53aU,
        0x510e527fU,
        0x9b05688cU,
        0x1f83d9abU,
        0x5be0cd19U,
    };
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_ = 0U;
    std::uint64_t total_bytes_ = 0U;
};

struct PayloadSizes {
    std::uint64_t ta_words = 0U;
    std::uint64_t weight_words = 0U;
    std::uint64_t ta_bytes = 0U;
    std::uint64_t weight_bytes = 0U;
    std::uint64_t file_bytes = 0U;
};

[[nodiscard]] PayloadSizes payload_sizes(const NativeTmModel& model) {
    PayloadSizes sizes;
    const auto literals = static_cast<std::uint64_t>(model.dimensions.feature_count) * 2U;
    const auto literal_words = (literals + kTmModelWordBits - 1U) / kTmModelWordBits;
    sizes.ta_words = checked_multiply(
        checked_multiply(model.dimensions.clause_count,
                         model.dimensions.state_bits,
                         "TA word count"),
        literal_words,
        "TA word count");
    sizes.weight_words = checked_multiply(
        model.dimensions.output_count,
        model.dimensions.clause_count,
        "weight count");
    sizes.ta_bytes = checked_multiply(sizes.ta_words, 4U, "TA byte size");
    sizes.weight_bytes = checked_multiply(sizes.weight_words, 4U, "weight byte size");
    sizes.file_bytes = checked_add(
        checked_add(kTmModelHeaderBytes, sizes.ta_bytes, "model file size"),
        sizes.weight_bytes,
        "model file size");
    return sizes;
}

[[nodiscard]] std::uint32_t model_flags(const NativeTmModel& model) noexcept {
    std::uint32_t flags = 0U;
    if (model.training.feature_negation) {
        flags |= kFeatureNegationFlag;
    }
    if (model.training.boost_true_positive_feedback) {
        flags |= kBoostTruePositiveFlag;
    }
    if (model.training.onset_sustain_hard_negatives) {
        flags |= kOnsetSustainHardNegativeFlag;
    }
    if (model.training.onset_sustain_hard_negative_weight_only) {
        flags |= kOnsetSustainHardNegativeWeightOnlyFlag;
    }
    return flags;
}

[[nodiscard]] std::array<std::uint8_t, kTmModelHeaderBytes> encode_header(
    const NativeTmModel& model,
    const std::array<std::uint8_t, kChecksumBytes>& checksum) {
    const auto sizes = payload_sizes(model);
    const auto literal_count = tm_model_literal_count(model);
    const auto literal_words = tm_model_literal_word_count(model);
    const auto weights_offset = static_cast<std::uint64_t>(kTmModelHeaderBytes) +
                                sizes.ta_bytes;

    std::array<std::uint8_t, kTmModelHeaderBytes> bytes{};
    std::copy(kMagic.begin(), kMagic.end(), bytes.begin());
    const auto has_feature_fingerprint = std::any_of(
        model.feature_fingerprint_sha256.begin(),
        model.feature_fingerprint_sha256.end(),
        [](const std::uint8_t value) { return value != 0U; });
    encode_little_endian<std::uint32_t>(
        bytes.data() + 8U,
        has_feature_fingerprint
            ? kTmModelFormatVersion
            : kTmModelPreviousFormatVersion);
    encode_little_endian<std::uint32_t>(bytes.data() + 12U, kTmModelHeaderBytes);
    encode_little_endian<std::uint32_t>(bytes.data() + 16U, kTmModelWordBits);
    encode_little_endian<std::uint32_t>(bytes.data() + 20U, model_flags(model));
    encode_little_endian<std::uint32_t>(
        bytes.data() + 24U, static_cast<std::uint32_t>(model.head));
    encode_little_endian<std::uint32_t>(
        bytes.data() + 28U, model.dimensions.state_bits);
    encode_little_endian<std::uint32_t>(
        bytes.data() + 32U, model.dimensions.feature_count);
    encode_little_endian<std::uint32_t>(
        bytes.data() + 36U, model.dimensions.output_count);
    encode_little_endian<std::uint32_t>(
        bytes.data() + 40U, model.dimensions.clause_count);
    encode_little_endian<std::uint32_t>(bytes.data() + 44U, literal_count);
    encode_little_endian<std::uint32_t>(bytes.data() + 48U, literal_words);
    encode_i32(bytes.data() + 52U, model.training.threshold);
    encode_i32(bytes.data() + 56U, model.score_threshold);
    encode_little_endian<std::uint32_t>(
        bytes.data() + 60U, model.training.max_included_literals);
    encode_f32(bytes.data() + 64U, model.training.specificity);
    encode_f32(bytes.data() + 68U, model.training.negative_samples);
    encode_f32(bytes.data() + 72U, model.training.type_i_ii_ratio);
    encode_little_endian<std::uint32_t>(
        bytes.data() + 76U, model.training.epochs_trained);
    encode_little_endian<std::uint64_t>(bytes.data() + 80U, model.training.seed);
    encode_i32(bytes.data() + 88U, model.midi.minimum_note);
    encode_i32(bytes.data() + 92U, model.midi.maximum_note);
    encode_little_endian<std::uint32_t>(bytes.data() + 96U, model.midi.channel);
    encode_little_endian<std::uint32_t>(
        bytes.data() + 100U, model.midi.audio_sample_rate);
    encode_little_endian<std::uint32_t>(
        bytes.data() + 104U, model.midi.analysis_hop_samples);
    encode_f32(
        bytes.data() + 108U,
        model.training.onset_sustain_hard_negative_probability);
    encode_little_endian<std::uint64_t>(bytes.data() + 112U, kTmModelHeaderBytes);
    encode_little_endian<std::uint64_t>(bytes.data() + 120U, sizes.ta_bytes);
    encode_little_endian<std::uint64_t>(bytes.data() + 128U, weights_offset);
    encode_little_endian<std::uint64_t>(bytes.data() + 136U, sizes.weight_bytes);
    encode_little_endian<std::uint64_t>(
        bytes.data() + 144U, sizes.ta_bytes + sizes.weight_bytes);
    encode_little_endian<std::uint64_t>(bytes.data() + 152U, sizes.file_bytes);
    std::copy(checksum.begin(), checksum.end(), bytes.begin() + kChecksumOffset);
    if (has_feature_fingerprint) {
        std::copy(
            model.feature_fingerprint_sha256.begin(),
            model.feature_fingerprint_sha256.end(),
            bytes.begin() + kFeatureFingerprintOffset);
    }
    return bytes;
}

void hash_u32_payload(Sha256& sha, const std::vector<std::uint32_t>& words) {
    std::array<std::uint8_t, kPayloadChunkWords * 4U> bytes{};
    std::size_t offset = 0U;
    while (offset < words.size()) {
        const auto count = std::min(kPayloadChunkWords, words.size() - offset);
        for (std::size_t index = 0; index < count; ++index) {
            encode_little_endian<std::uint32_t>(
                bytes.data() + index * 4U, words[offset + index]);
        }
        sha.update(bytes.data(), count * 4U);
        offset += count;
    }
}

void hash_i32_payload(Sha256& sha, const std::vector<std::int32_t>& words) {
    std::array<std::uint8_t, kPayloadChunkWords * 4U> bytes{};
    std::size_t offset = 0U;
    while (offset < words.size()) {
        const auto count = std::min(kPayloadChunkWords, words.size() - offset);
        for (std::size_t index = 0; index < count; ++index) {
            encode_i32(bytes.data() + index * 4U, words[offset + index]);
        }
        sha.update(bytes.data(), count * 4U);
        offset += count;
    }
}

void write_u32_payload(
    std::ofstream& stream,
    const std::vector<std::uint32_t>& words) {
    std::array<std::uint8_t, kPayloadChunkWords * 4U> bytes{};
    std::size_t offset = 0U;
    while (offset < words.size()) {
        const auto count = std::min(kPayloadChunkWords, words.size() - offset);
        for (std::size_t index = 0; index < count; ++index) {
            encode_little_endian<std::uint32_t>(
                bytes.data() + index * 4U, words[offset + index]);
        }
        write_exact(stream, bytes.data(), count * 4U);
        offset += count;
    }
}

void write_i32_payload(
    std::ofstream& stream,
    const std::vector<std::int32_t>& words) {
    std::array<std::uint8_t, kPayloadChunkWords * 4U> bytes{};
    std::size_t offset = 0U;
    while (offset < words.size()) {
        const auto count = std::min(kPayloadChunkWords, words.size() - offset);
        for (std::size_t index = 0; index < count; ++index) {
            encode_i32(bytes.data() + index * 4U, words[offset + index]);
        }
        write_exact(stream, bytes.data(), count * 4U);
        offset += count;
    }
}

template <typename Value, typename Decode>
void read_payload(
    std::ifstream& stream,
    std::vector<Value>& words,
    Sha256* sha,
    Decode decode) {
    std::array<std::uint8_t, kPayloadChunkWords * 4U> bytes{};
    std::size_t offset = 0U;
    while (offset < words.size()) {
        const auto count = std::min(kPayloadChunkWords, words.size() - offset);
        read_exact(stream, bytes.data(), count * 4U);
        if (sha != nullptr) {
            sha->update(bytes.data(), count * 4U);
        }
        for (std::size_t index = 0; index < count; ++index) {
            words[offset + index] = decode(bytes.data() + index * 4U);
        }
        offset += count;
    }
}

[[nodiscard]] bool all_zero(
    const std::array<std::uint8_t, kTmModelHeaderBytes>& header,
    const std::size_t begin,
    const std::size_t end) {
    return std::all_of(
        header.begin() + static_cast<std::ptrdiff_t>(begin),
        header.begin() + static_cast<std::ptrdiff_t>(end),
        [](const std::uint8_t value) { return value == 0U; });
}

}  // namespace

std::uint32_t tm_model_literal_count(const NativeTmModel& model) noexcept {
    return model.dimensions.feature_count * 2U;
}

std::uint32_t tm_model_literal_word_count(const NativeTmModel& model) noexcept {
    const auto literals = tm_model_literal_count(model);
    return literals / kTmModelWordBits +
           (literals % kTmModelWordBits == 0U ? 0U : 1U);
}

void validate_tm_model(const NativeTmModel& model) {
    if (model.head != TmModelHead::activity && model.head != TmModelHead::onset) {
        throw model_error("unknown target head");
    }
    if (model.dimensions.feature_count == 0U ||
        model.dimensions.output_count == 0U ||
        model.dimensions.clause_count == 0U) {
        throw model_error("feature, output, and clause counts must be positive");
    }
    if (model.dimensions.feature_count >
        std::numeric_limits<std::uint32_t>::max() / 2U) {
        throw model_error("feature count is too large for positive/negative literals");
    }
    if (model.dimensions.state_bits < 2U || model.dimensions.state_bits > 16U) {
        throw model_error("state_bits must be in [2, 16]");
    }
    if (model.training.threshold <= 0) {
        throw model_error("training threshold must be positive");
    }
    if (!std::isfinite(model.training.specificity) ||
        model.training.specificity <= 0.0F) {
        throw model_error("specificity must be finite and positive");
    }
    if (!std::isfinite(model.training.negative_samples) ||
        model.training.negative_samples < 0.0F) {
        throw model_error("negative_samples must be finite and non-negative");
    }
    if (!std::isfinite(model.training.type_i_ii_ratio) ||
        model.training.type_i_ii_ratio <= 0.0F) {
        throw model_error("type_i_ii_ratio must be finite and positive");
    }
    if (model.training.onset_sustain_hard_negatives &&
        model.head != TmModelHead::onset) {
        throw model_error(
            "onset sustain hard negatives are valid only for onset models");
    }
    if (!std::isfinite(
            model.training.onset_sustain_hard_negative_probability) ||
        model.training.onset_sustain_hard_negative_probability < 0.0F ||
        model.training.onset_sustain_hard_negative_probability > 1.0F ||
        (model.training.onset_sustain_hard_negatives &&
         model.training.onset_sustain_hard_negative_probability <= 0.0F) ||
        (!model.training.onset_sustain_hard_negatives &&
         model.training.onset_sustain_hard_negative_probability != 0.0F)) {
        throw model_error(
            "onset sustain hard-negative probability disagrees with policy flag");
    }
    if (model.training.onset_sustain_hard_negative_weight_only &&
        !model.training.onset_sustain_hard_negatives) {
        throw model_error(
            "weight-only sustain hard negatives require the policy flag");
    }
    const auto literal_count = tm_model_literal_count(model);
    if (model.training.max_included_literals > literal_count) {
        throw model_error("max_included_literals exceeds literal count");
    }
    if (model.midi.minimum_note < 0 || model.midi.maximum_note > 127 ||
        model.midi.maximum_note < model.midi.minimum_note) {
        throw model_error("MIDI note range must be inside [0, 127]");
    }
    const auto midi_outputs = static_cast<std::uint32_t>(
        model.midi.maximum_note - model.midi.minimum_note + 1);
    if (midi_outputs != model.dimensions.output_count) {
        throw model_error("MIDI note range does not match output count");
    }
    if (model.midi.channel < 1U || model.midi.channel > 16U) {
        throw model_error("MIDI channel must be in [1, 16]");
    }
    if (model.midi.audio_sample_rate == 0U ||
        model.midi.analysis_hop_samples == 0U) {
        throw model_error("audio sample rate and analysis hop must be positive");
    }

    const auto sizes = payload_sizes(model);
    if (model.ta_bitplanes.size() != checked_size(sizes.ta_words, "TA word count")) {
        throw model_error("TA bit-plane payload has the wrong size");
    }
    if (model.weights.size() != checked_size(sizes.weight_words, "weight count")) {
        throw model_error("weight payload has the wrong size");
    }
}

void validate_tm_dataset_compatibility(
    const NativeDataset& dataset,
    const NativeTmModel& model,
    const bool allow_legacy_feature_contract) {
    if (dataset.header.feature_count != model.dimensions.feature_count) {
        throw compatibility_error("feature count differs");
    }
    if (dataset.header.note_count != model.dimensions.output_count) {
        throw compatibility_error("output count differs");
    }
    if (dataset.header.midi_min != model.midi.minimum_note ||
        dataset.header.midi_max != model.midi.maximum_note) {
        throw compatibility_error("MIDI range differs");
    }
    if (dataset.header.sample_rate != model.midi.audio_sample_rate) {
        throw compatibility_error("sample rate differs");
    }
    if (dataset.header.hop_size != model.midi.analysis_hop_samples) {
        throw compatibility_error("analysis hop differs");
    }
    const auto dataset_legacy = std::all_of(
        dataset.header.feature_fingerprint_sha256.begin(),
        dataset.header.feature_fingerprint_sha256.end(),
        [](const std::uint8_t value) { return value == 0U; });
    const auto model_legacy = std::all_of(
        model.feature_fingerprint_sha256.begin(),
        model.feature_fingerprint_sha256.end(),
        [](const std::uint8_t value) { return value == 0U; });
    if (dataset_legacy || model_legacy) {
        if (!allow_legacy_feature_contract) {
            throw compatibility_error(
                "feature-semantics fingerprint is unavailable for a legacy "
                "artifact; pass an explicit legacy opt-in only for audit use");
        }
    } else if (dataset.header.feature_fingerprint_sha256 !=
               model.feature_fingerprint_sha256) {
        throw compatibility_error("feature-semantics fingerprint differs");
    }
}

std::array<std::uint8_t, 32> calculate_tm_model_checksum(
    const NativeTmModel& model) {
    validate_tm_model(model);
    const std::array<std::uint8_t, kChecksumBytes> empty_checksum{};
    const auto header = encode_header(model, empty_checksum);
    Sha256 sha;
    sha.update(header.data(), header.size());
    hash_u32_payload(sha, model.ta_bitplanes);
    hash_i32_payload(sha, model.weights);
    return sha.finish();
}

std::string tm_model_checksum_hex(
    const std::array<std::uint8_t, 32>& digest) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const auto byte : digest) {
        stream << std::setw(2) << static_cast<unsigned>(byte);
    }
    return stream.str();
}

void save_tm_model(
    const std::filesystem::path& path,
    const NativeTmModel& model) {
    validate_tm_model(model);
    const auto checksum = calculate_tm_model_checksum(model);
    const auto header = encode_header(model, checksum);

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("failed to open TMGMMOD model for writing: " +
                                 path.string());
    }
    write_exact(stream, header.data(), header.size());
    write_u32_payload(stream, model.ta_bitplanes);
    write_i32_payload(stream, model.weights);
    stream.flush();
    if (!stream) {
        throw std::runtime_error("failed to finish writing TMGMMOD model: " +
                                 path.string());
    }
}

NativeTmModel load_tm_model(
    const std::filesystem::path& path,
    const bool verify_checksum) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open TMGMMOD model: " + path.string());
    }

    std::array<std::uint8_t, kTmModelHeaderBytes> header{};
    read_exact(stream, header.data(), header.size());
    if (!std::equal(kMagic.begin(), kMagic.end(), header.begin())) {
        throw format_error("wrong magic");
    }
    const auto format_version =
        decode_little_endian<std::uint32_t>(header.data() + 8U);
    if (format_version != kTmModelLegacyFormatVersion &&
        format_version != kTmModelPreviousFormatVersion &&
        format_version != kTmModelFormatVersion) {
        throw format_error("unsupported version");
    }
    if (decode_little_endian<std::uint32_t>(header.data() + 12U) !=
        kTmModelHeaderBytes) {
        throw format_error("wrong header size");
    }
    if (decode_little_endian<std::uint32_t>(header.data() + 16U) !=
        kTmModelWordBits) {
        throw format_error("unsupported packed word size");
    }
    const auto flags = decode_little_endian<std::uint32_t>(header.data() + 20U);
    if ((flags & ~kKnownFlags) != 0U) {
        throw format_error("unknown flags");
    }
    NativeTmModel model;
    if (format_version == kTmModelFormatVersion) {
        std::copy_n(
            header.begin() +
                static_cast<std::ptrdiff_t>(kFeatureFingerprintOffset),
            kFeatureFingerprintBytes,
            model.feature_fingerprint_sha256.begin());
        if (std::all_of(
                model.feature_fingerprint_sha256.begin(),
                model.feature_fingerprint_sha256.end(),
                [](const std::uint8_t value) { return value == 0U; })) {
            throw format_error("v3 feature-semantics fingerprint is zero");
        }
        if (!all_zero(header, 224U, 256U)) {
            throw format_error("reserved header bytes are non-zero");
        }
    } else if (!all_zero(header, 192U, 256U)) {
        throw format_error("legacy reserved header bytes are non-zero");
    }
    const auto head = decode_little_endian<std::uint32_t>(header.data() + 24U);
    if (head == static_cast<std::uint32_t>(TmModelHead::activity)) {
        model.head = TmModelHead::activity;
    } else if (head == static_cast<std::uint32_t>(TmModelHead::onset)) {
        model.head = TmModelHead::onset;
    } else {
        throw format_error("unknown target head");
    }
    model.dimensions.state_bits =
        decode_little_endian<std::uint32_t>(header.data() + 28U);
    model.dimensions.feature_count =
        decode_little_endian<std::uint32_t>(header.data() + 32U);
    model.dimensions.output_count =
        decode_little_endian<std::uint32_t>(header.data() + 36U);
    model.dimensions.clause_count =
        decode_little_endian<std::uint32_t>(header.data() + 40U);
    model.training.threshold = decode_i32(header.data() + 52U);
    model.score_threshold = decode_i32(header.data() + 56U);
    model.training.max_included_literals =
        decode_little_endian<std::uint32_t>(header.data() + 60U);
    model.training.specificity = decode_f32(header.data() + 64U);
    model.training.negative_samples = decode_f32(header.data() + 68U);
    model.training.type_i_ii_ratio = decode_f32(header.data() + 72U);
    model.training.epochs_trained =
        decode_little_endian<std::uint32_t>(header.data() + 76U);
    model.training.seed =
        decode_little_endian<std::uint64_t>(header.data() + 80U);
    model.training.feature_negation = (flags & kFeatureNegationFlag) != 0U;
    model.training.boost_true_positive_feedback =
        (flags & kBoostTruePositiveFlag) != 0U;
    model.training.onset_sustain_hard_negatives =
        (flags & kOnsetSustainHardNegativeFlag) != 0U;
    model.training.onset_sustain_hard_negative_probability =
        decode_f32(header.data() + 108U);
    // Early hard-negative experiments were written as transitional v1 files:
    // the policy flag meant "always select", while the then-reserved
    // probability field stayed zero. Preserve those models when upgrading the
    // reader; v2 requires an explicit probability in (0, 1].
    if (format_version == kTmModelLegacyFormatVersion &&
        model.training.onset_sustain_hard_negatives &&
        model.training.onset_sustain_hard_negative_probability == 0.0F) {
        model.training.onset_sustain_hard_negative_probability = 1.0F;
    }
    model.training.onset_sustain_hard_negative_weight_only =
        (flags & kOnsetSustainHardNegativeWeightOnlyFlag) != 0U;
    model.midi.minimum_note = decode_i32(header.data() + 88U);
    model.midi.maximum_note = decode_i32(header.data() + 92U);
    model.midi.channel =
        decode_little_endian<std::uint32_t>(header.data() + 96U);
    model.midi.audio_sample_rate =
        decode_little_endian<std::uint32_t>(header.data() + 100U);
    model.midi.analysis_hop_samples =
        decode_little_endian<std::uint32_t>(header.data() + 104U);

    try {
        const auto sizes = payload_sizes(model);
        const auto literal_count =
            decode_little_endian<std::uint32_t>(header.data() + 44U);
        const auto literal_words =
            decode_little_endian<std::uint32_t>(header.data() + 48U);
        const auto ta_offset =
            decode_little_endian<std::uint64_t>(header.data() + 112U);
        const auto ta_bytes =
            decode_little_endian<std::uint64_t>(header.data() + 120U);
        const auto weights_offset =
            decode_little_endian<std::uint64_t>(header.data() + 128U);
        const auto weight_bytes =
            decode_little_endian<std::uint64_t>(header.data() + 136U);
        const auto payload_bytes =
            decode_little_endian<std::uint64_t>(header.data() + 144U);
        const auto file_bytes =
            decode_little_endian<std::uint64_t>(header.data() + 152U);

        if (literal_count != tm_model_literal_count(model) ||
            literal_words != tm_model_literal_word_count(model)) {
            throw format_error("literal dimensions are inconsistent");
        }
        if (ta_offset != kTmModelHeaderBytes || ta_bytes != sizes.ta_bytes ||
            weights_offset != kTmModelHeaderBytes + sizes.ta_bytes ||
            weight_bytes != sizes.weight_bytes ||
            payload_bytes != sizes.ta_bytes + sizes.weight_bytes ||
            file_bytes != sizes.file_bytes) {
            throw format_error("payload layout is inconsistent");
        }

        stream.seekg(0, std::ios::end);
        const auto end = stream.tellg();
        if (end < 0 || static_cast<std::uint64_t>(end) != file_bytes) {
            throw format_error("file size does not match header");
        }
        stream.seekg(static_cast<std::streamoff>(kTmModelHeaderBytes), std::ios::beg);
        if (!stream) {
            throw format_error("cannot seek to payload");
        }

        model.ta_bitplanes.resize(checked_size(sizes.ta_words, "TA word count"));
        model.weights.resize(checked_size(sizes.weight_words, "weight count"));

        std::array<std::uint8_t, kChecksumBytes> expected_checksum{};
        std::copy_n(
            header.begin() + static_cast<std::ptrdiff_t>(kChecksumOffset),
            kChecksumBytes,
            expected_checksum.begin());
        auto checksum_header = header;
        std::fill(checksum_header.begin() + static_cast<std::ptrdiff_t>(kChecksumOffset),
                  checksum_header.begin() +
                      static_cast<std::ptrdiff_t>(kChecksumOffset + kChecksumBytes),
                  std::uint8_t{0U});
        Sha256 sha;
        Sha256* sha_pointer = nullptr;
        if (verify_checksum) {
            sha.update(checksum_header.data(), checksum_header.size());
            sha_pointer = &sha;
        }
        read_payload(stream,
                     model.ta_bitplanes,
                     sha_pointer,
                     [](const std::uint8_t* bytes) {
                         return decode_little_endian<std::uint32_t>(bytes);
                     });
        read_payload(stream,
                     model.weights,
                     sha_pointer,
                     [](const std::uint8_t* bytes) { return decode_i32(bytes); });
        if (verify_checksum && sha.finish() != expected_checksum) {
            throw format_error("SHA-256 checksum mismatch");
        }
        validate_tm_model(model);
    } catch (const std::invalid_argument& error) {
        throw format_error(error.what());
    }
    return model;
}

}  // namespace tmgm::native
