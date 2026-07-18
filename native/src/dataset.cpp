#include "tmgm/dataset.hpp"

#include <algorithm>
#include <array>
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

constexpr std::array<std::uint8_t, 8> kMagic{'T', 'M', 'G', 'M', 'D', 'A', 'T', 0};
constexpr std::uint32_t kLegacyVersion = 1;
constexpr std::uint32_t kVersion = 2;
constexpr std::uint32_t kFlags = 0;
constexpr std::size_t kFeatureFingerprintOffset = 176U;
constexpr std::size_t kFeatureFingerprintBytes = 32U;

[[nodiscard]] std::runtime_error format_error(const std::string& message) {
    return std::runtime_error("invalid TMGMDAT dataset: " + message);
}

[[nodiscard]] std::uint64_t checked_multiply(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* label) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw format_error(std::string(label) + " overflows uint64");
    }
    return left * right;
}

[[nodiscard]] std::uint64_t checked_add(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* label) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw format_error(std::string(label) + " overflows uint64");
    }
    return left + right;
}

[[nodiscard]] std::size_t checked_size(const std::uint64_t value, const char* label) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw format_error(std::string(label) + " does not fit into address space");
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::uint32_t word_count(const std::uint32_t columns) {
    if (columns == 0) {
        throw format_error("binary matrix has zero columns");
    }
    return columns / kDatasetWordBits + (columns % kDatasetWordBits == 0 ? 0u : 1u);
}

template <typename T>
[[nodiscard]] T decode_little_endian(const std::uint8_t* bytes) {
    static_assert(std::is_unsigned_v<T>);
    T value = 0;
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        value |= static_cast<T>(bytes[index]) << (index * 8u);
    }
    return value;
}

[[nodiscard]] std::int32_t decode_i32(const std::uint8_t* bytes) {
    const auto unsigned_value = decode_little_endian<std::uint32_t>(bytes);
    std::int32_t value = 0;
    std::memcpy(&value, &unsigned_value, sizeof(value));
    return value;
}

template <typename T>
void encode_little_endian(std::uint8_t* bytes, T value) {
    static_assert(std::is_unsigned_v<T>);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        bytes[index] = static_cast<std::uint8_t>(value & static_cast<T>(0xffu));
        value >>= 8u;
    }
}

void encode_i32(std::uint8_t* bytes, const std::int32_t value) {
    std::uint32_t unsigned_value = 0;
    std::memcpy(&unsigned_value, &value, sizeof(value));
    encode_little_endian<std::uint32_t>(bytes, unsigned_value);
}

void read_exact(std::ifstream& stream, void* destination, const std::size_t byte_count) {
    if (byte_count == 0) {
        return;
    }
    stream.read(static_cast<char*>(destination), static_cast<std::streamsize>(byte_count));
    if (!stream || static_cast<std::size_t>(stream.gcount()) != byte_count) {
        throw format_error("file is truncated");
    }
}

void write_exact(std::ofstream& stream, const void* source, const std::size_t byte_count) {
    if (byte_count == 0) {
        return;
    }
    stream.write(static_cast<const char*>(source), static_cast<std::streamsize>(byte_count));
    if (!stream) {
        throw std::runtime_error("failed to write TMGMDAT payload");
    }
}

[[nodiscard]] bool host_is_little_endian() noexcept {
    constexpr std::uint16_t value = 1;
    return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

class Sha256 {
public:
    Sha256() = default;

    void update(const void* source, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(source);
        total_bytes_ += static_cast<std::uint64_t>(size);
        while (size != 0) {
            const auto amount = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, bytes, amount);
            block_size_ += amount;
            bytes += amount;
            size -= amount;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    [[nodiscard]] std::array<std::uint8_t, 32> finish() {
        const auto bit_length = total_bytes_ * 8u;
        block_[block_size_++] = 0x80u;
        if (block_size_ > 56) {
            std::fill(
                block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                block_.end(),
                std::uint8_t{0});
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(
            block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
            block_.begin() + 56,
            std::uint8_t{0});
        for (std::size_t index = 0; index < 8; ++index) {
            block_[63u - index] = static_cast<std::uint8_t>(bit_length >> (index * 8u));
        }
        transform(block_.data());

        std::array<std::uint8_t, 32> digest{};
        for (std::size_t word = 0; word < state_.size(); ++word) {
            for (std::size_t byte = 0; byte < 4; ++byte) {
                digest[word * 4u + byte] =
                    static_cast<std::uint8_t>(state_[word] >> ((3u - byte) * 8u));
            }
        }
        return digest;
    }

private:
    static constexpr std::array<std::uint32_t, 64> kRoundConstants{
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
        0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
        0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
        0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };

    [[nodiscard]] static std::uint32_t rotate_right(
        const std::uint32_t value,
        const unsigned shift) noexcept {
        return (value >> shift) | (value << (32u - shift));
    }

    void transform(const std::uint8_t* block) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            const auto offset = index * 4u;
            words[index] =
                (static_cast<std::uint32_t>(block[offset]) << 24u) |
                (static_cast<std::uint32_t>(block[offset + 1u]) << 16u) |
                (static_cast<std::uint32_t>(block[offset + 2u]) << 8u) |
                static_cast<std::uint32_t>(block[offset + 3u]);
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const auto left = words[index - 15u];
            const auto right = words[index - 2u];
            const auto sigma0 = rotate_right(left, 7u) ^ rotate_right(left, 18u) ^ (left >> 3u);
            const auto sigma1 = rotate_right(right, 17u) ^ rotate_right(right, 19u) ^ (right >> 10u);
            words[index] = words[index - 16u] + sigma0 + words[index - 7u] + sigma1;
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
            const auto sum1 = rotate_right(e, 6u) ^ rotate_right(e, 11u) ^ rotate_right(e, 25u);
            const auto choose = (e & f) ^ (~e & g);
            const auto temporary1 = h + sum1 + choose + kRoundConstants[index] + words[index];
            const auto sum0 = rotate_right(a, 2u) ^ rotate_right(a, 13u) ^ rotate_right(a, 22u);
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
        0x6a09e667u,
        0xbb67ae85u,
        0x3c6ef372u,
        0xa54ff53au,
        0x510e527fu,
        0x9b05688cu,
        0x1f83d9abu,
        0x5be0cd19u,
    };
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_ = 0;
    std::uint64_t total_bytes_ = 0;
};

[[nodiscard]] bool packed_value(
    const std::vector<std::uint64_t>& words,
    const std::uint64_t row,
    const std::uint32_t column,
    const std::uint32_t rows_words,
    const std::uint64_t rows,
    const std::uint32_t columns,
    const char* label) {
    if (row >= rows || column >= columns) {
        throw std::out_of_range(std::string(label) + " index is out of range");
    }
    const auto index = checked_size(
        checked_add(
            checked_multiply(row, rows_words, "packed index"),
            column / kDatasetWordBits,
            "packed index"),
        "packed index");
    return (words[index] & (std::uint64_t{1} << (column % kDatasetWordBits))) != 0;
}

void set_packed_value(
    std::vector<std::uint64_t>& words,
    const std::uint64_t row,
    const std::uint32_t column,
    const std::uint32_t rows_words,
    const std::uint64_t rows,
    const std::uint32_t columns,
    const bool value,
    const char* label) {
    if (row >= rows || column >= columns) {
        throw std::out_of_range(std::string(label) + " index is out of range");
    }
    const auto index = checked_size(
        checked_add(
            checked_multiply(row, rows_words, "packed index"),
            column / kDatasetWordBits,
            "packed index"),
        "packed index");
    const auto mask = std::uint64_t{1} << (column % kDatasetWordBits);
    if (value) {
        words[index] |= mask;
    } else {
        words[index] &= ~mask;
    }
}

void validate_padding(
    const std::vector<std::uint64_t>& words,
    const std::uint64_t rows,
    const std::uint32_t columns,
    const std::uint32_t stride,
    const char* label) {
    const auto valid_bits = columns % kDatasetWordBits;
    if (valid_bits == 0 || rows == 0) {
        return;
    }
    const auto padding_mask = ~((std::uint64_t{1} << valid_bits) - 1u);
    for (std::uint64_t row = 0; row < rows; ++row) {
        const auto index = checked_size(row * stride + stride - 1u, label);
        if ((words[index] & padding_mask) != 0) {
            throw format_error(std::string("non-zero padding bits in ") + label);
        }
    }
}

[[nodiscard]] NativeDatasetHeader parse_header(
    const std::array<std::uint8_t, kDatasetHeaderBytes>& bytes) {
    if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
        throw format_error("bad magic");
    }
    const auto version = decode_little_endian<std::uint32_t>(&bytes[8]);
    if (version != kLegacyVersion && version != kVersion) {
        throw format_error("unsupported version");
    }
    if (decode_little_endian<std::uint32_t>(&bytes[12]) != kDatasetHeaderBytes) {
        throw format_error("unsupported header size");
    }
    if (decode_little_endian<std::uint32_t>(&bytes[16]) != kDatasetWordBits) {
        throw format_error("unsupported word size");
    }
    if (decode_little_endian<std::uint32_t>(&bytes[20]) != kFlags) {
        throw format_error("unsupported flags");
    }
    NativeDatasetHeader header;
    header.frame_count = decode_little_endian<std::uint64_t>(&bytes[24]);
    header.feature_count = decode_little_endian<std::uint32_t>(&bytes[32]);
    header.feature_words_per_row = decode_little_endian<std::uint32_t>(&bytes[36]);
    header.note_count = decode_little_endian<std::uint32_t>(&bytes[40]);
    header.label_words_per_row = decode_little_endian<std::uint32_t>(&bytes[44]);
    header.midi_min = decode_i32(&bytes[48]);
    header.midi_max = decode_i32(&bytes[52]);
    header.sample_rate = decode_little_endian<std::uint32_t>(&bytes[56]);
    header.hop_size = decode_little_endian<std::uint32_t>(&bytes[60]);
    header.onset_index_count = decode_little_endian<std::uint64_t>(&bytes[64]);
    header.features_offset = decode_little_endian<std::uint64_t>(&bytes[72]);
    header.features_bytes = decode_little_endian<std::uint64_t>(&bytes[80]);
    header.activity_offset = decode_little_endian<std::uint64_t>(&bytes[88]);
    header.activity_bytes = decode_little_endian<std::uint64_t>(&bytes[96]);
    header.onset_offset = decode_little_endian<std::uint64_t>(&bytes[104]);
    header.onset_bytes = decode_little_endian<std::uint64_t>(&bytes[112]);
    header.onset_indices_offset = decode_little_endian<std::uint64_t>(&bytes[120]);
    header.onset_indices_bytes = decode_little_endian<std::uint64_t>(&bytes[128]);
    header.seed = decode_little_endian<std::uint64_t>(&bytes[136]);
    std::copy_n(bytes.begin() + 144, header.payload_sha256.size(), header.payload_sha256.begin());
    if (version == kLegacyVersion) {
        if (!std::all_of(
                bytes.begin() + static_cast<std::ptrdiff_t>(kFeatureFingerprintOffset),
                bytes.end(),
                [](const auto value) { return value == 0U; })) {
            throw format_error("legacy reserved header bytes are non-zero");
        }
    } else {
        std::copy_n(
            bytes.begin() + static_cast<std::ptrdiff_t>(kFeatureFingerprintOffset),
            kFeatureFingerprintBytes,
            header.feature_fingerprint_sha256.begin());
        if (std::all_of(
                header.feature_fingerprint_sha256.begin(),
                header.feature_fingerprint_sha256.end(),
                [](const auto value) { return value == 0U; })) {
            throw format_error("v2 feature fingerprint is zero");
        }
        if (!std::all_of(
                bytes.begin() + static_cast<std::ptrdiff_t>(
                    kFeatureFingerprintOffset + kFeatureFingerprintBytes),
                bytes.end(),
                [](const auto value) { return value == 0U; })) {
            throw format_error("v2 reserved header bytes are non-zero");
        }
    }
    return header;
}

[[nodiscard]] std::array<std::uint8_t, kDatasetHeaderBytes> make_header(
    const NativeDatasetHeader& header) {
    std::array<std::uint8_t, kDatasetHeaderBytes> bytes{};
    std::copy(kMagic.begin(), kMagic.end(), bytes.begin());
    const auto has_feature_fingerprint = std::any_of(
        header.feature_fingerprint_sha256.begin(),
        header.feature_fingerprint_sha256.end(),
        [](const auto value) { return value != 0U; });
    encode_little_endian<std::uint32_t>(
        &bytes[8], has_feature_fingerprint ? kVersion : kLegacyVersion);
    encode_little_endian<std::uint32_t>(&bytes[12], kDatasetHeaderBytes);
    encode_little_endian<std::uint32_t>(&bytes[16], kDatasetWordBits);
    encode_little_endian<std::uint32_t>(&bytes[20], kFlags);
    encode_little_endian<std::uint64_t>(&bytes[24], header.frame_count);
    encode_little_endian<std::uint32_t>(&bytes[32], header.feature_count);
    encode_little_endian<std::uint32_t>(&bytes[36], header.feature_words_per_row);
    encode_little_endian<std::uint32_t>(&bytes[40], header.note_count);
    encode_little_endian<std::uint32_t>(&bytes[44], header.label_words_per_row);
    encode_i32(&bytes[48], header.midi_min);
    encode_i32(&bytes[52], header.midi_max);
    encode_little_endian<std::uint32_t>(&bytes[56], header.sample_rate);
    encode_little_endian<std::uint32_t>(&bytes[60], header.hop_size);
    encode_little_endian<std::uint64_t>(&bytes[64], header.onset_index_count);
    encode_little_endian<std::uint64_t>(&bytes[72], header.features_offset);
    encode_little_endian<std::uint64_t>(&bytes[80], header.features_bytes);
    encode_little_endian<std::uint64_t>(&bytes[88], header.activity_offset);
    encode_little_endian<std::uint64_t>(&bytes[96], header.activity_bytes);
    encode_little_endian<std::uint64_t>(&bytes[104], header.onset_offset);
    encode_little_endian<std::uint64_t>(&bytes[112], header.onset_bytes);
    encode_little_endian<std::uint64_t>(&bytes[120], header.onset_indices_offset);
    encode_little_endian<std::uint64_t>(&bytes[128], header.onset_indices_bytes);
    encode_little_endian<std::uint64_t>(&bytes[136], header.seed);
    std::copy(header.payload_sha256.begin(), header.payload_sha256.end(), bytes.begin() + 144);
    if (has_feature_fingerprint) {
        std::copy(
            header.feature_fingerprint_sha256.begin(),
            header.feature_fingerprint_sha256.end(),
            bytes.begin() + static_cast<std::ptrdiff_t>(kFeatureFingerprintOffset));
    }
    return bytes;
}

[[nodiscard]] NativeDatasetHeader normalized_header(const NativeDataset& dataset) {
    auto header = dataset.header;
    header.feature_words_per_row = word_count(header.feature_count);
    header.label_words_per_row = word_count(header.note_count);
    header.midi_max = header.midi_min + static_cast<std::int32_t>(header.note_count) - 1;
    header.onset_index_count = static_cast<std::uint64_t>(dataset.onset_indices.size());
    header.features_offset = kDatasetHeaderBytes;
    header.features_bytes = checked_multiply(dataset.feature_words.size(), sizeof(std::uint64_t), "features bytes");
    header.activity_offset = checked_add(header.features_offset, header.features_bytes, "activity offset");
    header.activity_bytes = checked_multiply(dataset.activity_words.size(), sizeof(std::uint64_t), "activity bytes");
    header.onset_offset = checked_add(header.activity_offset, header.activity_bytes, "onset offset");
    header.onset_bytes = checked_multiply(dataset.onset_words.size(), sizeof(std::uint64_t), "onset bytes");
    header.onset_indices_offset = checked_add(header.onset_offset, header.onset_bytes, "index offset");
    header.onset_indices_bytes = checked_multiply(dataset.onset_indices.size(), sizeof(std::uint32_t), "index bytes");
    header.payload_sha256 = calculate_payload_sha256(dataset);
    return header;
}

}  // namespace

bool NativeDataset::feature(const std::uint64_t frame, const std::uint32_t column) const {
    return packed_value(
        feature_words,
        frame,
        column,
        header.feature_words_per_row,
        header.frame_count,
        header.feature_count,
        "feature");
}

bool NativeDataset::activity(const std::uint64_t frame, const std::uint32_t note) const {
    return packed_value(
        activity_words,
        frame,
        note,
        header.label_words_per_row,
        header.frame_count,
        header.note_count,
        "activity");
}

bool NativeDataset::onset(const std::uint64_t frame, const std::uint32_t note) const {
    return packed_value(
        onset_words,
        frame,
        note,
        header.label_words_per_row,
        header.frame_count,
        header.note_count,
        "onset");
}

void NativeDataset::set_feature(
    const std::uint64_t frame,
    const std::uint32_t column,
    const bool value) {
    set_packed_value(
        feature_words,
        frame,
        column,
        header.feature_words_per_row,
        header.frame_count,
        header.feature_count,
        value,
        "feature");
}

void NativeDataset::set_activity(
    const std::uint64_t frame,
    const std::uint32_t note,
    const bool value) {
    set_packed_value(
        activity_words,
        frame,
        note,
        header.label_words_per_row,
        header.frame_count,
        header.note_count,
        value,
        "activity");
}

void NativeDataset::set_onset(
    const std::uint64_t frame,
    const std::uint32_t note,
    const bool value) {
    set_packed_value(
        onset_words,
        frame,
        note,
        header.label_words_per_row,
        header.frame_count,
        header.note_count,
        value,
        "onset");
}

void NativeDataset::validate() const {
    if (header.feature_words_per_row != word_count(header.feature_count)) {
        throw format_error("feature word stride disagrees with feature count");
    }
    if (header.label_words_per_row != word_count(header.note_count)) {
        throw format_error("label word stride disagrees with note count");
    }
    if (header.note_count > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        static_cast<std::int64_t>(header.midi_min) + header.note_count - 1 != header.midi_max) {
        throw format_error("MIDI range disagrees with note count");
    }
    if (header.sample_rate == 0 || header.hop_size == 0) {
        throw format_error("audio grid must have non-zero sample rate and hop size");
    }

    const auto expected_features = checked_multiply(
        header.frame_count, header.feature_words_per_row, "feature words");
    const auto expected_labels = checked_multiply(
        header.frame_count, header.label_words_per_row, "label words");
    if (feature_words.size() != checked_size(expected_features, "feature words")) {
        throw format_error("feature payload length disagrees with dimensions");
    }
    if (activity_words.size() != checked_size(expected_labels, "activity words") ||
        onset_words.size() != checked_size(expected_labels, "onset words")) {
        throw format_error("label payload length disagrees with dimensions");
    }
    if (onset_indices.size() != checked_size(header.onset_index_count, "onset indices")) {
        throw format_error("onset index count disagrees with payload");
    }
    for (const auto index : onset_indices) {
        if (index >= header.frame_count) {
            throw format_error("onset training index is outside feature matrix");
        }
    }
    validate_padding(
        feature_words,
        header.frame_count,
        header.feature_count,
        header.feature_words_per_row,
        "features");
    validate_padding(
        activity_words,
        header.frame_count,
        header.note_count,
        header.label_words_per_row,
        "activity labels");
    validate_padding(
        onset_words,
        header.frame_count,
        header.note_count,
        header.label_words_per_row,
        "onset labels");
}

std::array<std::uint8_t, 32> calculate_payload_sha256(const NativeDataset& dataset) {
    dataset.validate();
    Sha256 digest;
    digest.update(dataset.feature_words.data(), dataset.feature_words.size() * sizeof(std::uint64_t));
    digest.update(dataset.activity_words.data(), dataset.activity_words.size() * sizeof(std::uint64_t));
    digest.update(dataset.onset_words.data(), dataset.onset_words.size() * sizeof(std::uint64_t));
    digest.update(dataset.onset_indices.data(), dataset.onset_indices.size() * sizeof(std::uint32_t));
    return digest.finish();
}

std::string sha256_hex(const std::array<std::uint8_t, 32>& digest) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const auto byte : digest) {
        stream << std::setw(2) << static_cast<unsigned>(byte);
    }
    return stream.str();
}

NativeDataset load_dataset(const std::filesystem::path& path, const bool verify_checksum) {
    if (!host_is_little_endian()) {
        throw std::runtime_error("TMGMDAT direct loader currently requires a little-endian host");
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open TMGMDAT dataset: " + path.string());
    }

    std::array<std::uint8_t, kDatasetHeaderBytes> header_bytes{};
    read_exact(stream, header_bytes.data(), header_bytes.size());
    NativeDataset dataset;
    dataset.header = parse_header(header_bytes);
    const auto& header = dataset.header;

    const auto expected_feature_words = checked_multiply(
        header.frame_count, header.feature_words_per_row, "feature words");
    const auto expected_label_words = checked_multiply(
        header.frame_count, header.label_words_per_row, "label words");
    const auto expected_feature_bytes = checked_multiply(
        expected_feature_words, sizeof(std::uint64_t), "feature bytes");
    const auto expected_label_bytes = checked_multiply(
        expected_label_words, sizeof(std::uint64_t), "label bytes");
    const auto expected_index_bytes = checked_multiply(
        header.onset_index_count, sizeof(std::uint32_t), "index bytes");
    if (header.features_bytes != expected_feature_bytes ||
        header.activity_bytes != expected_label_bytes ||
        header.onset_bytes != expected_label_bytes ||
        header.onset_indices_bytes != expected_index_bytes) {
        throw format_error("payload size disagrees with dimensions");
    }

    std::uint64_t expected_offset = kDatasetHeaderBytes;
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 4> sections{{
        {header.features_offset, header.features_bytes},
        {header.activity_offset, header.activity_bytes},
        {header.onset_offset, header.onset_bytes},
        {header.onset_indices_offset, header.onset_indices_bytes},
    }};
    for (const auto& section : sections) {
        if (section.first != expected_offset) {
            throw format_error("payload sections are not contiguous");
        }
        expected_offset = checked_add(expected_offset, section.second, "payload layout");
    }

    dataset.feature_words.resize(checked_size(expected_feature_words, "feature words"));
    dataset.activity_words.resize(checked_size(expected_label_words, "activity words"));
    dataset.onset_words.resize(checked_size(expected_label_words, "onset words"));
    dataset.onset_indices.resize(checked_size(header.onset_index_count, "onset indices"));
    read_exact(stream, dataset.feature_words.data(), checked_size(header.features_bytes, "features bytes"));
    read_exact(stream, dataset.activity_words.data(), checked_size(header.activity_bytes, "activity bytes"));
    read_exact(stream, dataset.onset_words.data(), checked_size(header.onset_bytes, "onset bytes"));
    read_exact(stream, dataset.onset_indices.data(), checked_size(header.onset_indices_bytes, "index bytes"));
    if (stream.peek() != std::ifstream::traits_type::eof()) {
        throw format_error("trailing bytes after payload");
    }

    dataset.validate();
    if (verify_checksum && calculate_payload_sha256(dataset) != header.payload_sha256) {
        throw format_error("payload SHA-256 mismatch");
    }
    return dataset;
}

void save_dataset(const std::filesystem::path& path, const NativeDataset& dataset) {
    if (!host_is_little_endian()) {
        throw std::runtime_error("TMGMDAT direct writer currently requires a little-endian host");
    }
    dataset.validate();
    const auto header = normalized_header(dataset);
    const auto header_bytes = make_header(header);

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot create TMGMDAT dataset: " + path.string());
    }
    write_exact(stream, header_bytes.data(), header_bytes.size());
    write_exact(stream, dataset.feature_words.data(), dataset.feature_words.size() * sizeof(std::uint64_t));
    write_exact(stream, dataset.activity_words.data(), dataset.activity_words.size() * sizeof(std::uint64_t));
    write_exact(stream, dataset.onset_words.data(), dataset.onset_words.size() * sizeof(std::uint64_t));
    write_exact(stream, dataset.onset_indices.data(), dataset.onset_indices.size() * sizeof(std::uint32_t));
}

}  // namespace tmgm::native
