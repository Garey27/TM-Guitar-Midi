#include "tmgm/strict_cap16_v3_frontend.hpp"

#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tmgm::native {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'T', 'M', 'G', 'M', 'F', 'R', 'T', 0};
constexpr std::uint32_t kFormatVersion = 2U;
constexpr std::uint32_t kHeaderBytes = 512U;
constexpr std::uint32_t kDescriptorBytes = 256U;
constexpr std::size_t kChecksumOffset = 376U;
constexpr std::size_t kChecksumBytes = 32U;
constexpr std::size_t kBankCount = kStrictCap16V3PackedRowCount;
constexpr std::size_t kNoteCount = kStrictCap16V3OutputCount;
constexpr std::size_t kHarmonics = 6U;
constexpr std::size_t kSpectrumBins =
    kStrictCap16V3FrontendFftSize / 2U + 1U;
constexpr std::size_t kMaxContextFrames = 33U;
constexpr std::uint32_t kThermometerEqualityUlps = 4U;
constexpr std::array<std::uint32_t, 7> kContextDelays{
    0U, 1U, 2U, 4U, 8U, 16U, 32U};
constexpr char kPocketFftCommit[] =
    "33ae5dc94c9cdc7f1c78346504a85de87cadaa12";
constexpr char kExpectedArtifactChecksum[] =
    "e6fa338c285ee6286f2917b16a35bcb18fbce6c025e8b2a58a2281ca9fcbbb61";

struct ExpectedBank {
    const char* identifier;
    std::uint32_t variant;
    std::uint32_t spectral_width;
    std::uint32_t continuous_width;
    std::uint32_t binary_width;
    std::uint32_t flags;
    float contrast_offset;
    const char* binarizer_sha256;
    const char* binarizer_signature;
    const char* semantic_sha256;
    const char* reference_sha256;
};

constexpr std::array<ExpectedBank, kBankCount> kExpectedBanks{{
    {
        "plain", 1U, 297U, 2079U, 7973U, 0U, 0.5F,
        "0681d84e265cb46acfef2b9ba5fa41cf4a9e381396855819e6f69db709544c1f",
        "d5dfa368eb42996bd042fb3605b1c6a44c7de63d6d9fb68cf1b020e4e873e9e0",
        "8cf43d89312dc5a481280a017a0f23f2b3ee4f3a573ebf992fb78edb2a305fc1",
        "e85bd175c09d0135d9204e5703f245e86eee4a643112a7b043e4d65183f190c3",
    },
    {
        "hcontrast", 2U, 297U, 2079U, 7973U, 1U, 1.5F,
        "c8e5a1a95f2af9757d6e34aff1a9f4203b3606d223c7300bb8cf7b03d4735b40",
        "2a300d75519251e2c7311eb1935b827e9a091c17558330639034d5437af53006",
        "bcfbbdd59dcbf30ceb8b184b61edfc768de031131a72e90b346d460ac3adb8e1",
        "fde35e458c2a9a7794fef243b9fb65e11d20fcc56fd9e28a2b3c9472dcb1536f",
    },
    {
        "hprofile", 3U, 591U, 4137U, 16205U, 3U, 1.5F,
        "a3e33df3adfc4cdd8ede92fd28d05d6fb79611b3e0b01eb8a0cbba1b082babe1",
        "01ee303abdcbea0f13d617cd69bb32972b2474e0c58780aea1df9103af6ddbc0",
        "4ce497eb7ed6fec1b8d292ef65ad28b38115045e985c4c2634bc0219a61a0ddd",
        "a101928dcc916326611f6d9d68a5ea95a4d47318657f2bb172298bfe92d8709d",
    },
    {
        "cattack", 4U, 395U, 2765U, 10374U, 5U, 1.5F,
        "2084e673ecb1b84e76047895898d25cac7f4bd6cfa9acdbed6451a9101f941c1",
        "93b13d0de76f92183f0de69224a4123283fc77e1c4e8c86caa87319a392eb649",
        "59d6e0de04f6121f427564d8d39c11bb5c7ba1731f43ef8b02e34eff59bf8b2c",
        "bd2554f74d0994605d8518b2bbc62941949f3f126ac586f34316dceeadff0b67",
    },
}};

[[nodiscard]] std::runtime_error artifact_error(const std::string& message) {
    return std::runtime_error(
        "invalid strict-cap16-v3 TMGMFRT artifact: " + message);
}

template <typename T>
[[nodiscard]] T decode_unsigned(const std::uint8_t* bytes) noexcept {
    static_assert(std::is_unsigned_v<T>);
    T value = 0;
    for (std::size_t index = 0U; index < sizeof(T); ++index) {
        value |= static_cast<T>(bytes[index]) << (index * 8U);
    }
    return value;
}

[[nodiscard]] std::int32_t decode_i32(const std::uint8_t* bytes) noexcept {
    const auto bits = decode_unsigned<std::uint32_t>(bytes);
    std::int32_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

[[nodiscard]] float decode_f32(const std::uint8_t* bytes) noexcept {
    const auto bits = decode_unsigned<std::uint32_t>(bytes);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

[[nodiscard]] double decode_f64(const std::uint8_t* bytes) noexcept {
    const auto bits = decode_unsigned<std::uint64_t>(bytes);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

[[nodiscard]] bool same_float_bits(
    const float left,
    const float right) noexcept {
    std::uint32_t left_bits = 0U;
    std::uint32_t right_bits = 0U;
    std::memcpy(&left_bits, &left, sizeof(left));
    std::memcpy(&right_bits, &right, sizeof(right));
    return left_bits == right_bits;
}

[[nodiscard]] std::array<std::uint8_t, 32> parse_sha256(
    const char* text) {
    if (text == nullptr || std::strlen(text) != 64U) {
        throw artifact_error("compiled SHA-256 is malformed");
    }
    const auto nibble = [](const char value) -> std::uint8_t {
        if (value >= '0' && value <= '9') {
            return static_cast<std::uint8_t>(value - '0');
        }
        if (value >= 'a' && value <= 'f') {
            return static_cast<std::uint8_t>(value - 'a' + 10);
        }
        if (value >= 'A' && value <= 'F') {
            return static_cast<std::uint8_t>(value - 'A' + 10);
        }
        throw artifact_error("compiled SHA-256 contains non-hex digits");
    };
    std::array<std::uint8_t, 32> digest{};
    for (std::size_t index = 0U; index < digest.size(); ++index) {
        digest[index] = static_cast<std::uint8_t>(
            (nibble(text[index * 2U]) << 4U) |
            nibble(text[index * 2U + 1U]));
    }
    return digest;
}

class Sha256 {
public:
    void update(const void* source, std::size_t size) noexcept {
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

    [[nodiscard]] std::array<std::uint8_t, 32> finish() noexcept {
        const auto bit_length = total_bytes_ * 8U;
        block_[block_size_++] = 0x80U;
        if (block_size_ > 56U) {
            std::fill(
                block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                block_.end(), std::uint8_t{0U});
            transform(block_.data());
            block_size_ = 0U;
        }
        std::fill(
            block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
            block_.begin() + 56, std::uint8_t{0U});
        for (std::size_t index = 0U; index < 8U; ++index) {
            block_[63U - index] =
                static_cast<std::uint8_t>(bit_length >> (index * 8U));
        }
        transform(block_.data());
        std::array<std::uint8_t, 32> digest{};
        for (std::size_t word = 0U; word < state_.size(); ++word) {
            for (std::size_t byte = 0U; byte < 4U; ++byte) {
                digest[word * 4U + byte] = static_cast<std::uint8_t>(
                    state_[word] >> ((3U - byte) * 8U));
            }
        }
        return digest;
    }

private:
    [[nodiscard]] static std::uint32_t rotate_right(
        const std::uint32_t value,
        const unsigned shift) noexcept {
        return (value >> shift) | (value << (32U - shift));
    }

    void transform(const std::uint8_t* block) noexcept {
        static constexpr std::array<std::uint32_t, 64> constants{
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0U; index < 16U; ++index) {
            const auto offset = index * 4U;
            words[index] =
                (static_cast<std::uint32_t>(block[offset]) << 24U) |
                (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
                (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
                static_cast<std::uint32_t>(block[offset + 3U]);
        }
        for (std::size_t index = 16U; index < words.size(); ++index) {
            const auto left = words[index - 15U];
            const auto right = words[index - 2U];
            const auto sigma0 = rotate_right(left, 7U) ^
                rotate_right(left, 18U) ^ (left >> 3U);
            const auto sigma1 = rotate_right(right, 17U) ^
                rotate_right(right, 19U) ^ (right >> 10U);
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
        for (std::size_t index = 0U; index < words.size(); ++index) {
            const auto sum1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                rotate_right(e, 25U);
            const auto choose = (e & f) ^ (~e & g);
            const auto temporary1 =
                h + sum1 + choose + constants[index] + words[index];
            const auto sum0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                rotate_right(a, 22U);
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
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_ = 0U;
    std::uint64_t total_bytes_ = 0U;
};

[[nodiscard]] std::array<std::uint8_t, 32> sha256(
    const void* bytes,
    const std::size_t size) noexcept {
    Sha256 hash;
    hash.update(bytes, size);
    return hash.finish();
}

[[nodiscard]] std::vector<std::uint8_t> read_file(
    const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error(
            "failed to open strict-cap16-v3 frontend: " + path.string());
    }
    const auto end = stream.tellg();
    if (end < 0) {
        throw std::runtime_error("failed to determine frontend artifact size");
    }
    const auto size = static_cast<std::uint64_t>(end);
    if (size > static_cast<std::uint64_t>(
                   std::numeric_limits<std::size_t>::max())) {
        throw artifact_error("file does not fit address space");
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    stream.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        stream.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    }
    if (!stream || static_cast<std::size_t>(stream.gcount()) != bytes.size()) {
        throw std::runtime_error("failed to read complete frontend artifact");
    }
    return bytes;
}

void require_range(
    const std::uint64_t offset,
    const std::uint64_t size,
    const std::size_t file_size,
    const char* label) {
    if (offset > file_size || size > file_size - offset) {
        throw artifact_error(std::string(label) + " lies outside the file");
    }
}

[[nodiscard]] bool all_zero(
    const std::vector<std::uint8_t>& bytes,
    const std::size_t begin,
    const std::size_t end) {
    return std::all_of(
        bytes.begin() + static_cast<std::ptrdiff_t>(begin),
        bytes.begin() + static_cast<std::ptrdiff_t>(end),
        [](const std::uint8_t value) { return value == 0U; });
}

[[nodiscard]] std::string read_identifier(const std::uint8_t* bytes) {
    std::size_t size = 0U;
    while (size < 32U && bytes[size] != 0U) {
        ++size;
    }
    if (size == 0U || size == 32U ||
        !std::all_of(bytes + size, bytes + 32U,
                     [](const std::uint8_t value) { return value == 0U; })) {
        throw artifact_error("bank identifier/padding is invalid");
    }
    return std::string(
        reinterpret_cast<const char*>(bytes),
        reinterpret_cast<const char*>(bytes + size));
}

struct ThermometerEntry {
    std::uint32_t raw_column = 0U;
    float threshold = 0.0F;
    std::uint32_t equality_ulps = 0U;
};

struct LoadedBank {
    std::uint32_t variant = 0U;
    std::uint32_t spectral_width = 0U;
    std::uint32_t continuous_width = 0U;
    std::uint32_t binary_width = 0U;
    std::vector<ThermometerEntry> entries;
};

struct LoadedArtifact {
    std::vector<float> window;
    std::vector<double> harmonic_bins;
    std::vector<float> harmonic_weights;
    std::vector<double> plain_side_low_bins;
    std::vector<double> plain_side_high_bins;
    std::vector<double> corrected_side_low_bins;
    std::vector<double> corrected_side_high_bins;
    std::vector<double> subharmonic_bins;
    std::vector<float> frequency_axis;
    float magnitude_normalizer = 1.0F;
    std::array<LoadedBank, kBankCount> banks;
};

template <typename T, typename Decode>
[[nodiscard]] std::vector<T> load_array(
    const std::vector<std::uint8_t>& bytes,
    const std::uint64_t offset,
    const std::uint64_t section_bytes,
    const std::size_t expected_count,
    Decode decode,
    const char* label) {
    if (section_bytes != expected_count * sizeof(T)) {
        throw artifact_error(std::string(label) + " has wrong size");
    }
    require_range(offset, section_bytes, bytes.size(), label);
    std::vector<T> values(expected_count);
    const auto* source = bytes.data() + static_cast<std::size_t>(offset);
    for (std::size_t index = 0U; index < expected_count; ++index) {
        values[index] = decode(source + index * sizeof(T));
    }
    return values;
}

[[nodiscard]] LoadedArtifact load_artifact(
    const std::filesystem::path& path) {
    auto bytes = read_file(path);
    if (bytes.size() < kHeaderBytes ||
        !std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
        throw artifact_error("magic/header is invalid");
    }
    if (decode_unsigned<std::uint32_t>(bytes.data() + 8U) != kFormatVersion ||
        decode_unsigned<std::uint32_t>(bytes.data() + 12U) != kHeaderBytes ||
        decode_unsigned<std::uint32_t>(bytes.data() + 16U) != kDescriptorBytes ||
        decode_unsigned<std::uint32_t>(bytes.data() + 20U) != kBankCount) {
        throw artifact_error("format geometry differs");
    }
    if (decode_unsigned<std::uint32_t>(bytes.data() + 24U) !=
            kStrictCap16V3FrontendSampleRate ||
        decode_unsigned<std::uint32_t>(bytes.data() + 28U) !=
            kStrictCap16V3FrontendHopSize ||
        decode_unsigned<std::uint32_t>(bytes.data() + 32U) !=
            kStrictCap16V3FrontendFftSize ||
        decode_unsigned<std::uint32_t>(bytes.data() + 36U) != 1U ||
        decode_i32(bytes.data() + 40U) != 40 ||
        decode_i32(bytes.data() + 44U) != 88 ||
        decode_i32(bytes.data() + 48U) != 6 ||
        decode_unsigned<std::uint32_t>(bytes.data() + 52U) != 4U ||
        !same_float_bits(decode_f32(bytes.data() + 56U), 0.08F) ||
        decode_unsigned<std::uint32_t>(bytes.data() + 60U) != 7U ||
        decode_unsigned<std::uint32_t>(bytes.data() + 64U) != 32U ||
        decode_unsigned<std::uint32_t>(bytes.data() + 68U) !=
            kThermometerEqualityUlps ||
        decode_unsigned<std::uint32_t>(bytes.data() + 72U) != 0U) {
        throw artifact_error("frontend/timebase configuration differs");
    }
    const auto descriptors_offset =
        decode_unsigned<std::uint64_t>(bytes.data() + 80U);
    const auto descriptors_bytes =
        decode_unsigned<std::uint64_t>(bytes.data() + 88U);
    const auto payload_offset =
        decode_unsigned<std::uint64_t>(bytes.data() + 96U);
    const auto payload_bytes =
        decode_unsigned<std::uint64_t>(bytes.data() + 104U);
    const auto declared_file_bytes =
        decode_unsigned<std::uint64_t>(bytes.data() + 112U);
    if (descriptors_offset != kHeaderBytes ||
        descriptors_bytes != kBankCount * kDescriptorBytes ||
        payload_offset != kHeaderBytes + descriptors_bytes ||
        payload_bytes != bytes.size() - payload_offset ||
        declared_file_bytes != bytes.size()) {
        throw artifact_error("descriptor/payload layout differs");
    }
    require_range(descriptors_offset, descriptors_bytes, bytes.size(),
                  "bank descriptors");
    if (!all_zero(bytes, 76U, 80U) ||
        !all_zero(bytes, 352U, kChecksumOffset) ||
        !all_zero(bytes, kChecksumOffset + kChecksumBytes, kHeaderBytes)) {
        throw artifact_error("header reserved bytes are non-zero");
    }
    if (std::string(
            reinterpret_cast<const char*>(bytes.data() + 312U), 40U) !=
        kPocketFftCommit) {
        throw artifact_error("PocketFFT source commit differs");
    }
    constexpr std::array<float, 4> expected_quantiles{
        0.5F, 0.7F, 0.85F, 0.95F};
    for (std::size_t index = 0U; index < expected_quantiles.size(); ++index) {
        if (!same_float_bits(
                decode_f32(bytes.data() + 268U + index * 4U),
                expected_quantiles[index])) {
            throw artifact_error("quantile order/value differs");
        }
    }
    for (std::size_t index = 0U; index < kContextDelays.size(); ++index) {
        if (decode_unsigned<std::uint32_t>(
                bytes.data() + 284U + index * 4U) != kContextDelays[index]) {
            throw artifact_error("context delay order differs");
        }
    }
    std::array<std::uint8_t, 32> stored_checksum{};
    std::copy_n(
        bytes.begin() + static_cast<std::ptrdiff_t>(kChecksumOffset),
        stored_checksum.size(), stored_checksum.begin());
    if (stored_checksum != parse_sha256(kExpectedArtifactChecksum)) {
        throw artifact_error("artifact checksum is not the frozen checksum");
    }
    auto canonical = bytes;
    std::fill(
        canonical.begin() + static_cast<std::ptrdiff_t>(kChecksumOffset),
        canonical.begin() + static_cast<std::ptrdiff_t>(
            kChecksumOffset + kChecksumBytes),
        std::uint8_t{0U});
    if (sha256(canonical.data(), canonical.size()) != stored_checksum) {
        throw artifact_error("artifact SHA-256 mismatch");
    }

    std::array<std::pair<std::uint64_t, std::uint64_t>, 9> sections{};
    for (std::size_t index = 0U; index < sections.size(); ++index) {
        sections[index] = {
            decode_unsigned<std::uint64_t>(
                bytes.data() + 120U + index * 16U),
            decode_unsigned<std::uint64_t>(
                bytes.data() + 128U + index * 16U),
        };
    }
    LoadedArtifact result;
    result.window = load_array<float>(
        bytes, sections[0].first, sections[0].second,
        kStrictCap16V3FrontendFftSize, decode_f32, "window");
    result.harmonic_bins = load_array<double>(
        bytes, sections[1].first, sections[1].second,
        kNoteCount * kHarmonics, decode_f64, "harmonic bins");
    result.harmonic_weights = load_array<float>(
        bytes, sections[2].first, sections[2].second,
        kHarmonics, decode_f32, "harmonic weights");
    result.plain_side_low_bins = load_array<double>(
        bytes, sections[3].first, sections[3].second,
        kNoteCount, decode_f64, "plain low-side bins");
    result.plain_side_high_bins = load_array<double>(
        bytes, sections[4].first, sections[4].second,
        kNoteCount, decode_f64, "plain high-side bins");
    result.corrected_side_low_bins = load_array<double>(
        bytes, sections[5].first, sections[5].second,
        kNoteCount * kHarmonics, decode_f64, "corrected low-side bins");
    result.corrected_side_high_bins = load_array<double>(
        bytes, sections[6].first, sections[6].second,
        kNoteCount * kHarmonics, decode_f64, "corrected high-side bins");
    result.subharmonic_bins = load_array<double>(
        bytes, sections[7].first, sections[7].second,
        kNoteCount, decode_f64, "subharmonic bins");
    result.frequency_axis = load_array<float>(
        bytes, sections[8].first, sections[8].second,
        kSpectrumBins, decode_f32, "frequency axis");
    result.magnitude_normalizer = decode_f32(bytes.data() + 264U);
    if (!std::isfinite(result.magnitude_normalizer) ||
        result.magnitude_normalizer <= 0.0F) {
        throw artifact_error("magnitude normalizer is invalid");
    }

    for (std::size_t bank_index = 0U;
         bank_index < kBankCount; ++bank_index) {
        const auto* descriptor = bytes.data() + descriptors_offset +
            bank_index * kDescriptorBytes;
        const auto& expected = kExpectedBanks[bank_index];
        if (read_identifier(descriptor) != expected.identifier ||
            decode_unsigned<std::uint32_t>(descriptor + 32U) != expected.variant ||
            decode_unsigned<std::uint32_t>(descriptor + 36U) !=
                expected.spectral_width ||
            decode_unsigned<std::uint32_t>(descriptor + 40U) !=
                expected.continuous_width ||
            decode_unsigned<std::uint32_t>(descriptor + 44U) !=
                expected.continuous_width * 4U ||
            decode_unsigned<std::uint32_t>(descriptor + 48U) !=
                expected.binary_width ||
            !same_float_bits(decode_f32(descriptor + 52U),
                             expected.contrast_offset) ||
            decode_unsigned<std::uint32_t>(descriptor + 56U) !=
                expected.flags) {
            throw artifact_error(
                std::string("bank contract differs: ") + expected.identifier);
        }
        const auto require_digest = [&](const std::size_t offset,
                                        const char* expected_hex,
                                        const char* label) {
            const auto expected_digest = parse_sha256(expected_hex);
            if (!std::equal(
                    expected_digest.begin(), expected_digest.end(),
                    descriptor + offset)) {
                throw artifact_error(
                    std::string(expected.identifier) + " " + label +
                    " differs");
            }
        };
        require_digest(80U, expected.binarizer_sha256, "binarizer SHA-256");
        require_digest(112U, expected.binarizer_signature,
                       "binarizer signature");
        require_digest(144U, expected.semantic_sha256,
                       "semantic fingerprint");
        require_digest(176U, expected.reference_sha256,
                       "reference SHA-256");
        if (!all_zero(bytes,
                      static_cast<std::size_t>(descriptors_offset) +
                          bank_index * kDescriptorBytes + 76U,
                      static_cast<std::size_t>(descriptors_offset) +
                          bank_index * kDescriptorBytes + 80U) ||
            !all_zero(bytes,
                      static_cast<std::size_t>(descriptors_offset) +
                          bank_index * kDescriptorBytes + 208U,
                      static_cast<std::size_t>(descriptors_offset) +
                          (bank_index + 1U) * kDescriptorBytes)) {
            throw artifact_error("bank descriptor reserved bytes are non-zero");
        }
        const auto entry_offset =
            decode_unsigned<std::uint64_t>(descriptor + 60U);
        const auto entry_bytes =
            decode_unsigned<std::uint64_t>(descriptor + 68U);
        if (entry_bytes != expected.binary_width * 12ULL) {
            throw artifact_error("thermometer entry bytes differ");
        }
        require_range(entry_offset, entry_bytes, bytes.size(),
                      "thermometer entries");
        auto& bank = result.banks[bank_index];
        bank.variant = expected.variant;
        bank.spectral_width = expected.spectral_width;
        bank.continuous_width = expected.continuous_width;
        bank.binary_width = expected.binary_width;
        bank.entries.resize(expected.binary_width);
        std::uint32_t previous_raw = 0U;
        bool found_equality_record = false;
        for (std::size_t entry_index = 0U;
             entry_index < bank.entries.size(); ++entry_index) {
            const auto* entry = bytes.data() + entry_offset + entry_index * 12U;
            const auto raw = decode_unsigned<std::uint32_t>(entry);
            const auto threshold = decode_f32(entry + 4U);
            const auto equality_ulps =
                decode_unsigned<std::uint32_t>(entry + 8U);
            const bool is_expected_equality_record =
                expected.variant == 4U && entry_index == 8748U && raw == 9336U;
            if (raw >= expected.continuous_width * 4U ||
                (entry_index != 0U && raw <= previous_raw) ||
                !std::isfinite(threshold) ||
                equality_ulps != (is_expected_equality_record
                    ? kThermometerEqualityUlps : 0U)) {
                throw artifact_error("thermometer entry is invalid/noncanonical");
            }
            bank.entries[entry_index] = {raw, threshold, equality_ulps};
            found_equality_record |= is_expected_equality_record;
            previous_raw = raw;
        }
        if ((expected.variant == 4U) != found_equality_record) {
            throw artifact_error("thermometer equality record differs");
        }
    }
    return result;
}

[[nodiscard]] float clip_unit(const float value) noexcept {
    return std::max(0.0F, std::min(1.0F, value));
}

[[nodiscard]] float numpy_complex64_absolute(
    const float real,
    const float imaginary) noexcept {
    const float absolute_real = std::abs(real);
    const float absolute_imaginary = std::abs(imaginary);
    const float larger = std::max(absolute_real, absolute_imaginary);
    const float smaller = std::min(absolute_real, absolute_imaginary);
    if (larger == 0.0F) {
        return 0.0F;
    }
    if (std::isinf(smaller)) {
        return std::numeric_limits<float>::infinity();
    }
    const float ratio = smaller / larger;
    return std::sqrt(std::fma(ratio, ratio, 1.0F)) * larger;
}

[[nodiscard]] bool thermometer_compare(
    const float value,
    const float threshold,
    const std::uint32_t equality_ulps) noexcept {
    if (value >= threshold) {
        return true;
    }
    if (!(value >= 0.0F) || !(threshold >= 0.0F) ||
        !std::isfinite(value) || !std::isfinite(threshold)) {
        return false;
    }
    std::uint32_t value_bits = 0U;
    std::uint32_t threshold_bits = 0U;
    std::memcpy(&value_bits, &value, sizeof(value_bits));
    std::memcpy(&threshold_bits, &threshold, sizeof(threshold_bits));
    return threshold_bits >= value_bits &&
        threshold_bits - value_bits <= equality_ulps;
}

[[nodiscard]] float unit_db_float(const float value) noexcept {
    const float limited = std::max(value, 1.0e-7F);
    const float db = 20.0F * std::log10(limited);
    return clip_unit((db - (-100.0F)) / 100.0F);
}

[[nodiscard]] float unit_db_double(const double value) noexcept {
    const double limited = std::max(value, 1.0e-7);
    const double db = 20.0 * std::log10(limited);
    const double unit = std::max(0.0, std::min(1.0, (db - (-100.0)) / 100.0));
    return static_cast<float>(unit);
}

[[nodiscard]] float sample_bin(
    const std::array<float, kSpectrumBins>& values,
    const double position) noexcept {
    constexpr auto maximum = kSpectrumBins - 1U;
    if (position > static_cast<double>(maximum)) {
        return 0.0F;
    }
    const auto clipped = std::max(
        0.0, std::min(static_cast<double>(maximum), position));
    const auto lower = static_cast<std::size_t>(std::floor(clipped));
    const auto upper = std::min(lower + 1U, maximum);
    const auto fraction = clipped - static_cast<double>(lower);
    const auto result =
        static_cast<double>(values[lower]) * (1.0 - fraction) +
        static_cast<double>(values[upper]) * fraction;
    return static_cast<float>(result);
}

[[nodiscard]] float pairwise_sum_float(
    const float* values,
    const std::size_t count) noexcept {
    if (count < 8U) {
        float result = -0.0F;
        for (std::size_t index = 0U; index < count; ++index) {
            result += values[index];
        }
        return result;
    }
    if (count <= 128U) {
        std::array<float, 8> accumulators{
            values[0], values[1], values[2], values[3],
            values[4], values[5], values[6], values[7],
        };
        std::size_t index = 8U;
        for (; index < count - (count % 8U); index += 8U) {
            for (std::size_t lane = 0U; lane < 8U; ++lane) {
                accumulators[lane] += values[index + lane];
            }
        }
        float result =
            ((accumulators[0] + accumulators[1]) +
             (accumulators[2] + accumulators[3])) +
            ((accumulators[4] + accumulators[5]) +
             (accumulators[6] + accumulators[7]));
        for (; index < count; ++index) {
            result += values[index];
        }
        return result;
    }
    auto left_count = count / 2U;
    left_count -= left_count % 8U;
    return pairwise_sum_float(values, left_count) +
        pairwise_sum_float(values + left_count, count - left_count);
}

[[nodiscard]] double pairwise_sum_double_from_float(
    const float* values,
    const std::size_t count) noexcept {
    if (count < 8U) {
        double result = -0.0;
        for (std::size_t index = 0U; index < count; ++index) {
            result += static_cast<double>(values[index]);
        }
        return result;
    }
    if (count <= 128U) {
        std::array<double, 8> accumulators{
            values[0], values[1], values[2], values[3],
            values[4], values[5], values[6], values[7],
        };
        std::size_t index = 8U;
        for (; index < count - (count % 8U); index += 8U) {
            for (std::size_t lane = 0U; lane < 8U; ++lane) {
                accumulators[lane] += static_cast<double>(values[index + lane]);
            }
        }
        double result =
            ((accumulators[0] + accumulators[1]) +
             (accumulators[2] + accumulators[3])) +
            ((accumulators[4] + accumulators[5]) +
             (accumulators[6] + accumulators[7]));
        for (; index < count; ++index) {
            result += static_cast<double>(values[index]);
        }
        return result;
    }
    auto left_count = count / 2U;
    left_count -= left_count % 8U;
    return pairwise_sum_double_from_float(values, left_count) +
        pairwise_sum_double_from_float(
            values + left_count, count - left_count);
}

[[nodiscard]] float dot_float(
    const float* left,
    const float* right,
    const std::size_t count) noexcept {
    float sum = 0.0F;
    for (std::size_t index = 0U; index < count; ++index) {
        sum += left[index] * right[index];
    }
    return sum;
}

}  // namespace

struct StrictCap16V3StreamingFrontend::Impl {
    struct BankRuntime {
        LoadedBank definition;
        std::vector<float> history;
        std::vector<std::uint32_t> packed_words;

        explicit BankRuntime(LoadedBank bank)
            : definition(std::move(bank)),
              history(kMaxContextFrames * definition.spectral_width, 0.0F),
              packed_words(
                  (static_cast<std::size_t>(definition.binary_width) + 31U) /
                  32U,
                  0U) {}

        void reset() noexcept {
            std::fill(history.begin(), history.end(), 0.0F);
            std::fill(packed_words.begin(), packed_words.end(), 0U);
        }

        void encode(
            const float* spectral,
            const std::uint64_t frame_index) noexcept {
            const auto history_slot = static_cast<std::size_t>(
                frame_index % kMaxContextFrames);
            std::copy_n(
                spectral,
                definition.spectral_width,
                history.data() + history_slot * definition.spectral_width);
            std::fill(packed_words.begin(), packed_words.end(), 0U);
            for (std::size_t output = 0U;
                 output < definition.entries.size(); ++output) {
                const auto& entry = definition.entries[output];
                const auto continuous = entry.raw_column / 4U;
                const auto context_slot = continuous / definition.spectral_width;
                const auto feature = continuous % definition.spectral_width;
                const auto delay = kContextDelays[context_slot];
                float value = 0.0F;
                if (frame_index >= delay) {
                    const auto source_slot = static_cast<std::size_t>(
                        (frame_index - delay) % kMaxContextFrames);
                    value = history[
                        source_slot * definition.spectral_width + feature];
                }
                if (thermometer_compare(
                        value, entry.threshold, entry.equality_ulps)) {
                    packed_words[output / 32U] |=
                        std::uint32_t{1U} << (output % 32U);
                }
            }
        }
    };

    LoadedArtifact artifact;
    std::unique_ptr<pocketfft::detail::pocketfft_r<float>> fft_plan;
    std::array<BankRuntime, kBankCount> banks;
    StrictCap16V3FrameInput frame_rows;

    std::array<float, kStrictCap16V3FrontendFftSize> ring{};
    std::array<float, kStrictCap16V3FrontendFftSize> ordered_frame{};
    std::array<float, kStrictCap16V3FrontendFftSize> windowed_frame{};
    std::array<float, kStrictCap16V3FrontendFftSize> squared_frame{};
    std::array<std::complex<float>, kSpectrumBins> fft_buffer{};
    std::array<float, kStrictCap16V3FrontendFftSize> fft_scratch{};
    std::array<float, kSpectrumBins> magnitude{};
    std::array<float, kSpectrumBins> previous_magnitude{};
    std::array<float, kSpectrumBins> magnitude_flux{};
    std::array<float, kNoteCount * kHarmonics> harmonic_unit{};
    std::array<float, kNoteCount * kHarmonics> corrected_side_unit{};
    std::array<float, kNoteCount> fundamental{};
    std::array<float, kNoteCount> salience{};
    std::array<float, kNoteCount> plain_contrast{};
    std::array<float, kNoteCount> corrected_contrast{};
    std::array<float, kNoteCount> subharmonic_margin{};
    std::array<float, kNoteCount> positive_flux{};
    std::array<float, kNoteCount> fast_slow{};
    std::array<float, kNoteCount> previous_salience{};
    std::array<float, kNoteCount> ema_salience{};
    std::array<float, kNoteCount> previous_contrast{};
    std::array<float, kNoteCount> ema_contrast{};
    std::array<float, kNoteCount> contrast_positive_flux{};
    std::array<float, kNoteCount> contrast_fast_slow{};
    std::array<float, 297> plain_features{};
    std::array<float, 297> hcontrast_features{};
    std::array<float, 591> hprofile_features{};
    std::array<float, 395> cattack_features{};

    std::size_t write_index = 0U;
    std::uint64_t sample_count = 0U;
    std::uint64_t next_frame_sample = 1U;
    std::uint64_t frame_count = 0U;

    explicit Impl(LoadedArtifact loaded)
        : artifact(std::move(loaded)),
          fft_plan(std::make_unique<pocketfft::detail::pocketfft_r<float>>(
              kStrictCap16V3FrontendFftSize)),
          banks{{
              BankRuntime(std::move(artifact.banks[0])),
              BankRuntime(std::move(artifact.banks[1])),
              BankRuntime(std::move(artifact.banks[2])),
              BankRuntime(std::move(artifact.banks[3])),
          }} {
        for (std::size_t bank = 0U; bank < banks.size(); ++bank) {
            frame_rows.rows[bank] = {
                banks[bank].packed_words.data(),
                banks[bank].packed_words.size(),
            };
        }
        reset();
    }

    void reset() noexcept {
        ring.fill(0.0F);
        ordered_frame.fill(0.0F);
        windowed_frame.fill(0.0F);
        squared_frame.fill(0.0F);
        fft_buffer.fill({0.0F, 0.0F});
        fft_scratch.fill(0.0F);
        magnitude.fill(0.0F);
        previous_magnitude.fill(0.0F);
        magnitude_flux.fill(0.0F);
        harmonic_unit.fill(0.0F);
        corrected_side_unit.fill(0.0F);
        fundamental.fill(0.0F);
        salience.fill(0.0F);
        plain_contrast.fill(0.0F);
        corrected_contrast.fill(0.0F);
        subharmonic_margin.fill(0.0F);
        positive_flux.fill(0.0F);
        fast_slow.fill(0.0F);
        previous_salience.fill(0.0F);
        ema_salience.fill(0.0F);
        previous_contrast.fill(0.5F);
        ema_contrast.fill(0.5F);
        contrast_positive_flux.fill(0.0F);
        contrast_fast_slow.fill(0.0F);
        plain_features.fill(0.0F);
        hcontrast_features.fill(0.0F);
        hprofile_features.fill(0.0F);
        cattack_features.fill(0.0F);
        for (auto& bank : banks) {
            bank.reset();
        }
        write_index = 0U;
        sample_count = 0U;
        next_frame_sample = 1U;
        frame_count = 0U;
    }

    void append(const float* samples, const std::size_t count) noexcept {
        const auto first = std::min(
            count, kStrictCap16V3FrontendFftSize - write_index);
        std::copy_n(samples, first, ring.data() + write_index);
        const auto remaining = count - first;
        if (remaining != 0U) {
            std::copy_n(samples + first, remaining, ring.data());
        }
        write_index =
            (write_index + count) % kStrictCap16V3FrontendFftSize;
    }

    void build_pitch_features() noexcept {
        for (std::size_t index = 0U;
             index < kStrictCap16V3FrontendFftSize; ++index) {
            const auto ring_index =
                (write_index + index) % kStrictCap16V3FrontendFftSize;
            const float sample = ring[ring_index];
            ordered_frame[index] = sample;
            windowed_frame[index] = sample * artifact.window[index];
            squared_frame[index] = sample * sample;
        }

        auto* fft_storage = reinterpret_cast<float*>(fft_buffer.data());
        for (std::size_t index = 0U;
             index < kStrictCap16V3FrontendFftSize; ++index) {
            fft_storage[index + 1U] = windowed_frame[index];
        }
        fft_storage[kSpectrumBins * 2U - 1U] = 0.0F;
        fft_plan->exec_with_scratch(
            fft_storage + 1U,
            fft_scratch.data(),
            1.0F,
            pocketfft::FORWARD);
        fft_buffer[0] = fft_buffer[0].imag();
        for (std::size_t index = 0U; index < kSpectrumBins; ++index) {
            magnitude[index] = numpy_complex64_absolute(
                fft_buffer[index].real(), fft_buffer[index].imag());
            magnitude[index] /= artifact.magnitude_normalizer;
        }

        for (std::size_t pitch = 0U; pitch < kNoteCount; ++pitch) {
            for (std::size_t harmonic = 0U;
                 harmonic < kHarmonics; ++harmonic) {
                const auto index = pitch * kHarmonics + harmonic;
                const auto sampled = sample_bin(
                    magnitude, artifact.harmonic_bins[index]);
                harmonic_unit[index] = unit_db_float(sampled);
                const auto low_sampled = sample_bin(
                    magnitude, artifact.corrected_side_low_bins[index]);
                const auto high_sampled = sample_bin(
                    magnitude, artifact.corrected_side_high_bins[index]);
                const auto low = unit_db_float(low_sampled);
                const auto high = unit_db_float(high_sampled);
                corrected_side_unit[index] = 0.5F * (low + high);
            }
            fundamental[pitch] = harmonic_unit[pitch * kHarmonics];
            salience[pitch] = dot_float(
                harmonic_unit.data() + pitch * kHarmonics,
                artifact.harmonic_weights.data(), kHarmonics);
            const float plain_side = 0.5F * (
                unit_db_float(sample_bin(
                    magnitude, artifact.plain_side_low_bins[pitch])) +
                unit_db_float(sample_bin(
                    magnitude, artifact.plain_side_high_bins[pitch])));
            const float corrected_side = dot_float(
                corrected_side_unit.data() + pitch * kHarmonics,
                artifact.harmonic_weights.data(), kHarmonics);
            plain_contrast[pitch] =
                clip_unit(salience[pitch] - plain_side + 0.5F);
            corrected_contrast[pitch] =
                clip_unit(salience[pitch] - corrected_side + 0.5F);
            const float subharmonic = unit_db_float(sample_bin(
                magnitude, artifact.subharmonic_bins[pitch]));
            subharmonic_margin[pitch] =
                clip_unit(salience[pitch] - subharmonic + 0.5F);
            positive_flux[pitch] =
                clip_unit(salience[pitch] - previous_salience[pitch]);
            ema_salience[pitch] += 0.08F *
                (salience[pitch] - ema_salience[pitch]);
            fast_slow[pitch] =
                clip_unit(salience[pitch] - ema_salience[pitch] + 0.5F);
            contrast_positive_flux[pitch] = clip_unit(
                corrected_contrast[pitch] - previous_contrast[pitch]);
            ema_contrast[pitch] += 0.08F *
                (corrected_contrast[pitch] - ema_contrast[pitch]);
            contrast_fast_slow[pitch] = clip_unit(
                corrected_contrast[pitch] - ema_contrast[pitch] + 0.5F);
        }

        const double rms_mean = pairwise_sum_double_from_float(
            squared_frame.data(), squared_frame.size()) /
            static_cast<double>(squared_frame.size());
        const float rms_unit = unit_db_double(std::sqrt(rms_mean));
        for (std::size_t index = 0U; index < kSpectrumBins; ++index) {
            magnitude_flux[index] =
                std::max(magnitude[index] - previous_magnitude[index], 0.0F);
        }
        const float broadband_flux = pairwise_sum_float(
            magnitude_flux.data(), magnitude_flux.size()) /
            static_cast<float>(magnitude_flux.size());
        const float broadband_flux_unit = static_cast<float>(std::max(
            0.0, std::min(1.0, static_cast<double>(broadband_flux) * 40.0)));
        const float magnitude_sum_float =
            pairwise_sum_float(magnitude.data(), magnitude.size());
        const double magnitude_sum = static_cast<double>(magnitude_sum_float);
        const double centroid_numerator = static_cast<double>(dot_float(
            magnitude.data(), artifact.frequency_axis.data(), magnitude.size()));
        const double centroid = magnitude_sum > 1.0e-9
            ? centroid_numerator / magnitude_sum
            : 0.0;
        const float centroid_unit = static_cast<float>(std::max(
            0.0, std::min(1.0, centroid / 11025.0)));

        const auto copy_group = [](auto& destination,
                                   const std::size_t offset,
                                   const auto& source) noexcept {
            std::copy(source.begin(), source.end(),
                      destination.begin() + static_cast<std::ptrdiff_t>(offset));
        };
        for (auto* destination : {plain_features.data(),
                                  hcontrast_features.data()}) {
            std::copy(fundamental.begin(), fundamental.end(), destination + 0U);
            std::copy(salience.begin(), salience.end(), destination + 49U);
            std::copy(subharmonic_margin.begin(), subharmonic_margin.end(),
                      destination + 147U);
            std::copy(positive_flux.begin(), positive_flux.end(),
                      destination + 196U);
            std::copy(fast_slow.begin(), fast_slow.end(), destination + 245U);
            destination[294U] = rms_unit;
            destination[295U] = broadband_flux_unit;
            destination[296U] = centroid_unit;
        }
        std::copy(plain_contrast.begin(), plain_contrast.end(),
                  plain_features.begin() + 98);
        std::copy(corrected_contrast.begin(), corrected_contrast.end(),
                  hcontrast_features.begin() + 98);

        copy_group(hprofile_features, 0U, fundamental);
        copy_group(hprofile_features, 49U, salience);
        copy_group(hprofile_features, 98U, corrected_contrast);
        copy_group(hprofile_features, 147U, subharmonic_margin);
        copy_group(hprofile_features, 196U, positive_flux);
        copy_group(hprofile_features, 245U, fast_slow);
        for (std::size_t harmonic = 0U;
             harmonic < kHarmonics; ++harmonic) {
            for (std::size_t pitch = 0U; pitch < kNoteCount; ++pitch) {
                const auto source = pitch * kHarmonics + harmonic;
                hprofile_features[294U + harmonic * kNoteCount + pitch] =
                    clip_unit(
                        harmonic_unit[source] - corrected_side_unit[source] +
                        0.5F);
            }
        }
        hprofile_features[588U] = rms_unit;
        hprofile_features[589U] = broadband_flux_unit;
        hprofile_features[590U] = centroid_unit;

        copy_group(cattack_features, 0U, fundamental);
        copy_group(cattack_features, 49U, salience);
        copy_group(cattack_features, 98U, corrected_contrast);
        copy_group(cattack_features, 147U, subharmonic_margin);
        copy_group(cattack_features, 196U, positive_flux);
        copy_group(cattack_features, 245U, fast_slow);
        copy_group(cattack_features, 294U, contrast_positive_flux);
        copy_group(cattack_features, 343U, contrast_fast_slow);
        cattack_features[392U] = rms_unit;
        cattack_features[393U] = broadband_flux_unit;
        cattack_features[394U] = centroid_unit;

        previous_salience = salience;
        previous_contrast = corrected_contrast;
        previous_magnitude = magnitude;
    }

    void emit_frame(
        StrictCap16V3FrontendFrameCallback callback,
        void* callback_user) {
        build_pitch_features();
        banks[0].encode(plain_features.data(), frame_count);
        banks[1].encode(hcontrast_features.data(), frame_count);
        banks[2].encode(hprofile_features.data(), frame_count);
        banks[3].encode(cattack_features.data(), frame_count);
        callback(callback_user, frame_count, frame_rows);
        ++frame_count;
    }
};

StrictCap16V3StreamingFrontend::StrictCap16V3StreamingFrontend(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

StrictCap16V3StreamingFrontend::~StrictCap16V3StreamingFrontend() = default;
StrictCap16V3StreamingFrontend::StrictCap16V3StreamingFrontend(
    StrictCap16V3StreamingFrontend&&) noexcept = default;
StrictCap16V3StreamingFrontend&
StrictCap16V3StreamingFrontend::operator=(
    StrictCap16V3StreamingFrontend&&) noexcept = default;

StrictCap16V3StreamingFrontend StrictCap16V3StreamingFrontend::load(
    const std::filesystem::path& artifact_path) {
    return StrictCap16V3StreamingFrontend(
        std::make_unique<Impl>(load_artifact(artifact_path)));
}

void StrictCap16V3StreamingFrontend::reset() noexcept {
    if (impl_) {
        impl_->reset();
    }
}

StrictCap16V3FrontendStatus StrictCap16V3StreamingFrontend::process_block(
    const float* mono_samples,
    const std::size_t sample_count,
    const StrictCap16V3FrontendFrameCallback callback,
    void* callback_user) noexcept {
    if (!impl_) {
        return StrictCap16V3FrontendStatus::unprepared;
    }
    if (sample_count != 0U && mono_samples == nullptr) {
        return StrictCap16V3FrontendStatus::null_audio;
    }
    if (callback == nullptr) {
        return StrictCap16V3FrontendStatus::null_frame_callback;
    }
    try {
        std::size_t offset = 0U;
        while (offset < sample_count) {
            const auto needed_u64 =
                impl_->next_frame_sample - impl_->sample_count;
            const auto needed = static_cast<std::size_t>(needed_u64);
            const auto take = std::min(needed, sample_count - offset);
            impl_->append(mono_samples + offset, take);
            impl_->sample_count += take;
            offset += take;
            if (impl_->sample_count == impl_->next_frame_sample) {
                impl_->emit_frame(callback, callback_user);
                impl_->next_frame_sample += kStrictCap16V3FrontendHopSize;
            }
        }
    } catch (...) {
        // PocketFFT execution is allocation-free after plan creation and is
        // not expected to throw. Contain any third-party failure regardless;
        // exceptions must never cross a realtime host callback.
        return StrictCap16V3FrontendStatus::processing_failure;
    }
    return StrictCap16V3FrontendStatus::success;
}

std::uint64_t StrictCap16V3StreamingFrontend::consumed_sample_count()
    const noexcept {
    return impl_ ? impl_->sample_count : 0U;
}

std::uint64_t StrictCap16V3StreamingFrontend::emitted_frame_count()
    const noexcept {
    return impl_ ? impl_->frame_count : 0U;
}

}  // namespace tmgm::native
