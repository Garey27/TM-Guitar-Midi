#include "tmgm/ensemble_bundle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tmgm::native {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'T', 'M', 'G', 'M', 'B', 'N', 'D', 0};
constexpr std::size_t kChecksumOffset = 224U;
constexpr std::size_t kChecksumBytes = 32U;
constexpr std::uint32_t kWeightBits = 16U;

[[nodiscard]] std::runtime_error format_error(const std::string& message) {
    return std::runtime_error("invalid TMGMBND bundle: " + message);
}

[[nodiscard]] std::invalid_argument bundle_error(const std::string& message) {
    return std::invalid_argument("invalid native TM ensemble bundle: " + message);
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

[[nodiscard]] std::uint64_t checked_multiply(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* label) {
    if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw format_error(std::string(label) + " overflows uint64");
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_size(
    const std::uint64_t value,
    const char* label) {
    if (value > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
        throw format_error(std::string(label) + " does not fit address space");
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::uint64_t align8(const std::uint64_t value) {
    return checked_add(value, 7U, "alignment") & ~std::uint64_t{7U};
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

[[nodiscard]] std::int16_t decode_i16(const std::uint8_t* bytes) {
    const auto bits = decode_little_endian<std::uint16_t>(bytes);
    std::int16_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
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

[[nodiscard]] bool all_zero(
    const std::vector<std::uint8_t>& bytes,
    const std::size_t begin,
    const std::size_t end) {
    return std::all_of(
        bytes.begin() + static_cast<std::ptrdiff_t>(begin),
        bytes.begin() + static_cast<std::ptrdiff_t>(end),
        [](const std::uint8_t value) { return value == 0U; });
}

[[nodiscard]] bool digest_is_zero(
    const std::array<std::uint8_t, 32>& digest) noexcept {
    return std::all_of(
        digest.begin(), digest.end(), [](const std::uint8_t byte) {
            return byte == 0U;
        });
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
            std::fill(
                block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                block_.end(),
                std::uint8_t{0U});
            transform(block_.data());
            block_size_ = 0U;
        }
        std::fill(
            block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
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
                rotate_right(right, 17U) ^ rotate_right(right, 19U) ^
                (right >> 10U);
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

[[nodiscard]] std::array<std::uint8_t, 32> sha256(
    const void* bytes,
    const std::size_t size) {
    Sha256 sha;
    sha.update(bytes, size);
    return sha.finish();
}

[[nodiscard]] bool valid_identifier(const std::string& value) noexcept {
    if (value.empty() || value.size() > 64U) {
        return false;
    }
    const auto ascii_alnum = [](const unsigned char character) {
        return (character >= 'A' && character <= 'Z') ||
               (character >= 'a' && character <= 'z') ||
               (character >= '0' && character <= '9');
    };
    if (!ascii_alnum(static_cast<unsigned char>(value.front()))) {
        return false;
    }
    return std::all_of(
        value.begin(), value.end(), [&](const char character) {
            const auto byte = static_cast<unsigned char>(character);
            return ascii_alnum(byte) || character == '.' || character == '_' ||
                   character == '-';
        });
}

[[nodiscard]] std::string decode_identifier(const std::uint8_t* bytes) {
    std::size_t size = 0U;
    while (size < 64U && bytes[size] != 0U) {
        ++size;
    }
    if (size < 64U && !std::all_of(
            bytes + size, bytes + 64U,
            [](const std::uint8_t value) { return value == 0U; })) {
        throw format_error("member ID padding is non-zero");
    }
    const std::string result(
        reinterpret_cast<const char*>(bytes),
        reinterpret_cast<const char*>(bytes + size));
    if (!valid_identifier(result)) {
        throw format_error("invalid member ID");
    }
    return result;
}

[[nodiscard]] std::array<std::uint8_t, 32> member_order_digest(
    const EnsembleBundle& bundle,
    const EnsembleMemberHead head) {
    Sha256 sha;
    bool first = true;
    for (const auto& member : bundle.members) {
        if (member.head != head) {
            continue;
        }
        if (!first) {
            const std::uint8_t separator = 0U;
            sha.update(&separator, sizeof(separator));
        }
        sha.update(member.identifier.data(), member.identifier.size());
        first = false;
    }
    return sha.finish();
}

[[nodiscard]] std::vector<std::uint8_t> read_file(
    const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error(
            "failed to open TMGMBND bundle: " + path.string());
    }
    const auto end = stream.tellg();
    if (end < 0) {
        throw std::runtime_error(
            "failed to determine TMGMBND bundle size: " + path.string());
    }
    const auto size = static_cast<std::uint64_t>(end);
    std::vector<std::uint8_t> bytes(checked_size(size, "bundle file"));
    stream.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        stream.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    }
    if (!stream || static_cast<std::size_t>(stream.gcount()) != bytes.size()) {
        throw std::runtime_error(
            "failed to read complete TMGMBND bundle: " + path.string());
    }
    return bytes;
}

void require_range(
    const std::uint64_t offset,
    const std::uint64_t size,
    const std::uint64_t file_size,
    const char* label) {
    if (offset > file_size || size > file_size - offset) {
        throw format_error(std::string(label) + " is outside the file");
    }
}

}  // namespace

std::string ensemble_sha256_hex(
    const std::array<std::uint8_t, 32>& digest) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const auto byte : digest) {
        stream << std::setw(2) << static_cast<unsigned>(byte);
    }
    return stream.str();
}

void validate_ensemble_bundle(const EnsembleBundle& bundle) {
    if (bundle.format_version != kEnsembleBundleLegacyFormatVersion &&
        bundle.format_version != kEnsembleBundleFormatVersion) {
        throw bundle_error("unsupported in-memory bundle format version");
    }
    if (bundle.feature_count == 0U || bundle.feature_count > 32768U) {
        throw bundle_error("feature count must be in [1, 32768]");
    }
    if (bundle.output_count == 0U || bundle.midi_min < 0 ||
        bundle.midi_max > 127 || bundle.midi_max < bundle.midi_min ||
        static_cast<std::uint32_t>(bundle.midi_max - bundle.midi_min + 1) !=
            bundle.output_count) {
        throw bundle_error("MIDI range disagrees with output count");
    }
    if (bundle.sample_rate == 0U || bundle.hop_size == 0U) {
        throw bundle_error("audio timebase must be positive");
    }
    const auto validate_head = [](const EnsembleHeadConfig& config) {
        if (config.fusion != EnsembleFusion::mean) {
            throw bundle_error("unsupported per-head fusion rule");
        }
        if (config.quantization == 0U || config.quantization > 1000000U) {
            throw bundle_error("per-head quantization must be in [1, 1000000]");
        }
        if (digest_is_zero(config.member_order_sha256)) {
            throw bundle_error("per-head member order fingerprint is zero");
        }
    };
    validate_head(bundle.activity);
    validate_head(bundle.onset);
    if (bundle.members.size() < 2U) {
        throw bundle_error("bundle needs activity and onset members");
    }
    if (digest_is_zero(bundle.feature_fingerprint_sha256) ||
        digest_is_zero(bundle.bundle_checksum_sha256)) {
        throw bundle_error("bundle fingerprints/checksum must be non-zero");
    }

    std::set<std::string> identifiers;
    std::size_t activity_count = 0U;
    std::size_t onset_count = 0U;
    bool onset_started = false;
    for (std::size_t index = 0; index < bundle.members.size(); ++index) {
        const auto& member = bundle.members[index];
        if (!valid_identifier(member.identifier) ||
            !identifiers.insert(member.identifier).second) {
            throw bundle_error("member IDs must be valid and unique");
        }
        if (member.head == EnsembleMemberHead::activity) {
            if (onset_started) {
                throw bundle_error("activity member appears after onset member");
            }
            ++activity_count;
        } else if (member.head == EnsembleMemberHead::onset) {
            onset_started = true;
            ++onset_count;
        } else {
            throw bundle_error("unknown member head");
        }
        if (!std::isfinite(member.robust_scale) ||
            member.robust_scale <= 0.0F) {
            throw bundle_error("member robust scale must be finite and positive");
        }
        if (member.feature_count != bundle.feature_count ||
            member.output_count != bundle.output_count ||
            member.clause_count == 0U ||
            member.literal_count != bundle.feature_count * 2U) {
            throw bundle_error("member geometry disagrees with bundle");
        }
        if (digest_is_zero(member.source_model_sha256)) {
            throw bundle_error("source model SHA-256 must be non-zero");
        }
        if (member.clause_offsets.size() !=
            static_cast<std::size_t>(member.clause_count) + 1U) {
            throw bundle_error("clause-offset payload has the wrong size");
        }
        if (member.clause_offsets.front() != 0U ||
            member.clause_offsets.back() != member.literal_ids.size()) {
            throw bundle_error("clause offsets do not span literal IDs");
        }
        if (member.weights.size() !=
            static_cast<std::size_t>(member.clause_count) *
                member.output_count) {
            throw bundle_error("weight payload has the wrong size");
        }
        for (std::uint32_t clause = 0; clause < member.clause_count; ++clause) {
            const auto begin = member.clause_offsets[clause];
            const auto end = member.clause_offsets[clause + 1U];
            if (begin > end || end > member.literal_ids.size()) {
                throw bundle_error("clause offsets are not monotonic");
            }
            std::uint32_t previous = 0U;
            bool first = true;
            for (auto literal = begin; literal < end; ++literal) {
                const auto value = member.literal_ids[literal];
                if (value >= member.literal_count ||
                    (!first && value <= previous)) {
                    throw bundle_error(
                        "clause literal IDs must be in-range and strictly sorted");
                }
                previous = value;
                first = false;
            }
        }
    }
    if (activity_count < 1U || onset_count < 1U) {
        throw bundle_error("bundle needs at least one member for each head");
    }
    if (member_order_digest(bundle, EnsembleMemberHead::activity) !=
            bundle.activity.member_order_sha256 ||
        member_order_digest(bundle, EnsembleMemberHead::onset) !=
            bundle.onset.member_order_sha256) {
        throw bundle_error("per-head member order fingerprint mismatch");
    }
}

EnsembleBundle load_ensemble_bundle(
    const std::filesystem::path& path,
    const bool verify_checksum) {
    auto bytes = read_file(path);
    if (bytes.size() < kEnsembleBundleHeaderBytes) {
        throw format_error("file is truncated");
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
        throw format_error("wrong magic");
    }
    const auto format_version =
        decode_little_endian<std::uint32_t>(bytes.data() + 8U);
    if (format_version != kEnsembleBundleLegacyFormatVersion &&
        format_version != kEnsembleBundleFormatVersion) {
        throw format_error("unsupported version");
    }
    if (decode_little_endian<std::uint32_t>(bytes.data() + 12U) !=
        kEnsembleBundleHeaderBytes) {
        throw format_error("wrong header size");
    }
    if (decode_little_endian<std::uint32_t>(bytes.data() + 16U) !=
        kEnsembleMemberDescriptorBytes) {
        throw format_error("wrong member descriptor size");
    }
    if (decode_little_endian<std::uint32_t>(bytes.data() + 20U) != 0U ||
        decode_little_endian<std::uint32_t>(bytes.data() + 84U) != 0U) {
        throw format_error("unknown flags or non-zero reserved header bytes");
    }

    const auto model_count =
        decode_little_endian<std::uint32_t>(bytes.data() + 24U);
    const auto activity_count =
        decode_little_endian<std::uint32_t>(bytes.data() + 28U);
    const auto onset_count =
        decode_little_endian<std::uint32_t>(bytes.data() + 32U);
    const auto descriptors_offset =
        decode_little_endian<std::uint64_t>(bytes.data() + 88U);
    const auto descriptors_bytes =
        decode_little_endian<std::uint64_t>(bytes.data() + 96U);
    const auto payload_offset =
        decode_little_endian<std::uint64_t>(bytes.data() + 104U);
    const auto payload_bytes =
        decode_little_endian<std::uint64_t>(bytes.data() + 112U);
    const auto file_bytes =
        decode_little_endian<std::uint64_t>(bytes.data() + 120U);
    if (model_count < 2U || activity_count < 1U || onset_count < 1U ||
        activity_count + onset_count != model_count) {
        throw format_error("invalid activity/onset member counts");
    }
    const auto expected_descriptors_bytes = checked_multiply(
        model_count, kEnsembleMemberDescriptorBytes, "descriptor byte count");
    const auto expected_payload_offset = align8(checked_add(
        kEnsembleBundleHeaderBytes,
        expected_descriptors_bytes,
        "payload offset"));
    if (descriptors_offset != kEnsembleBundleHeaderBytes ||
        descriptors_bytes != expected_descriptors_bytes ||
        payload_offset != expected_payload_offset ||
        checked_add(payload_offset, payload_bytes, "bundle file size") !=
            file_bytes ||
        file_bytes != bytes.size()) {
        throw format_error("top-level section layout is inconsistent");
    }
    require_range(descriptors_offset, descriptors_bytes, file_bytes, "descriptors");
    require_range(payload_offset, payload_bytes, file_bytes, "payload");
    if (!all_zero(
            bytes,
            checked_size(descriptors_offset + descriptors_bytes, "descriptor end"),
            checked_size(payload_offset, "payload offset"))) {
        throw format_error("non-zero padding before payload");
    }

    std::array<std::uint8_t, 32> stored_checksum{};
    std::copy_n(
        bytes.begin() + static_cast<std::ptrdiff_t>(kChecksumOffset),
        stored_checksum.size(),
        stored_checksum.begin());
    if (digest_is_zero(stored_checksum)) {
        throw format_error("bundle checksum is zero");
    }
    if (verify_checksum) {
        auto canonical = bytes;
        std::fill(
            canonical.begin() + static_cast<std::ptrdiff_t>(kChecksumOffset),
            canonical.begin() +
                static_cast<std::ptrdiff_t>(kChecksumOffset + kChecksumBytes),
            std::uint8_t{0U});
        if (sha256(canonical.data(), canonical.size()) != stored_checksum) {
            throw format_error("SHA-256 mismatch");
        }
    }

    EnsembleBundle bundle;
    bundle.format_version = format_version;
    bundle.feature_count =
        decode_little_endian<std::uint32_t>(bytes.data() + 36U);
    bundle.output_count =
        decode_little_endian<std::uint32_t>(bytes.data() + 40U);
    bundle.midi_min = decode_i32(bytes.data() + 44U);
    bundle.midi_max = decode_i32(bytes.data() + 48U);
    bundle.sample_rate =
        decode_little_endian<std::uint32_t>(bytes.data() + 52U);
    bundle.hop_size =
        decode_little_endian<std::uint32_t>(bytes.data() + 56U);
    const auto activity_fusion =
        decode_little_endian<std::uint32_t>(bytes.data() + 60U);
    const auto onset_fusion =
        decode_little_endian<std::uint32_t>(bytes.data() + 72U);
    if (activity_fusion != static_cast<std::uint32_t>(EnsembleFusion::mean) ||
        onset_fusion != static_cast<std::uint32_t>(EnsembleFusion::mean)) {
        throw format_error("unsupported per-head fusion rule");
    }
    bundle.activity.fusion = EnsembleFusion::mean;
    bundle.activity.quantization =
        decode_little_endian<std::uint32_t>(bytes.data() + 64U);
    bundle.activity.ensemble_threshold = decode_i32(bytes.data() + 68U);
    bundle.onset.fusion = EnsembleFusion::mean;
    bundle.onset.quantization =
        decode_little_endian<std::uint32_t>(bytes.data() + 76U);
    bundle.onset.ensemble_threshold = decode_i32(bytes.data() + 80U);
    std::copy_n(
        bytes.begin() + 128,
        bundle.activity.member_order_sha256.size(),
        bundle.activity.member_order_sha256.begin());
    std::copy_n(
        bytes.begin() + 160,
        bundle.onset.member_order_sha256.size(),
        bundle.onset.member_order_sha256.begin());
    std::copy_n(
        bytes.begin() + 192,
        bundle.feature_fingerprint_sha256.size(),
        bundle.feature_fingerprint_sha256.begin());
    bundle.bundle_checksum_sha256 = stored_checksum;

    bundle.members.reserve(model_count);
    std::uint64_t expected_section_offset = payload_offset;
    for (std::uint32_t index = 0; index < model_count; ++index) {
        const auto descriptor_offset = descriptors_offset +
            static_cast<std::uint64_t>(index) * kEnsembleMemberDescriptorBytes;
        const auto* descriptor = bytes.data() + descriptor_offset;
        SparseTmEnsembleMember member;
        member.identifier = decode_identifier(descriptor);
        const auto head = decode_little_endian<std::uint32_t>(descriptor + 64U);
        if (head == static_cast<std::uint32_t>(EnsembleMemberHead::activity)) {
            member.head = EnsembleMemberHead::activity;
        } else if (head == static_cast<std::uint32_t>(EnsembleMemberHead::onset)) {
            member.head = EnsembleMemberHead::onset;
        } else {
            throw format_error("unknown member head");
        }
        const auto expected_head = index < activity_count
            ? EnsembleMemberHead::activity
            : EnsembleMemberHead::onset;
        if (member.head != expected_head) {
            throw format_error("member head order disagrees with header counts");
        }
        if (decode_little_endian<std::uint32_t>(descriptor + 68U) != 0U ||
            decode_little_endian<std::uint64_t>(descriptor + 184U) != 0U) {
            throw format_error("unknown member flags or reserved value");
        }
        member.score_threshold = decode_i32(descriptor + 72U);
        member.robust_scale = decode_f32(descriptor + 76U);
        member.feature_count =
            decode_little_endian<std::uint32_t>(descriptor + 80U);
        member.output_count =
            decode_little_endian<std::uint32_t>(descriptor + 84U);
        member.clause_count =
            decode_little_endian<std::uint32_t>(descriptor + 88U);
        member.literal_count =
            decode_little_endian<std::uint32_t>(descriptor + 92U);
        const auto included_literal_count =
            decode_little_endian<std::uint32_t>(descriptor + 96U);
        if (decode_little_endian<std::uint32_t>(descriptor + 100U) !=
            kWeightBits) {
            throw format_error("unsupported member weight width");
        }
        std::copy_n(
            descriptor + 104U,
            member.source_model_sha256.size(),
            member.source_model_sha256.begin());

        const auto offsets_offset =
            decode_little_endian<std::uint64_t>(descriptor + 136U);
        const auto offsets_bytes =
            decode_little_endian<std::uint64_t>(descriptor + 144U);
        const auto literals_offset =
            decode_little_endian<std::uint64_t>(descriptor + 152U);
        const auto literals_bytes =
            decode_little_endian<std::uint64_t>(descriptor + 160U);
        const auto weights_offset =
            decode_little_endian<std::uint64_t>(descriptor + 168U);
        const auto weights_bytes =
            decode_little_endian<std::uint64_t>(descriptor + 176U);
        const auto expected_offsets_bytes = checked_multiply(
            static_cast<std::uint64_t>(member.clause_count) + 1U,
            sizeof(std::uint32_t),
            "clause offsets");
        const auto expected_literals_bytes = checked_multiply(
            included_literal_count, sizeof(std::uint16_t), "literal IDs");
        const auto expected_weights_bytes = checked_multiply(
            checked_multiply(
                member.clause_count, member.output_count, "weight count"),
            sizeof(std::int16_t),
            "weights");
        if (offsets_bytes != expected_offsets_bytes ||
            literals_bytes != expected_literals_bytes ||
            weights_bytes != expected_weights_bytes) {
            throw format_error("member section size disagrees with dimensions");
        }

        const auto check_section = [&](const std::uint64_t offset,
                                       const std::uint64_t size,
                                       const char* label) {
            const auto aligned = align8(expected_section_offset);
            if (offset != aligned) {
                throw format_error(std::string(label) + " is not canonical/contiguous");
            }
            require_range(offset, size, file_bytes, label);
            if (!all_zero(
                    bytes,
                    checked_size(expected_section_offset, "section padding"),
                    checked_size(aligned, "section padding"))) {
                throw format_error(std::string(label) + " has non-zero alignment padding");
            }
            expected_section_offset = checked_add(offset, size, label);
        };
        check_section(offsets_offset, offsets_bytes, "clause offsets");
        check_section(literals_offset, literals_bytes, "literal IDs");
        check_section(weights_offset, weights_bytes, "weights");

        member.clause_offsets.resize(
            static_cast<std::size_t>(member.clause_count) + 1U);
        for (std::size_t item = 0; item < member.clause_offsets.size(); ++item) {
            member.clause_offsets[item] = decode_little_endian<std::uint32_t>(
                bytes.data() + offsets_offset + item * sizeof(std::uint32_t));
        }
        member.literal_ids.resize(included_literal_count);
        for (std::size_t item = 0; item < member.literal_ids.size(); ++item) {
            member.literal_ids[item] = decode_little_endian<std::uint16_t>(
                bytes.data() + literals_offset + item * sizeof(std::uint16_t));
        }
        const auto weight_count = checked_size(
            checked_multiply(
                member.clause_count, member.output_count, "weight count"),
            "weight count");
        member.weights.resize(weight_count);
        for (std::size_t item = 0; item < member.weights.size(); ++item) {
            member.weights[item] = decode_i16(
                bytes.data() + weights_offset + item * sizeof(std::int16_t));
        }
        bundle.members.push_back(std::move(member));
    }
    if (expected_section_offset != file_bytes) {
        throw format_error("trailing bytes after canonical member sections");
    }

    try {
        validate_ensemble_bundle(bundle);
    } catch (const std::invalid_argument& error) {
        throw format_error(error.what());
    }
    return bundle;
}

}  // namespace tmgm::native
