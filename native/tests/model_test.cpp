#include "tmgm/model.hpp"

#include "tmgm/dataset.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

std::uint32_t read_u32_le(const std::vector<std::uint8_t>& bytes, const std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1U]) << 8U) |
           (static_cast<std::uint32_t>(bytes[offset + 2U]) << 16U) |
           (static_cast<std::uint32_t>(bytes[offset + 3U]) << 24U);
}

void write_u32_le(
    std::vector<std::uint8_t>& bytes,
    const std::size_t offset,
    std::uint32_t value) {
    for (std::size_t index = 0; index < 4U; ++index) {
        bytes[offset + index] = static_cast<std::uint8_t>(value & 0xffU);
        value >>= 8U;
    }
}

void write_f32_le(
    std::vector<std::uint8_t>& bytes,
    const std::size_t offset,
    const float value) {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32_le(bytes, offset, bits);
}

void write_sha256_hex(
    std::vector<std::uint8_t>& bytes,
    const std::size_t offset,
    const std::string& hex) {
    require(hex.size() == 64U, "test SHA-256 fixture has the wrong length");
    const auto digit = [](const char value) -> std::uint8_t {
        if (value >= '0' && value <= '9') {
            return static_cast<std::uint8_t>(value - '0');
        }
        if (value >= 'a' && value <= 'f') {
            return static_cast<std::uint8_t>(value - 'a' + 10);
        }
        throw std::runtime_error("test SHA-256 fixture is not hexadecimal");
    };
    for (std::size_t index = 0; index < 32U; ++index) {
        bytes[offset + index] = static_cast<std::uint8_t>(
            (digit(hex[index * 2U]) << 4U) | digit(hex[index * 2U + 1U]));
    }
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("could not open test model");
    }
    stream.seekg(0, std::ios::end);
    const auto length = stream.tellg();
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    stream.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        throw std::runtime_error("could not read test model");
    }
    return bytes;
}

void write_file(
    const std::filesystem::path& path,
    const std::vector<std::uint8_t>& bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        throw std::runtime_error("could not write corrupt test model");
    }
}

tmgm::native::NativeTmModel make_model() {
    tmgm::native::NativeTmModel model;
    model.head = tmgm::native::TmModelHead::onset;
    model.dimensions.feature_count = 17U;
    model.dimensions.output_count = 4U;
    model.dimensions.clause_count = 7U;
    model.dimensions.state_bits = 8U;
    model.training.threshold = 93;
    model.training.specificity = 4.25F;
    model.training.negative_samples = 7.5F;
    model.training.type_i_ii_ratio = 1.75F;
    model.training.max_included_literals = 29U;
    model.training.epochs_trained = 42U;
    model.training.seed = 0x0123456789abcdefULL;
    model.training.feature_negation = true;
    model.training.boost_true_positive_feedback = false;
    model.score_threshold = -3;
    model.midi.minimum_note = 40;
    model.midi.maximum_note = 43;
    model.midi.channel = 9U;
    model.midi.audio_sample_rate = 22050U;
    model.midi.analysis_hop_samples = 256U;
    model.feature_fingerprint_sha256.fill(0x42U);

    const auto ta_words = static_cast<std::size_t>(model.dimensions.clause_count) *
                          model.dimensions.state_bits *
                          tmgm::native::tm_model_literal_word_count(model);
    model.ta_bitplanes.resize(ta_words);
    for (std::size_t index = 0; index < model.ta_bitplanes.size(); ++index) {
        model.ta_bitplanes[index] =
            0xa5a50000U ^ static_cast<std::uint32_t>(index * 0x10203U);
    }
    model.weights.resize(static_cast<std::size_t>(model.dimensions.output_count) *
                         model.dimensions.clause_count);
    for (std::size_t index = 0; index < model.weights.size(); ++index) {
        model.weights[index] = static_cast<std::int32_t>(index * 17U) - 91;
    }
    return model;
}

void require_same_model(
    const tmgm::native::NativeTmModel& left,
    const tmgm::native::NativeTmModel& right) {
    require(left.head == right.head, "head did not round-trip");
    require(left.dimensions.feature_count == right.dimensions.feature_count,
            "feature count did not round-trip");
    require(left.dimensions.output_count == right.dimensions.output_count,
            "output count did not round-trip");
    require(left.dimensions.clause_count == right.dimensions.clause_count,
            "clause count did not round-trip");
    require(left.dimensions.state_bits == right.dimensions.state_bits,
            "state bits did not round-trip");
    require(left.training.threshold == right.training.threshold,
            "training threshold did not round-trip");
    require(left.training.specificity == right.training.specificity,
            "specificity did not round-trip");
    require(left.training.negative_samples == right.training.negative_samples,
            "negative samples did not round-trip");
    require(left.training.type_i_ii_ratio == right.training.type_i_ii_ratio,
            "feedback ratio did not round-trip");
    require(left.training.max_included_literals ==
                right.training.max_included_literals,
            "literal limit did not round-trip");
    require(left.training.epochs_trained == right.training.epochs_trained,
            "epoch count did not round-trip");
    require(left.training.seed == right.training.seed, "seed did not round-trip");
    require(left.training.feature_negation == right.training.feature_negation,
            "feature-negation flag did not round-trip");
    require(left.training.boost_true_positive_feedback ==
                right.training.boost_true_positive_feedback,
            "boost flag did not round-trip");
    require(left.training.onset_sustain_hard_negatives ==
                right.training.onset_sustain_hard_negatives,
            "onset sustain hard-negative flag did not round-trip");
    require(left.training.onset_sustain_hard_negative_probability ==
                right.training.onset_sustain_hard_negative_probability,
            "onset sustain hard-negative probability did not round-trip");
    require(left.training.onset_sustain_hard_negative_weight_only ==
                right.training.onset_sustain_hard_negative_weight_only,
            "weight-only hard-negative policy did not round-trip");
    require(left.score_threshold == right.score_threshold,
            "score threshold did not round-trip");
    require(left.midi.minimum_note == right.midi.minimum_note,
            "minimum MIDI note did not round-trip");
    require(left.midi.maximum_note == right.midi.maximum_note,
            "maximum MIDI note did not round-trip");
    require(left.midi.channel == right.midi.channel,
            "MIDI channel did not round-trip");
    require(left.midi.audio_sample_rate == right.midi.audio_sample_rate,
            "sample rate did not round-trip");
    require(left.midi.analysis_hop_samples == right.midi.analysis_hop_samples,
            "analysis hop did not round-trip");
    require(left.feature_fingerprint_sha256 ==
                right.feature_fingerprint_sha256,
            "feature-semantics fingerprint did not round-trip");
    require(left.ta_bitplanes == right.ta_bitplanes,
            "TA bit-planes did not round-trip");
    require(left.weights == right.weights, "weights did not round-trip");
}

void test_round_trip_and_wire_layout(const std::filesystem::path& path) {
    const auto original = make_model();
    tmgm::native::save_tm_model(path, original);
    const auto loaded = tmgm::native::load_tm_model(path);
    require_same_model(original, loaded);

    const auto bytes = read_file(path);
    const std::vector<std::uint8_t> expected_magic{
        'T', 'M', 'G', 'M', 'M', 'O', 'D', 0U};
    require(std::equal(expected_magic.begin(), expected_magic.end(), bytes.begin()),
            "wrong model magic");
    require(read_u32_le(bytes, 8U) == tmgm::native::kTmModelFormatVersion,
            "version is not little-endian");
    require(read_u32_le(bytes, 12U) == tmgm::native::kTmModelHeaderBytes,
            "header byte count is wrong");
    require(read_u32_le(bytes, 16U) == 32U, "packed word width is wrong");
    require(read_u32_le(bytes, 24U) == 2U, "head encoding is wrong");
    require(read_u32_le(bytes, 32U) == 17U, "feature encoding is wrong");
    require(read_u32_le(bytes, 44U) == 34U, "literal count is wrong");
    require(read_u32_le(bytes, 48U) == 2U, "literal word count is wrong");
    require(std::all_of(
                bytes.begin() + 192,
                bytes.begin() + 224,
                [](const std::uint8_t value) { return value == 0x42U; }),
            "feature-semantics fingerprint is missing from the wire header");
    require(read_u32_le(bytes, tmgm::native::kTmModelHeaderBytes) ==
                original.ta_bitplanes.front(),
            "TA payload is not little-endian or is at the wrong offset");

    const auto checksum = tmgm::native::calculate_tm_model_checksum(original);
    require(tmgm::native::tm_model_checksum_hex(checksum) ==
                "46298fb15119b062f65ed42372bb50c4d0cbf7d45ca4fa459399d3c2f78c5acf",
            "canonical v3 model SHA-256 differs from the independent fixture");
}

void test_checksum_covers_payload_and_metadata(const std::filesystem::path& path) {
    auto bytes = read_file(path);
    bytes[tmgm::native::kTmModelHeaderBytes + 5U] ^= 0x40U;
    write_file(path, bytes);
    require_throws(
        [&] { static_cast<void>(tmgm::native::load_tm_model(path)); },
        "payload corruption was not detected");
    static_cast<void>(tmgm::native::load_tm_model(path, false));

    const auto original = make_model();
    tmgm::native::save_tm_model(path, original);
    bytes = read_file(path);
    // Low mantissa byte of specificity: remains a valid float, but checksum
    // must still detect this config-only mutation.
    bytes[64U] ^= 0x01U;
    write_file(path, bytes);
    require_throws(
        [&] { static_cast<void>(tmgm::native::load_tm_model(path)); },
        "metadata corruption was not detected");
}

void test_onset_sustain_hard_negative_flag(const std::filesystem::path& path) {
    const auto baseline = make_model();
    auto hard_negative = baseline;
    hard_negative.training.onset_sustain_hard_negatives = true;
    hard_negative.training.onset_sustain_hard_negative_probability = 0.125F;
    hard_negative.training.onset_sustain_hard_negative_weight_only = true;
    require(tmgm::native::calculate_tm_model_checksum(baseline) !=
                tmgm::native::calculate_tm_model_checksum(hard_negative),
            "hard-negative policy was not covered by the checksum");

    tmgm::native::save_tm_model(path, hard_negative);
    const auto bytes = read_file(path);
    require((read_u32_le(bytes, 20U) & (1U << 2U)) != 0U,
            "hard-negative policy flag is missing from the wire header");
    require((read_u32_le(bytes, 20U) & (1U << 3U)) != 0U,
            "weight-only hard-negative flag is missing from the wire header");
    const auto loaded = tmgm::native::load_tm_model(path);
    require_same_model(hard_negative, loaded);

    hard_negative.head = tmgm::native::TmModelHead::activity;
    require_throws(
        [&] { tmgm::native::validate_tm_model(hard_negative); },
        "activity model accepted onset sustain hard negatives");
}

void test_v1_and_transitional_backwards_compatibility(
    const std::filesystem::path& path) {
    auto plain = make_model();
    plain.feature_fingerprint_sha256.fill(0U);
    tmgm::native::save_tm_model(path, plain);
    auto bytes = read_file(path);
    write_u32_le(bytes, 8U, tmgm::native::kTmModelLegacyFormatVersion);
    write_sha256_hex(
        bytes,
        160U,
        "f672b62b8a20fb77bda84d0b0f973dc489b46321c71b8c789d05a25e83aa67bf");
    write_file(path, bytes);
    const auto loaded_plain = tmgm::native::load_tm_model(path);
    require_same_model(plain, loaded_plain);

    auto legacy_hard = make_model();
    legacy_hard.feature_fingerprint_sha256.fill(0U);
    legacy_hard.training.onset_sustain_hard_negatives = true;
    legacy_hard.training.onset_sustain_hard_negative_probability = 1.0F;
    tmgm::native::save_tm_model(path, legacy_hard);
    bytes = read_file(path);
    write_u32_le(bytes, 8U, tmgm::native::kTmModelLegacyFormatVersion);
    write_f32_le(bytes, 108U, 0.0F);
    write_sha256_hex(
        bytes,
        160U,
        "fad09eca9ce58caed58d493306ec30425e38b45165975545e01e633437be2225");
    write_file(path, bytes);
    const auto loaded_legacy_hard = tmgm::native::load_tm_model(path);
    require_same_model(legacy_hard, loaded_legacy_hard);
}

void test_current_rejects_inconsistent_hard_negative_wire_metadata(
    const std::filesystem::path& path) {
    auto hard = make_model();
    hard.training.onset_sustain_hard_negatives = true;
    hard.training.onset_sustain_hard_negative_probability = 0.25F;
    tmgm::native::save_tm_model(path, hard);
    auto bytes = read_file(path);
    write_f32_le(bytes, 108U, 0.0F);
    write_file(path, bytes);
    require_throws(
        [&] { static_cast<void>(tmgm::native::load_tm_model(path, false)); },
        "v2 accepted a zero hard-negative probability");

    tmgm::native::save_tm_model(path, hard);
    bytes = read_file(path);
    write_f32_le(bytes, 108U, 2.0F);
    write_file(path, bytes);
    require_throws(
        [&] { static_cast<void>(tmgm::native::load_tm_model(path, false)); },
        "v2 accepted a hard-negative probability above one");

    const auto plain = make_model();
    tmgm::native::save_tm_model(path, plain);
    bytes = read_file(path);
    write_u32_le(bytes, 20U, read_u32_le(bytes, 20U) | (1U << 3U));
    write_file(path, bytes);
    require_throws(
        [&] { static_cast<void>(tmgm::native::load_tm_model(path, false)); },
        "v2 accepted weight-only feedback without hard negatives");

    tmgm::native::save_tm_model(path, plain);
    bytes = read_file(path);
    write_u32_le(bytes, 20U, read_u32_le(bytes, 20U) | (1U << 31U));
    write_file(path, bytes);
    require_throws(
        [&] { static_cast<void>(tmgm::native::load_tm_model(path, false)); },
        "v2 accepted an unknown model flag");
}

void test_validation_rejects_inconsistent_model() {
    auto model = make_model();
    model.weights.pop_back();
    require_throws(
        [&] { tmgm::native::validate_tm_model(model); },
        "wrong weight count was accepted");

    model = make_model();
    model.midi.maximum_note = 44;
    require_throws(
        [&] { tmgm::native::validate_tm_model(model); },
        "MIDI/output mismatch was accepted");

    model = make_model();
    model.training.specificity = 0.0F;
    require_throws(
        [&] { tmgm::native::validate_tm_model(model); },
        "invalid specificity was accepted");
}

void test_dataset_model_compatibility() {
    const auto model = make_model();
    tmgm::native::NativeDataset dataset;
    dataset.header.feature_count = model.dimensions.feature_count;
    dataset.header.note_count = model.dimensions.output_count;
    dataset.header.midi_min = model.midi.minimum_note;
    dataset.header.midi_max = model.midi.maximum_note;
    dataset.header.sample_rate = model.midi.audio_sample_rate;
    dataset.header.hop_size = model.midi.analysis_hop_samples;
    dataset.header.feature_fingerprint_sha256 =
        model.feature_fingerprint_sha256;

    tmgm::native::validate_tm_dataset_compatibility(dataset, model);

    auto mismatch = dataset;
    ++mismatch.header.feature_count;
    require_throws(
        [&] { tmgm::native::validate_tm_dataset_compatibility(mismatch, model); },
        "feature-count mismatch was accepted");

    mismatch = dataset;
    ++mismatch.header.note_count;
    require_throws(
        [&] { tmgm::native::validate_tm_dataset_compatibility(mismatch, model); },
        "output-count mismatch was accepted");

    mismatch = dataset;
    ++mismatch.header.midi_min;
    require_throws(
        [&] { tmgm::native::validate_tm_dataset_compatibility(mismatch, model); },
        "minimum MIDI-note mismatch was accepted");

    mismatch = dataset;
    --mismatch.header.midi_max;
    require_throws(
        [&] { tmgm::native::validate_tm_dataset_compatibility(mismatch, model); },
        "maximum MIDI-note mismatch was accepted");

    mismatch = dataset;
    mismatch.header.sample_rate = 44100U;
    require_throws(
        [&] { tmgm::native::validate_tm_dataset_compatibility(mismatch, model); },
        "sample-rate mismatch was accepted");

    mismatch = dataset;
    mismatch.header.hop_size = 512U;
    require_throws(
        [&] { tmgm::native::validate_tm_dataset_compatibility(mismatch, model); },
        "analysis-hop mismatch was accepted");

    mismatch = dataset;
    mismatch.header.feature_fingerprint_sha256.fill(0x43U);
    require_throws(
        [&] { tmgm::native::validate_tm_dataset_compatibility(mismatch, model); },
        "same-width feature-semantics mismatch was accepted");

    mismatch = dataset;
    mismatch.header.feature_fingerprint_sha256.fill(0U);
    require_throws(
        [&] { tmgm::native::validate_tm_dataset_compatibility(mismatch, model); },
        "legacy dataset was accepted without explicit opt-in");
    tmgm::native::validate_tm_dataset_compatibility(mismatch, model, true);

    auto legacy_model = model;
    legacy_model.feature_fingerprint_sha256.fill(0U);
    require_throws(
        [&] {
            tmgm::native::validate_tm_dataset_compatibility(
                dataset, legacy_model);
        },
        "legacy model was accepted without explicit opt-in");
    tmgm::native::validate_tm_dataset_compatibility(
        dataset, legacy_model, true);
}

void test_v3_rejects_fingerprint_corruption(
    const std::filesystem::path& path) {
    const auto model = make_model();
    tmgm::native::save_tm_model(path, model);
    auto bytes = read_file(path);
    bytes[192U] ^= 0x01U;
    write_file(path, bytes);
    require_throws(
        [&] { static_cast<void>(tmgm::native::load_tm_model(path)); },
        "feature-semantics fingerprint corruption escaped checksum validation");

    tmgm::native::save_tm_model(path, model);
    bytes = read_file(path);
    std::fill(bytes.begin() + 192, bytes.begin() + 224, std::uint8_t{0U});
    write_file(path, bytes);
    require_throws(
        [&] { static_cast<void>(tmgm::native::load_tm_model(path, false)); },
        "v3 accepted an all-zero feature-semantics fingerprint");
}

}  // namespace

int main() {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    const auto path = std::filesystem::temp_directory_path() /
                      ("tmgm-model-test-" + std::to_string(nonce) + ".tmgmmod");
    try {
        test_round_trip_and_wire_layout(path);
        test_checksum_covers_payload_and_metadata(path);
        test_onset_sustain_hard_negative_flag(path);
        test_v1_and_transitional_backwards_compatibility(path);
        test_current_rejects_inconsistent_hard_negative_wire_metadata(path);
        test_validation_rejects_inconsistent_model();
        test_dataset_model_compatibility();
        test_v3_rejects_fingerprint_corruption(path);
        std::filesystem::remove(path);
        std::cout << "native TM model persistence tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::filesystem::remove(path);
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
