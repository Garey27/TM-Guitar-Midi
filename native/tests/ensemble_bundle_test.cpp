#include "tmgm/ensemble_bundle.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
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

[[nodiscard]] std::vector<std::uint8_t> read_file(
    const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open bundle test fixture");
    }
    const auto length = stream.tellg();
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> result(static_cast<std::size_t>(length));
    stream.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(result.size()));
    if (!stream) {
        throw std::runtime_error("cannot read bundle test fixture");
    }
    return result;
}

void write_file(
    const std::filesystem::path& path,
    const std::vector<std::uint8_t>& value) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(
        reinterpret_cast<const char*>(value.data()),
        static_cast<std::streamsize>(value.size()));
    if (!stream) {
        throw std::runtime_error("cannot write bundle test fixture");
    }
}

}  // namespace

int main(const int argc, char** argv) {
    namespace fs = std::filesystem;
    if (argc != 2) {
        std::cerr << "usage: ensemble_bundle_test <production.tmgmbundle>\n";
        return 2;
    }
    const fs::path fixture = argv[1];
    const auto corrupt = fs::temp_directory_path() /
        "tmgmbnd_checksum_corrupt_test.tmgmbundle";
    const auto truncated = fs::temp_directory_path() /
        "tmgmbnd_truncated_test.tmgmbundle";
    try {
        const auto bundle = tmgm::native::load_ensemble_bundle(fixture);
        require(
            bundle.format_version ==
                tmgm::native::kEnsembleBundleLegacyFormatVersion,
            "production audit fixture is not recognized as legacy v1");
        require(bundle.feature_count == 7973U, "wrong production feature count");
        require(bundle.output_count == 49U, "wrong production output count");
        require(bundle.midi_min == 40 && bundle.midi_max == 88,
                "wrong production MIDI range");
        require(bundle.sample_rate == 22050U && bundle.hop_size == 256U,
                "wrong production timebase");
        require(bundle.activity.quantization == 1024U &&
                    bundle.activity.ensemble_threshold == -116,
                "wrong production activity fusion calibration");
        require(bundle.onset.quantization == 1024U &&
                    bundle.onset.ensemble_threshold == -386,
                "wrong production onset fusion calibration");
        require(bundle.members.size() == 7U, "wrong production member count");
        const std::vector<std::string> expected_ids{
            "c256", "c512", "q05", "q1", "q2", "q4", "q8"};
        const std::vector<std::uint32_t> expected_clauses{
            256U, 512U, 256U, 256U, 256U, 256U, 256U};
        for (std::size_t index = 0; index < expected_ids.size(); ++index) {
            require(bundle.members[index].identifier == expected_ids[index],
                    "wrong production member order");
            require(bundle.members[index].clause_count == expected_clauses[index],
                    "wrong production clause count");
            require(!bundle.members[index].literal_ids.empty(),
                    "production sparse literal payload is empty");
        }

        auto bytes = read_file(fixture);
        bytes[224] ^= 1U;
        write_file(corrupt, bytes);
        require_throws(
            [&] {
                static_cast<void>(
                    tmgm::native::load_ensemble_bundle(corrupt));
            },
            "corrupt bundle checksum was accepted");
        static_cast<void>(tmgm::native::load_ensemble_bundle(corrupt, false));

        bytes = read_file(fixture);
        bytes[0] ^= 1U;
        write_file(corrupt, bytes);
        require_throws(
            [&] {
                static_cast<void>(
                    tmgm::native::load_ensemble_bundle(corrupt, false));
            },
            "wrong bundle magic was accepted");

        bytes = read_file(fixture);
        bytes[8] = 3U;
        write_file(corrupt, bytes);
        require_throws(
            [&] {
                static_cast<void>(
                    tmgm::native::load_ensemble_bundle(corrupt, false));
            },
            "unknown bundle version was accepted");

        bytes = read_file(fixture);
        bytes[84] = 1U;
        write_file(corrupt, bytes);
        require_throws(
            [&] {
                static_cast<void>(
                    tmgm::native::load_ensemble_bundle(corrupt, false));
            },
            "non-zero reserved header value was accepted");

        bytes = read_file(fixture);
        // First member literal-ID offset must follow its clause-offset section.
        std::fill(
            bytes.begin() + 256 + 152,
            bytes.begin() + 256 + 160,
            std::uint8_t{0U});
        write_file(corrupt, bytes);
        require_throws(
            [&] {
                static_cast<void>(
                    tmgm::native::load_ensemble_bundle(corrupt, false));
            },
            "overlapping/non-canonical member sections were accepted");

        bytes = read_file(fixture);
        bytes.pop_back();
        write_file(truncated, bytes);
        require_throws(
            [&] {
                static_cast<void>(
                    tmgm::native::load_ensemble_bundle(truncated));
            },
            "truncated bundle was accepted");

        fs::remove(corrupt);
        fs::remove(truncated);
        std::cout << "TMGMBND production loader tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        fs::remove(corrupt);
        fs::remove(truncated);
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
