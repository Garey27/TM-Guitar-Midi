// Modified file: headless posterior/event exporter added by the TM Guitar MIDI
// project for offline teacher generation. Based on NeuralNote at commit
// f979e51dfeab54d5921858af39403308ab06e60c; Apache-2.0 license retained in
// LICENSE-NeuralNote-Apache-2.0.txt. This tool is not linked into the plugin.

#include <JuceHeader.h>


#include "AudioUtils.h"
#include "BasicPitch.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef NEURALNOTE_UPSTREAM_COMMIT
#define NEURALNOTE_UPSTREAM_COMMIT "unknown"
#endif

namespace {

constexpr std::uint32_t kPosteriorVersion = 1U;
constexpr std::uint32_t kQuantizationLevels = 255U;
constexpr std::size_t kPosteriorFrameBytes =
    static_cast<std::size_t>(NUM_FREQ_OUT) * 2U
    + static_cast<std::size_t>(NUM_FREQ_IN);

#pragma pack(push, 1)
struct PosteriorHeader {
    std::array<char, 4> magic{'N', 'N', 'P', 'G'};
    std::uint32_t version{kPosteriorVersion};
    std::uint32_t headerBytes{256U};
    std::uint32_t frameBytes{static_cast<std::uint32_t>(kPosteriorFrameBytes)};
    std::uint64_t frameCount{};
    std::uint32_t noteBins{NUM_FREQ_OUT};
    std::uint32_t onsetBins{NUM_FREQ_OUT};
    std::uint32_t contourBins{NUM_FREQ_IN};
    std::uint32_t hopSize{FFT_HOP};
    std::uint32_t sampleRate{AUDIO_SAMPLE_RATE};
    std::uint32_t quantizationLevels{kQuantizationLevels};
    float noteSensitivity{0.7F};
    float splitSensitivity{0.5F};
    float minimumNoteDurationMs{125.0F};
    double sourceSampleRate{};
    std::uint32_t sourceChannels{};
    std::uint64_t sourceSamples{};
    std::uint64_t sourceBytes{};
    std::uint64_t sourceFnv1a64{};
    std::array<char, 41> neuralNoteCommit{};
    std::array<std::byte, 119> reserved{};
};
#pragma pack(pop)

static_assert(sizeof(PosteriorHeader) == 256U);
static_assert(kPosteriorFrameBytes == 440U);

struct Job {
    std::string id;
    std::filesystem::path input;
    std::filesystem::path outputRelative;
};

struct Options {
    std::filesystem::path outputRoot;
    std::filesystem::path inputList;
    std::vector<std::filesystem::path> inputs;
    float noteSensitivity{0.7F};
    float splitSensitivity{0.5F};
    float minimumNoteDurationMs{125.0F};
    std::size_t limit{std::numeric_limits<std::size_t>::max()};
    bool force{};
};

[[nodiscard]] std::vector<std::string> splitTabs(std::string line)
{
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<std::string> fields;
    std::size_t begin{};
    while (begin <= line.size()) {
        const std::size_t end = line.find('\t', begin);
        fields.emplace_back(line.substr(begin,
            end == std::string::npos ? std::string::npos : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1U;
    }
    return fields;
}

[[nodiscard]] float parseFloat(
    const std::string_view text,
    const std::string_view name)
{
    std::size_t consumed{};
    const float value = std::stof(std::string{text}, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::runtime_error("Invalid " + std::string{name});
    }
    return value;
}

[[nodiscard]] std::size_t parseSize(
    const std::string_view text,
    const std::string_view name)
{
    std::size_t consumed{};
    const auto value = std::stoull(std::string{text}, &consumed);
    if (consumed != text.size()
        || value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Invalid " + std::string{name});
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] Options parseOptions(const int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        const auto value = [&]() -> std::string_view {
            if (++index >= argc) {
                throw std::runtime_error(
                    "Missing value after " + std::string{argument});
            }
            return argv[index];
        };
        if (argument == "--output-root") {
            options.outputRoot = value();
        } else if (argument == "--input-list") {
            options.inputList = value();
        } else if (argument == "--input") {
            options.inputs.emplace_back(value());
        } else if (argument == "--note-sensitivity") {
            options.noteSensitivity = parseFloat(value(), argument);
        } else if (argument == "--split-sensitivity") {
            options.splitSensitivity = parseFloat(value(), argument);
        } else if (argument == "--minimum-note-ms") {
            options.minimumNoteDurationMs = parseFloat(value(), argument);
        } else if (argument == "--limit") {
            options.limit = parseSize(value(), argument);
        } else if (argument == "--force") {
            options.force = true;
        } else {
            throw std::runtime_error(
                "Unknown option: " + std::string{argument});
        }
    }
    if (options.outputRoot.empty()
        || (options.inputList.empty() && options.inputs.empty())
        || (!options.inputList.empty() && !options.inputs.empty())) {
        throw std::runtime_error(
            "Usage: NeuralNoteBatch --output-root DIR"
            " (--input WAV [...] | --input-list jobs.tsv)"
            " [--note-sensitivity .7] [--split-sensitivity .5]"
            " [--minimum-note-ms 125] [--limit N] [--force]");
    }
    if (options.noteSensitivity < 0.05F || options.noteSensitivity > 0.95F
        || options.splitSensitivity < 0.05F
        || options.splitSensitivity > 0.95F
        || options.minimumNoteDurationMs < 1.0F) {
        throw std::runtime_error("NeuralNote parameter is outside its range");
    }
    return options;
}

void validateRelativePath(const std::filesystem::path& path)
{
    if (path.empty() || path.is_absolute()) {
        throw std::runtime_error("Output path must be relative: "
            + path.string());
    }
    for (const auto& component : path.lexically_normal()) {
        if (component == "..") {
            throw std::runtime_error("Output path escapes root: "
                + path.string());
        }
    }
}

[[nodiscard]] std::vector<Job> readJobs(const Options& options)
{
    std::vector<Job> jobs;
    if (!options.inputList.empty()) {
        std::ifstream stream(options.inputList);
        if (!stream) {
            throw std::runtime_error("Cannot open input list: "
                + options.inputList.string());
        }
        std::string line;
        if (!std::getline(stream, line)
            || splitTabs(line)
                != std::vector<std::string>{"id", "input", "output_rel"}) {
            throw std::runtime_error(
                "Input-list header must be: id<TAB>input<TAB>output_rel");
        }
        std::size_t lineNumber = 1U;
        while (std::getline(stream, line)) {
            ++lineNumber;
            if (line.empty()) continue;
            const auto fields = splitTabs(line);
            if (fields.size() != 3U || fields[0].empty()
                || fields[1].empty() || fields[2].empty()) {
                throw std::runtime_error("Malformed input-list line "
                    + std::to_string(lineNumber));
            }
            Job job{fields[0], fields[1], fields[2]};
            validateRelativePath(job.outputRelative);
            jobs.push_back(std::move(job));
        }
    } else {
        for (const auto& input : options.inputs) {
            Job job;
            job.id = input.stem().string();
            job.input = input;
            job.outputRelative = input.stem();
            validateRelativePath(job.outputRelative);
            jobs.push_back(std::move(job));
        }
    }
    if (jobs.empty()) throw std::runtime_error("Input list contains no jobs");
    if (jobs.size() > options.limit) jobs.resize(options.limit);
    return jobs;
}

[[nodiscard]] std::filesystem::path withSuffix(
    const std::filesystem::path& base,
    const std::string_view suffix)
{
    auto result = base;
    result += suffix;
    return result;
}

[[nodiscard]] std::uint64_t fnv1a64File(
    const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot hash input: " + path.string());
    }
    std::uint64_t hash = 14'695'981'039'346'656'037ULL;
    std::array<char, 1U << 16U> buffer{};
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        for (std::streamsize index = 0; index < count; ++index) {
            hash ^= static_cast<std::uint8_t>(buffer[static_cast<std::size_t>(index)]);
            hash *= 1'099'511'628'211ULL;
        }
    }
    return hash;
}

[[nodiscard]] std::uint8_t quantizeProbability(const float value) noexcept
{
    if (!std::isfinite(value)) return 0U;
    return static_cast<std::uint8_t>(std::lround(
        std::clamp(value, 0.0F, 1.0F)
        * static_cast<float>(kQuantizationLevels)));
}

void writePosteriorgram(
    const std::filesystem::path& path,
    const BasicPitch& model,
    const Options& options,
    const double sourceSampleRate,
    const std::uint32_t sourceChannels,
    const std::uint64_t sourceSamples,
    const std::uint64_t sourceBytes,
    const std::uint64_t sourceHash)
{
    const auto& notes = model.getNotePosteriorgram();
    const auto& onsets = model.getOnsetPosteriorgram();
    const auto& contours = model.getContourPosteriorgram();
    if (notes.size() != model.getNumFrames()
        || onsets.size() != notes.size() || contours.size() != notes.size()) {
        throw std::runtime_error("NeuralNote posteriorgram dimensions disagree");
    }

    PosteriorHeader header;
    header.frameCount = static_cast<std::uint64_t>(notes.size());
    header.noteSensitivity = options.noteSensitivity;
    header.splitSensitivity = options.splitSensitivity;
    header.minimumNoteDurationMs = options.minimumNoteDurationMs;
    header.sourceSampleRate = sourceSampleRate;
    header.sourceChannels = sourceChannels;
    header.sourceSamples = sourceSamples;
    header.sourceBytes = sourceBytes;
    header.sourceFnv1a64 = sourceHash;
    constexpr std::string_view commit{NEURALNOTE_UPSTREAM_COMMIT};
    std::copy_n(commit.data(),
        std::min(commit.size(), header.neuralNoteCommit.size() - 1U),
        header.neuralNoteCommit.data());

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Cannot create posteriorgram: "
            + path.string());
    }
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    std::array<std::uint8_t, kPosteriorFrameBytes> frame{};
    for (std::size_t index = 0U; index < notes.size(); ++index) {
        if (notes[index].size() != NUM_FREQ_OUT
            || onsets[index].size() != NUM_FREQ_OUT
            || contours[index].size() != NUM_FREQ_IN) {
            throw std::runtime_error("NeuralNote posteriorgram frame is ragged");
        }
        std::size_t destination{};
        for (const float value : notes[index]) {
            frame[destination++] = quantizeProbability(value);
        }
        for (const float value : onsets[index]) {
            frame[destination++] = quantizeProbability(value);
        }
        for (const float value : contours[index]) {
            frame[destination++] = quantizeProbability(value);
        }
        stream.write(reinterpret_cast<const char*>(frame.data()),
            static_cast<std::streamsize>(frame.size()));
    }
    if (!stream) {
        throw std::runtime_error("Cannot finalize posteriorgram: "
            + path.string());
    }
}

void writeEvents(
    const std::filesystem::path& path,
    const std::vector<Notes::Event>& events)
{
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Cannot create event TSV: " + path.string());
    }
    stream << "start_sec\tend_sec\tpitch\tamplitude\tstart_frame"
              "\tend_frame\tbends_thirds_of_semitone\n";
    stream << std::setprecision(17);
    for (const auto& event : events) {
        stream << event.startTime << '\t' << event.endTime << '\t'
               << event.pitch << '\t' << event.amplitude << '\t'
               << event.startFrame << '\t' << event.endFrame << '\t';
        for (std::size_t index = 0U; index < event.bends.size(); ++index) {
            if (index != 0U) stream << ',';
            stream << event.bends[index];
        }
        stream << '\n';
    }
    if (!stream) throw std::runtime_error("Cannot finalize event TSV");
}

void writeMidi(
    const std::filesystem::path& path,
    const std::vector<Notes::Event>& events)
{
    constexpr double ticksPerSecond = 1'000.0;
    juce::MidiMessageSequence sequence;
    for (const auto& event : events) {
        auto noteOn = juce::MidiMessage::noteOn(1, event.pitch,
            static_cast<float>(std::clamp(event.amplitude, 0.0, 1.0)));
        noteOn.setTimeStamp(event.startTime * ticksPerSecond);
        sequence.addEvent(noteOn);
        auto noteOff = juce::MidiMessage::noteOff(1, event.pitch);
        noteOff.setTimeStamp(event.endTime * ticksPerSecond);
        sequence.addEvent(noteOff);
    }
    sequence.sort();
    sequence.updateMatchedPairs();
    juce::MidiFile midi;
    midi.setSmpteTimeFormat(25, 40);
    midi.addTrack(sequence);
    const juce::File file(path.string());
    file.deleteFile();
    juce::FileOutputStream stream(file);
    if (!stream.openedOk() || !midi.writeTo(stream)) {
        throw std::runtime_error("Cannot write MIDI: " + path.string());
    }
    stream.flush();
}

void writeMetadata(
    const std::filesystem::path& path,
    const Job& job,
    const Options& options,
    const double sourceSampleRate,
    const int sourceChannels,
    const int sourceSamples,
    const int downsampledSamples,
    const std::uint64_t sourceBytes,
    const std::uint64_t sourceHash,
    const std::size_t frameCount,
    const std::size_t eventCount,
    const double elapsedSeconds)
{
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Cannot create metadata: " + path.string());
    }
    stream << std::setprecision(17)
           << "key\tvalue\n"
           << "id\t" << job.id << '\n'
           << "input\t" << job.input.string() << '\n'
           << "neuralnote_commit\t" << NEURALNOTE_UPSTREAM_COMMIT << '\n'
           << "source_fnv1a64\t0x" << std::hex << sourceHash << std::dec << '\n'
           << "source_bytes\t" << sourceBytes << '\n'
           << "source_sample_rate\t" << sourceSampleRate << '\n'
           << "source_channels\t" << sourceChannels << '\n'
           << "source_samples\t" << sourceSamples << '\n'
           << "channel_policy\tNeuralNote_GUI_channel_0\n"
           << "teacher_sample_rate\t" << AUDIO_SAMPLE_RATE << '\n'
           << "teacher_hop\t" << FFT_HOP << '\n'
           << "downsampled_samples\t" << downsampledSamples << '\n'
           << "frames\t" << frameCount << '\n'
           << "events\t" << eventCount << '\n'
           << "note_sensitivity\t" << options.noteSensitivity << '\n'
           << "split_sensitivity\t" << options.splitSensitivity << '\n'
           << "minimum_note_ms\t" << options.minimumNoteDurationMs << '\n'
           << "posterior_quantization\tuint8_probability_0_255\n"
           << "elapsed_seconds\t" << elapsedSeconds << '\n';
    if (!stream) throw std::runtime_error("Cannot finalize metadata");
}

void replaceFile(
    const std::filesystem::path& temporary,
    const std::filesystem::path& final)
{
    std::error_code error;
    std::filesystem::remove(final, error);
    error.clear();
    std::filesystem::rename(temporary, final, error);
    if (error) {
        throw std::runtime_error("Cannot publish output " + final.string()
            + ": " + error.message());
    }
}

void processJob(
    BasicPitch& model,
    const Job& job,
    const Options& options,
    const std::size_t jobIndex,
    const std::size_t jobCount)
{
    if (!std::filesystem::is_regular_file(job.input)) {
        throw std::runtime_error("Input is not a file: " + job.input.string());
    }
    const auto base = options.outputRoot / job.outputRelative;
    const auto posteriorPath = withSuffix(base, ".nnpg");
    const auto eventsPath = withSuffix(base, ".events.tsv");
    const auto midiPath = withSuffix(base, ".mid");
    const auto metadataPath = withSuffix(base, ".meta.tsv");
    if (!options.force
        && std::filesystem::is_regular_file(posteriorPath)
        && std::filesystem::is_regular_file(eventsPath)
        && std::filesystem::is_regular_file(midiPath)
        && std::filesystem::is_regular_file(metadataPath)) {
        std::cout << '[' << jobIndex + 1U << '/' << jobCount << "] skip "
                  << job.id << '\n';
        return;
    }
    std::filesystem::create_directories(base.parent_path());
    const auto posteriorTemporary = withSuffix(posteriorPath, ".tmp");
    const auto eventsTemporary = withSuffix(eventsPath, ".tmp");
    const auto midiTemporary = withSuffix(midiPath, ".tmp");
    const auto metadataTemporary = withSuffix(metadataPath, ".tmp");
    std::filesystem::remove(posteriorTemporary);
    std::filesystem::remove(eventsTemporary);
    std::filesystem::remove(midiTemporary);
    std::filesystem::remove(metadataTemporary);

    std::cout << '[' << jobIndex + 1U << '/' << jobCount << "] transcribe "
              << job.id << " <- " << job.input << std::endl;
    const auto started = std::chrono::steady_clock::now();
    juce::AudioBuffer<float> source;
    double sourceSampleRate{};
    if (!AudioUtils::loadAudioFile(
            juce::File(job.input.string()), source, sourceSampleRate)
        || source.getNumChannels() <= 0 || source.getNumSamples() <= 0) {
        throw std::runtime_error("NeuralNote cannot load audio: "
            + job.input.string());
    }
    const int sourceChannels = source.getNumChannels();
    const int sourceSamples = source.getNumSamples();
    juce::AudioBuffer<float> downsampled;
    AudioUtils::resampleBuffer(
        source, downsampled, sourceSampleRate, BASIC_PITCH_SAMPLE_RATE);
    if (downsampled.getNumSamples() <= 0
        || downsampled.getNumSamples() > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Downsampled audio size is unsupported");
    }

    model.reset();
    model.setParameters(options.noteSensitivity,
        options.splitSensitivity, options.minimumNoteDurationMs);
    model.transcribeToMIDI(
        downsampled.getWritePointer(0), downsampled.getNumSamples());

    const std::uint64_t sourceBytes = std::filesystem::file_size(job.input);
    const std::uint64_t sourceHash = fnv1a64File(job.input);
    writePosteriorgram(posteriorTemporary, model, options,
        sourceSampleRate, static_cast<std::uint32_t>(sourceChannels),
        static_cast<std::uint64_t>(sourceSamples), sourceBytes, sourceHash);
    writeEvents(eventsTemporary, model.getNoteEvents());
    writeMidi(midiTemporary, model.getNoteEvents());
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    writeMetadata(metadataTemporary, job, options, sourceSampleRate,
        sourceChannels, sourceSamples, downsampled.getNumSamples(),
        sourceBytes, sourceHash, model.getNumFrames(),
        model.getNoteEvents().size(), elapsed);

    replaceFile(posteriorTemporary, posteriorPath);
    replaceFile(eventsTemporary, eventsPath);
    replaceFile(midiTemporary, midiPath);
    replaceFile(metadataTemporary, metadataPath);
    const double audioSeconds =
        static_cast<double>(sourceSamples) / sourceSampleRate;
    std::cout << "  frames=" << model.getNumFrames()
              << " events=" << model.getNoteEvents().size()
              << " audio=" << std::fixed << std::setprecision(2)
              << audioSeconds << "s elapsed=" << elapsed
              << "s speed=" << (elapsed > 0.0 ? audioSeconds / elapsed : 0.0)
              << "x\n";
}

} // namespace

int main(const int argc, char** argv)
{
    try {
        const Options options = parseOptions(argc, argv);
        const auto jobs = readJobs(options);
        std::filesystem::create_directories(options.outputRoot);
        std::cout << "NeuralNoteBatch commit=" << NEURALNOTE_UPSTREAM_COMMIT
                  << " jobs=" << jobs.size()
                  << " note/split/min_ms=" << options.noteSensitivity << '/'
                  << options.splitSensitivity << '/'
                  << options.minimumNoteDurationMs << '\n';
        BasicPitch model;
        std::size_t failures{};
        for (std::size_t index = 0U; index < jobs.size(); ++index) {
            try {
                processJob(model, jobs[index], options, index, jobs.size());
            } catch (const std::exception& error) {
                ++failures;
                std::cerr << "FAILED [" << index + 1U << '/' << jobs.size()
                          << "] " << jobs[index].id << ": "
                          << error.what() << '\n';
            }
        }
        if (failures != 0U) {
            std::cerr << "Completed with " << failures << " failed job(s)\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
