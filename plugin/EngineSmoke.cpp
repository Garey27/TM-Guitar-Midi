#include "TmRealtimeEngine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct WavAudio {
    std::uint32_t sample_rate = 0U;
    std::vector<float> mono;
};

[[nodiscard]] std::uint16_t u16(const std::uint8_t* p) noexcept {
    return static_cast<std::uint16_t>(p[0])
        | static_cast<std::uint16_t>(p[1]) << 8U;
}

[[nodiscard]] std::uint32_t u32(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0])
        | static_cast<std::uint32_t>(p[1]) << 8U
        | static_cast<std::uint32_t>(p[2]) << 16U
        | static_cast<std::uint32_t>(p[3]) << 24U;
}

[[nodiscard]] WavAudio read_wav(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open WAV");
    }
    std::vector<std::uint8_t> bytes{
        std::istreambuf_iterator<char>(stream), {}};
    if (bytes.size() < 44U
        || std::memcmp(bytes.data(), "RIFF", 4U) != 0
        || std::memcmp(bytes.data() + 8U, "WAVE", 4U) != 0) {
        throw std::runtime_error("not a RIFF/WAVE file");
    }
    std::uint16_t format = 0U;
    std::uint16_t channels = 0U;
    std::uint16_t bits = 0U;
    std::uint32_t sample_rate = 0U;
    const std::uint8_t* data = nullptr;
    std::size_t data_size = 0U;
    for (std::size_t offset = 12U; offset + 8U <= bytes.size();) {
        const auto size = static_cast<std::size_t>(u32(bytes.data() + offset + 4U));
        const auto payload = offset + 8U;
        if (payload + size > bytes.size()) {
            throw std::runtime_error("truncated WAV chunk");
        }
        if (std::memcmp(bytes.data() + offset, "fmt ", 4U) == 0 && size >= 16U) {
            format = u16(bytes.data() + payload);
            channels = u16(bytes.data() + payload + 2U);
            sample_rate = u32(bytes.data() + payload + 4U);
            bits = u16(bytes.data() + payload + 14U);
        } else if (std::memcmp(bytes.data() + offset, "data", 4U) == 0) {
            data = bytes.data() + payload;
            data_size = size;
        }
        offset = payload + size + (size & 1U);
    }
    if (data == nullptr || channels == 0U || sample_rate == 0U) {
        throw std::runtime_error("WAV lacks fmt/data");
    }
    const std::size_t bytes_per_sample = bits / 8U;
    const std::size_t frame_bytes = bytes_per_sample * channels;
    if (bytes_per_sample == 0U || frame_bytes == 0U
        || data_size % frame_bytes != 0U) {
        throw std::runtime_error("unsupported WAV geometry");
    }
    WavAudio result{sample_rate, {}};
    result.mono.resize(data_size / frame_bytes);
    for (std::size_t frame = 0U; frame < result.mono.size(); ++frame) {
        double sum = 0.0;
        for (std::size_t channel = 0U; channel < channels; ++channel) {
            const auto* source = data + frame * frame_bytes
                + channel * bytes_per_sample;
            float value = 0.0F;
            if (format == 3U && bits == 32U) {
                std::memcpy(&value, source, sizeof(value));
            } else if (format == 1U && bits == 16U) {
                const auto raw = static_cast<std::int16_t>(u16(source));
                value = static_cast<float>(raw) / 32768.0F;
            } else if (format == 1U && bits == 24U) {
                std::int32_t raw = static_cast<std::int32_t>(source[0])
                    | static_cast<std::int32_t>(source[1]) << 8U
                    | static_cast<std::int32_t>(source[2]) << 16U;
                if ((raw & 0x00800000) != 0) {
                    raw |= static_cast<std::int32_t>(0xFF000000U);
                }
                value = static_cast<float>(raw) / 8388608.0F;
            } else {
                throw std::runtime_error("unsupported WAV encoding");
            }
            sum += value;
        }
        result.mono[frame] = static_cast<float>(sum / channels);
    }
    return result;
}

struct TimedEvent {
    tmgm::preview::MidiEvent event;
    std::uint64_t absolute_sample = 0U;
    friend bool operator==(const TimedEvent& left, const TimedEvent& right) {
        return left.event.kind == right.event.kind
            && left.event.pitch == right.event.pitch
            && left.event.velocity == right.event.velocity
            && left.absolute_sample == right.absolute_sample;
    }
};

struct Sink {
    std::vector<TimedEvent>* events = nullptr;
    std::uint64_t block_start = 0U;
};

void collect(void* user, const tmgm::preview::MidiEvent& event) noexcept {
    auto& sink = *static_cast<Sink*>(user);
    sink.events->push_back({event, sink.block_start + event.sample_offset});
}

[[nodiscard]] std::vector<TimedEvent> render(
    const std::filesystem::path& package,
    const WavAudio& audio,
    const float gain_db,
    const std::vector<std::size_t>& blocks) {
    tmgm::preview::TmRealtimeEngine engine;
    std::string error;
    if (!engine.load_package(package, error)) {
        throw std::runtime_error("load failed: " + error);
    }
    const auto max_block = *std::max_element(blocks.begin(), blocks.end());
    if (!engine.prepare(audio.sample_rate, max_block, error)) {
        throw std::runtime_error("prepare failed: " + error);
    }
    std::vector<TimedEvent> events;
    Sink sink{&events, 0U};
    const float gain = std::pow(10.0F, gain_db / 20.0F);
    std::size_t offset = 0U;
    std::size_t block_index = 0U;
    while (offset < audio.mono.size()) {
        const auto count = std::min(
            blocks[block_index % blocks.size()], audio.mono.size() - offset);
        sink.block_start = offset;
        if (engine.process_block(audio.mono.data() + offset, count, gain,
                &collect, &sink)
            != tmgm::preview::EngineProcessStatus::success) {
            throw std::runtime_error("realtime processing failed");
        }
        offset += count;
        ++block_index;
    }
    std::array<float, 4096> silence{};
    for (int tail = 0; tail < 12; ++tail) {
        sink.block_start = offset;
        if (engine.process_block(silence.data(), silence.size(), gain,
                &collect, &sink)
            != tmgm::preview::EngineProcessStatus::success) {
            throw std::runtime_error("tail processing failed");
        }
        offset += silence.size();
    }
    std::array<tmgm::preview::MidiEvent, tmgm::preview::kPitchCount> panic{};
    const auto panic_count = engine.release_active_notes(panic);
    for (std::size_t index = 0U; index < panic_count; ++index) {
        events.push_back({panic[index], offset});
    }
    return events;
}

void assert_balanced(const std::vector<TimedEvent>& events) {
    std::array<bool, 128> active{};
    std::size_t note_ons = 0U;
    for (const auto& timed : events) {
        const auto pitch = timed.event.pitch;
        if (timed.event.kind == tmgm::preview::MidiEvent::Kind::note_on) {
            if (active[pitch]) {
                throw std::runtime_error("NoteOn without preceding NoteOff");
            }
            active[pitch] = true;
            ++note_ons;
        } else {
            if (!active[pitch]) {
                throw std::runtime_error("orphan NoteOff");
            }
            active[pitch] = false;
        }
    }
    if (note_ons == 0U
        || std::any_of(active.begin(), active.end(), [](bool value) { return value; })) {
        throw std::runtime_error("empty or stuck-note render");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: engine-smoke PACKAGE WAV ACOUSTIC_GAIN_DB\n";
            return 2;
        }
        const std::filesystem::path package = argv[1];
        const auto audio = read_wav(argv[2]);
        const float gain_db = std::stof(argv[3]);
        const auto fixed = render(package, audio, gain_db, {512U});
        const auto partitioned = render(
            package, audio, gain_db, {1U, 17U, 64U, 255U, 511U, 7U, 1033U});
        assert_balanced(fixed);
        assert_balanced(partitioned);
        if (fixed != partitioned) {
            const auto common = std::min(fixed.size(), partitioned.size());
            std::size_t mismatch = 0U;
            while (mismatch < common && fixed[mismatch] == partitioned[mismatch]) {
                ++mismatch;
            }
            std::cerr << "fixed_events=" << fixed.size()
                      << " partitioned_events=" << partitioned.size()
                      << " first_mismatch=" << mismatch << '\n';
            if (mismatch < fixed.size()) {
                const auto& value = fixed[mismatch];
                std::cerr << "fixed kind="
                          << static_cast<int>(value.event.kind)
                          << " pitch=" << static_cast<int>(value.event.pitch)
                          << " velocity=" << static_cast<int>(value.event.velocity)
                          << " absolute=" << value.absolute_sample
                          << " local=" << value.event.sample_offset << '\n';
            }
            if (mismatch < partitioned.size()) {
                const auto& value = partitioned[mismatch];
                std::cerr << "partitioned kind="
                          << static_cast<int>(value.event.kind)
                          << " pitch=" << static_cast<int>(value.event.pitch)
                          << " velocity=" << static_cast<int>(value.event.velocity)
                          << " absolute=" << value.absolute_sample
                          << " local=" << value.event.sample_offset << '\n';
            }
            throw std::runtime_error("block-partition event mismatch");
        }
        const auto note_ons = std::count_if(
            fixed.begin(), fixed.end(), [](const TimedEvent& event) {
                return event.event.kind
                    == tmgm::preview::MidiEvent::Kind::note_on;
            });
        std::cout << "sample_rate=" << audio.sample_rate
                  << " samples=" << audio.mono.size()
                  << " note_ons=" << note_ons
                  << " events=" << fixed.size() << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
