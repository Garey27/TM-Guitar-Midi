#include <juce_audio_utils/juce_audio_utils.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

#if defined(TMGM_HOST_TEST_VST3)
using TestedPluginFormat = juce::VST3PluginFormat;
constexpr std::string_view kExpectedFormat = "VST3";
constexpr std::string_view kExpectedLayoutSummary = "mono+stereo";
#elif defined(TMGM_HOST_TEST_LV2)
using TestedPluginFormat = juce::LV2PluginFormat;
constexpr std::string_view kExpectedFormat = "LV2";
constexpr std::string_view kExpectedLayoutSummary = "mono";
constexpr std::string_view kExpectedPluginIdentifier =
    "urn:tmgm:plugin:tm-guitar-midi";
#else
#error "Select exactly one host-smoke plug-in format"
#endif

constexpr std::string_view kExpectedPluginName = "TM Guitar MIDI";

void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

juce::AudioProcessorParameter& parameter_named(
    juce::AudioProcessor& processor,
    const juce::String& name) {
    for (auto* parameter : processor.getParameters()) {
        if (parameter != nullptr && parameter->getName(128) == name) {
            return *parameter;
        }
    }
    throw std::runtime_error(
        "missing plug-in parameter: " + name.toStdString());
}

void require_audio_is_zero(const juce::AudioBuffer<float>& audio) {
    for (int channel = 0; channel < audio.getNumChannels(); ++channel) {
        const auto* data = audio.getReadPointer(channel);
        for (int sample = 0; sample < audio.getNumSamples(); ++sample) {
            require(data[sample] == 0.0F,
                    "Dry Audio default Off did not mute plug-in audio output");
        }
    }
}

void require_audio_equal(
    const juce::AudioBuffer<float>& actual,
    const juce::AudioBuffer<float>& expected) {
    require(actual.getNumChannels() == expected.getNumChannels()
                && actual.getNumSamples() == expected.getNumSamples(),
            "dry passthrough geometry differs");
    for (int channel = 0; channel < actual.getNumChannels(); ++channel) {
        for (int sample = 0; sample < actual.getNumSamples(); ++sample) {
            require(actual.getSample(channel, sample)
                        == expected.getSample(channel, sample),
                    "Dry Audio On modified the host audio buffer");
        }
    }
}

struct MidiAudit {
    std::array<bool, 128> active{};
    std::size_t note_ons = 0U;
    std::size_t note_offs = 0U;
    std::size_t duplicate_ons = 0U;
    std::size_t orphan_offs = 0U;
    std::size_t maximum_polyphony = 0U;

    void add(const juce::MidiBuffer& midi) {
        for (const auto metadata : midi) {
            const auto message = metadata.getMessage();
            if (!message.isNoteOn() && !message.isNoteOff()) {
                continue;
            }
            const auto pitch = static_cast<std::size_t>(message.getNoteNumber());
            require(pitch >= 40U && pitch <= 88U,
                    "plug-in emitted a pitch outside MIDI 40..88");
            if (message.isNoteOn()) {
                ++note_ons;
                if (active[pitch]) ++duplicate_ons;
                active[pitch] = true;
                require(message.getVelocity() > 0.0F,
                        "Note On velocity is zero");
            } else {
                ++note_offs;
                if (!active[pitch]) ++orphan_offs;
                active[pitch] = false;
            }
            maximum_polyphony = std::max(
                maximum_polyphony,
                static_cast<std::size_t>(std::count(
                    active.begin(), active.end(), true)));
        }
    }

    [[nodiscard]] std::size_t active_count() const {
        return static_cast<std::size_t>(
            std::count(active.begin(), active.end(), true));
    }
};

void set_mono_layout(juce::AudioProcessor& processor) {
    auto layout = processor.getBusesLayout();
    require(!layout.inputBuses.isEmpty() && !layout.outputBuses.isEmpty(),
            "plug-in exposes no main audio buses");
    layout.inputBuses.getReference(0) = juce::AudioChannelSet::mono();
    layout.outputBuses.getReference(0) = juce::AudioChannelSet::mono();
    require(processor.setBusesLayout(layout),
            "plug-in rejected its declared mono layout");
    require(processor.getTotalNumInputChannels() == 1
                && processor.getTotalNumOutputChannels() == 1,
            "plug-in mono bus geometry differs");
}

void test_stereo_layout(juce::AudioProcessor& processor) {
    auto layout = processor.getBusesLayout();
    layout.inputBuses.getReference(0) = juce::AudioChannelSet::stereo();
    layout.outputBuses.getReference(0) = juce::AudioChannelSet::stereo();
    require(processor.setBusesLayout(layout),
            "plug-in rejected supported stereo layout");
    require(processor.getTotalNumInputChannels() == 2
                && processor.getTotalNumOutputChannels() == 2,
            "plug-in stereo bus geometry differs");
    set_mono_layout(processor);
}

void test_dry_and_status(juce::AudioProcessor& processor) {
    processor.setRateAndBufferSizeDetails(48000.0, 512);
    processor.prepareToPlay(48000.0, 512);
    for (auto* parameter : processor.getParameters()) {
        if (parameter != nullptr) {
            std::cerr << "parameter " << parameter->getName(128)
                      << "=" << parameter->getValue()
                      << " text=" << parameter->getCurrentValueAsText()
                      << '\n';
        }
    }
    require(parameter_named(processor, "Model Loaded (status)").getValue()
                >= 0.5F,
            "host instance did not authenticate bundled TM model");
    require(parameter_named(
                processor, "Sample Rate 44.1/48 kHz (status)").getValue()
                >= 0.5F,
            "host instance rejected 48 kHz");
    auto& dry = parameter_named(processor, "Dry Audio");
    require(dry.getValue() < 0.5F, "Dry Audio does not default Off");

    juce::AudioBuffer<float> audio{1, 512};
    for (int sample = 0; sample < audio.getNumSamples(); ++sample) {
        audio.setSample(0, sample,
            static_cast<float>(sample - 256) * 1.0e-9F);
    }
    juce::MidiBuffer midi;
    processor.processBlock(audio, midi);
    require_audio_is_zero(audio);

    dry.setValueNotifyingHost(1.0F);
    for (int sample = 0; sample < audio.getNumSamples(); ++sample) {
        audio.setSample(0, sample,
            static_cast<float>(sample - 256) * 1.0e-9F);
    }
    juce::AudioBuffer<float> reference;
    reference.makeCopyOf(audio);
    midi.clear();
    processor.processBlock(audio, midi);
    require_audio_equal(audio, reference);
    dry.setValueNotifyingHost(0.0F);
    processor.releaseResources();
}

MidiAudit process_wav(
    juce::AudioProcessor& processor,
    const juce::File& wav) {
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    auto reader = std::unique_ptr<juce::AudioFormatReader>(
        formats.createReaderFor(wav));
    require(reader != nullptr, "cannot open host smoke WAV");
    require(std::llround(reader->sampleRate) == 44100,
            "host smoke WAV must be 44.1 kHz");

    set_mono_layout(processor);
    processor.setRateAndBufferSizeDetails(44100.0, 512);
    processor.prepareToPlay(44100.0, 512);
    auto& dry = parameter_named(processor, "Dry Audio");
    dry.setValueNotifyingHost(0.0F);

    juce::AudioBuffer<float> block{1, 512};
    juce::MidiBuffer midi;
    MidiAudit audit;
    juce::int64 position = 0;
    while (position < reader->lengthInSamples) {
        const int count = static_cast<int>(std::min<juce::int64>(
            block.getNumSamples(), reader->lengthInSamples - position));
        block.clear();
        require(reader->read(
                    &block, 0, count, position, true, false),
                "failed reading host smoke WAV block");
        midi.clear();
        processor.processBlock(block, midi);
        audit.add(midi);
        position += count;
    }
    // Let the causal activity/release state settle, then exercise host bypass
    // panic as REAPER does when an FX is disabled.
    for (int tail = 0; tail < 100; ++tail) {
        block.clear();
        midi.clear();
        processor.processBlock(block, midi);
        audit.add(midi);
    }
    block.clear();
    midi.clear();
    processor.processBlockBypassed(block, midi);
    audit.add(midi);
    processor.releaseResources();
    return audit;
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            std::cerr << "usage: tmgm_plugin_host_smoke <plugin-bundle> [test.wav]\n";
            return 2;
        }
        juce::ScopedJuceInitialiser_GUI initialise_juce;
        const juce::String plugin_path = juce::File{argv[1]}.getFullPathName();
        TestedPluginFormat format;
        juce::OwnedArray<juce::PluginDescription> descriptions;
#if defined(TMGM_HOST_TEST_LV2)
        const auto plugin_parent =
            juce::File{plugin_path}.getParentDirectory().getFullPathName();
        format.searchPathsForPlugins(
            juce::FileSearchPath{plugin_parent}, false, false);
        format.findAllTypesForFile(
            descriptions, juce::String{kExpectedPluginIdentifier.data()});
#else
        format.findAllTypesForFile(descriptions, plugin_path);
#endif
        require(descriptions.size() == 1,
                "plug-in scan did not return exactly one audio class");
        const auto& description = *descriptions[0];
        require(description.name.toStdString() == kExpectedPluginName,
                "scanned plug-in name differs");
        require(description.pluginFormatName.toStdString() == kExpectedFormat,
                "scanned plug-in format differs");

        juce::String creation_error;
        auto instance = format.createInstanceFromDescription(
            description, 48000.0, 512, creation_error);
        require(instance != nullptr,
                std::string("plug-in host instantiate failed: ")
                    + creation_error.toStdString());
        require(instance->getName().toStdString() == kExpectedPluginName,
                "instantiated plug-in name differs");
        require(instance->producesMidi(), "host reports no MIDI output");
        require(!instance->acceptsMidi(),
                "plug-in unexpectedly requires MIDI input");
        set_mono_layout(*instance);
#if defined(TMGM_HOST_TEST_VST3)
        test_stereo_layout(*instance);
#endif
        test_dry_and_status(*instance);

        MidiAudit audit;
        if (argc == 3) {
            // Use a fresh host instance so the lifecycle audit cannot inherit
            // state from the independent Dry Audio probe above.
            juce::String render_error;
            auto render_instance = format.createInstanceFromDescription(
                description, 44100.0, 512, render_error);
            require(render_instance != nullptr,
                    std::string("plug-in render instantiate failed: ")
                        + render_error.toStdString());
            audit = process_wav(*render_instance, juce::File{argv[2]});
            require(audit.note_ons > 0U,
                    "real WAV host render emitted no Note On");
            require(audit.duplicate_ons == 0U,
                    "host render duplicated Note On without Note Off");
            require(audit.orphan_offs == 0U,
                    "host render emitted orphan Note Off");
            require(audit.active_count() == 0U,
                    "host render/bypass left stuck notes");
        }

        std::cout << kExpectedFormat << " host smoke passed: name='"
                  << description.name << "' uid="
                  << description.uniqueId << ' ' << kExpectedLayoutSummary
                  << ", MIDI-out, "
                  << "model/status 48k, Dry Off/On";
        if (argc == 3) {
            std::cout << ", wav note_on=" << audit.note_ons
                      << " note_off=" << audit.note_offs
                      << " max_poly=" << audit.maximum_polyphony;
        }
        std::cout << ", bundle=" << plugin_path << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << kExpectedFormat << " host smoke failure: "
                  << exception.what() << '\n';
        return 1;
    }
}
