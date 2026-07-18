#include "PluginProcessor.hpp"

#include <algorithm>
#include <cmath>
#include <exception>

#if JUCE_WINDOWS
#include <windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

// Linker-provided PE image base for this loaded module. This identifies the
// VST3 DLL even when the host executable is the process entry module.
extern "C" IMAGE_DOS_HEADER __ImageBase;
#elif JUCE_MAC || JUCE_LINUX || JUCE_BSD
#include <dlfcn.h>
#endif

namespace {

constexpr const char* kPackageEnvironment = "TM_GUITARMIDI_TM_PACKAGE";

#if JUCE_MAC || JUCE_LINUX || JUCE_BSD
// dladdr resolves the image containing this address, so the result is the
// loaded plug-in module rather than the REAPER/host executable.
char module_anchor = 0;
#endif

#if JUCE_WINDOWS
[[nodiscard]] juce::File plugin_module_file() {
    std::array<wchar_t, 32768> buffer{};
    const auto module = reinterpret_cast<HMODULE>(&__ImageBase);
    const DWORD length = ::GetModuleFileNameW(
        module, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length != 0U && length < buffer.size()) {
        return juce::File{juce::String{
            buffer.data(), static_cast<std::size_t>(length)}};
    }
    return juce::File::getSpecialLocation(juce::File::currentExecutableFile);
}
#elif JUCE_MAC || JUCE_LINUX || JUCE_BSD
[[nodiscard]] juce::File plugin_module_file() {
    Dl_info information{};
    if (::dladdr(static_cast<const void*>(&module_anchor), &information) != 0
        && information.dli_fname != nullptr) {
        return juce::File{
            juce::String::fromUTF8(information.dli_fname)};
    }
    return juce::File::getSpecialLocation(juce::File::currentExecutableFile);
}
#else
[[nodiscard]] juce::File plugin_module_file() {
    return juce::File::getSpecialLocation(juce::File::currentExecutableFile);
}
#endif

}  // namespace

TmPreviewProcessor::TmPreviewProcessor()
    : AudioProcessor(BusesProperties()
          .withInput("Input", juce::AudioChannelSet::mono(), true)
          .withOutput("Output", juce::AudioChannelSet::mono(), true)),
      parameters_(*this, nullptr, "TM_PREVIEW_PARAMETERS", createParameterLayout()) {
    input_gain_db_ = parameters_.getRawParameterValue("inputGain");
    dry_audio_ = parameters_.getRawParameterValue("dryAudio");
    model_loaded_ = parameters_.getRawParameterValue("modelLoaded");
    model_loaded_parameter_ = parameters_.getParameter("modelLoaded");
    sample_rate_parameter_ = parameters_.getParameter("sampleRateOk");
    loadPackage();
    reflectStatus();
}

juce::AudioProcessorValueTreeState::ParameterLayout
TmPreviewProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> parameters;
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"inputGain", 1},
        "Acoustic Gain (gate/velocity)",
        juce::NormalisableRange<float>{-24.0F, 30.0F, 0.1F},
        20.0F,
        juce::AudioParameterFloatAttributes{}.withLabel("dB")));
    parameters.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{"dryAudio", 1},
        "Dry Audio",
        false));
    parameters.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{"modelLoaded", 1},
        "Model Loaded (status)",
        false));
    parameters.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{"sampleRateOk", 1},
        "Sample Rate 44.1/48 kHz (status)",
        false));
    return {parameters.begin(), parameters.end()};
}

void TmPreviewProcessor::prepareToPlay(
    const double sample_rate,
    const int maximum_expected_samples_per_block) {
    queuePanic();
    mono_scratch_.assign(
        static_cast<std::size_t>(std::max(maximum_expected_samples_per_block, 1)),
        0.0F);
    const float sample_rate_status =
        (std::abs(sample_rate - 44100.0) < 0.5
            || std::abs(sample_rate - 48000.0) < 0.5)
            ? 1.0F : 0.0F;
    sample_rate_parameter_->setValueNotifyingHost(sample_rate_status);
    std::string error;
    runtime_healthy_ = engine_.prepare(
        sample_rate, mono_scratch_.size(), error);
    if (!runtime_healthy_) {
        package_error_ = std::move(error);
    }
    setLatencySamples(runtime_healthy_
        ? static_cast<int>(engine_.latency_samples())
        : 0);
    reflectStatus();
}

void TmPreviewProcessor::releaseResources() {
    queuePanic();
}

void TmPreviewProcessor::reset() {
    queuePanic();
}

bool TmPreviewProcessor::isBusesLayoutSupported(
    const BusesLayout& layouts) const {
    const auto input = layouts.getMainInputChannelSet();
    return (input == juce::AudioChannelSet::mono()
            || input == juce::AudioChannelSet::stereo())
        && layouts.getMainOutputChannelSet() == input;
}

void TmPreviewProcessor::processBlock(
    juce::AudioBuffer<float>& audio,
    juce::MidiBuffer& midi) {
    juce::ScopedNoDenormals no_denormals;
    flushPanic(midi);
    const int sample_count = audio.getNumSamples();
    if (audio.getNumChannels() == 0 || sample_count == 0) {
        return;
    }

    const float* analysis = audio.getReadPointer(0);
    if (audio.getNumChannels() > 1
        && static_cast<std::size_t>(sample_count) <= mono_scratch_.size()) {
        const auto* right = audio.getReadPointer(1);
        for (int sample = 0; sample < sample_count; ++sample) {
            mono_scratch_[static_cast<std::size_t>(sample)] =
                0.5F * (analysis[sample] + right[sample]);
        }
        analysis = mono_scratch_.data();
    }

    if (runtime_healthy_) {
        MidiSink sink{&midi, sample_count};
        const float gain = juce::Decibels::decibelsToGain(
            input_gain_db_->load(std::memory_order_relaxed));
        const auto status = engine_.process_block(
            analysis,
            static_cast<std::size_t>(sample_count),
            gain,
            &TmPreviewProcessor::engineMidiCallback,
            &sink);
        if (status != tmgm::preview::EngineProcessStatus::success) {
            queuePanic();
            // Some NoteOns may already have been emitted earlier in this
            // block. Terminate them after, not at offset zero where JUCE's
            // sorting could place the NoteOff before the new NoteOn.
            flushPanic(midi, std::max(sample_count - 1, 0));
            runtime_healthy_ = false;
            // Do not broadcast host parameter notifications from the audio
            // callback. The cached status still fails low immediately.
            model_loaded_->store(0.0F, std::memory_order_relaxed);
        }
    }

    if (dry_audio_->load(std::memory_order_relaxed) < 0.5F) {
        audio.clear();
    }
}

void TmPreviewProcessor::processBlockBypassed(
    juce::AudioBuffer<float>&,
    juce::MidiBuffer& midi) {
    flushPanic(midi);
    queuePanic();
    flushPanic(midi);
}

void TmPreviewProcessor::engineMidiCallback(
    void* const user,
    const tmgm::preview::MidiEvent& event) noexcept {
    auto& sink = *static_cast<MidiSink*>(user);
    if (sink.buffer == nullptr || sink.sample_count <= 0) {
        return;
    }
    const int offset = std::clamp(
        static_cast<int>(event.sample_offset), 0, sink.sample_count - 1);
    if (event.kind == tmgm::preview::MidiEvent::Kind::note_on) {
        sink.buffer->addEvent(
            juce::MidiMessage::noteOn(
                1, static_cast<int>(event.pitch), event.velocity),
            offset);
    } else {
        sink.buffer->addEvent(
            juce::MidiMessage::noteOff(
                1, static_cast<int>(event.pitch), static_cast<juce::uint8>(0)),
            offset);
    }
}

std::filesystem::path TmPreviewProcessor::jucePath(const juce::File& file) {
    const juce::String full_path = file.getFullPathName();
#if JUCE_WINDOWS
    return std::filesystem::path{full_path.toWideCharPointer()};
#else
    return std::filesystem::path{full_path.toStdString()};
#endif
}

std::filesystem::path TmPreviewProcessor::findPackageRoot() const {
    const auto environment = juce::SystemStats::getEnvironmentVariable(
        kPackageEnvironment, {});
    if (environment.isNotEmpty()) {
        const juce::File candidate{environment};
        if (candidate.isDirectory()) {
            return jucePath(candidate);
        }
    }

    auto cursor = plugin_module_file().getParentDirectory();
    for (int depth = 0; depth < 8; ++depth) {
        const auto direct = cursor.getChildFile("TMModel");
        if (direct.isDirectory()) {
            return jucePath(direct);
        }
        const auto resources = cursor.getChildFile("Resources")
            .getChildFile("TMModel");
        if (resources.isDirectory()) {
            return jucePath(resources);
        }
        cursor = cursor.getParentDirectory();
    }

#if defined(TMGM_TM_PACKAGE_PATH)
    const juce::File development_package{TMGM_TM_PACKAGE_PATH};
    if (development_package.isDirectory()) {
        return jucePath(development_package);
    }
#endif
    return {};
}

void TmPreviewProcessor::loadPackage() {
    const auto package = findPackageRoot();
    if (package.empty()) {
        package_error_ = "strict-cap16-v3 package not found";
        runtime_healthy_ = false;
        return;
    }
    runtime_healthy_ = engine_.load_package(package, package_error_);
}

void TmPreviewProcessor::queuePanic() noexcept {
    std::array<tmgm::preview::MidiEvent, tmgm::preview::kPitchCount> released_events{};
    const auto released = engine_.release_active_notes(released_events, 0U);
    for (std::size_t index = 0U; index < released; ++index) {
        const auto pitch = released_events[index].pitch;
        const bool already_queued = std::any_of(
            panic_.begin(), panic_.begin() + static_cast<std::ptrdiff_t>(panic_count_),
            [pitch](const auto& event) { return event.pitch == pitch; });
        if (!already_queued && panic_count_ < panic_.size()) {
            panic_[panic_count_++] = released_events[index];
        }
    }
    engine_.reset();
}

void TmPreviewProcessor::flushPanic(
    juce::MidiBuffer& midi,
    const int sample_offset) noexcept {
    for (std::size_t index = 0U; index < panic_count_; ++index) {
        midi.addEvent(
            juce::MidiMessage::noteOff(
                1, static_cast<int>(panic_[index].pitch),
                static_cast<juce::uint8>(0)),
            std::max(sample_offset, 0));
    }
    panic_count_ = 0U;
}

void TmPreviewProcessor::reflectStatus() {
    const float status =
        engine_.package_loaded() && runtime_healthy_ ? 1.0F : 0.0F;
    model_loaded_parameter_->setValueNotifyingHost(status);
}

juce::String TmPreviewProcessor::modelStatus() const {
    if (engine_.package_loaded() && runtime_healthy_) {
        return "TM strict-cap16-v3 loaded";
    }
    return juce::String{package_error_};
}

void TmPreviewProcessor::getStateInformation(juce::MemoryBlock& destination) {
    if (const auto xml = parameters_.copyState().createXml()) {
        copyXmlToBinary(*xml, destination);
    }
}

void TmPreviewProcessor::setStateInformation(
    const void* data,
    const int size_in_bytes) {
    if (const auto xml = getXmlFromBinary(data, size_in_bytes)) {
        if (xml->hasTagName(parameters_.state.getType())) {
            parameters_.replaceState(juce::ValueTree::fromXml(*xml));
        }
    }
    reflectStatus();
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new TmPreviewProcessor();
}
