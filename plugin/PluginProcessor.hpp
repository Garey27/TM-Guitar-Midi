#pragma once

#include "TmRealtimeEngine.hpp"

#include <JuceHeader.h>

#include <array>
#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

class TmPreviewProcessor final : public juce::AudioProcessor {
public:
    TmPreviewProcessor();
    ~TmPreviewProcessor() override = default;

    void prepareToPlay(double sampleRate, int maximumExpectedSamplesPerBlock) override;
    void releaseResources() override;
    void reset() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    void processBlockBypassed(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override { return nullptr; }
    bool hasEditor() const override { return false; }
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return true; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 1.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;

    [[nodiscard]] juce::String modelStatus() const;

private:
    struct MidiSink {
        juce::MidiBuffer* buffer = nullptr;
        int sample_count = 0;
    };

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    static void engineMidiCallback(
        void* user,
        const tmgm::preview::MidiEvent& event) noexcept;
    static std::filesystem::path jucePath(const juce::File& file);
    [[nodiscard]] std::filesystem::path findPackageRoot() const;
    void loadPackage();
    void queuePanic() noexcept;
    void flushPanic(juce::MidiBuffer& midi, int sample_offset = 0) noexcept;
    void reflectStatus();

    juce::AudioProcessorValueTreeState parameters_;
    std::atomic<float>* input_gain_db_ = nullptr;
    std::atomic<float>* dry_audio_ = nullptr;
    std::atomic<float>* model_loaded_ = nullptr;
    juce::RangedAudioParameter* model_loaded_parameter_ = nullptr;
    juce::RangedAudioParameter* sample_rate_parameter_ = nullptr;

    tmgm::preview::TmRealtimeEngine engine_;
    std::vector<float> mono_scratch_;
    std::array<tmgm::preview::MidiEvent, tmgm::preview::kPitchCount> panic_{};
    std::size_t panic_count_ = 0U;
    std::string package_error_;
    bool runtime_healthy_ = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TmPreviewProcessor)
};
