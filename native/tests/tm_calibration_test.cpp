#include "tmgm/tm_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
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

tmgm::native::NativeDataset make_dataset() {
    tmgm::native::NativeDataset dataset;
    dataset.header.frame_count = 4U;
    dataset.header.feature_count = 1U;
    dataset.header.feature_words_per_row = 1U;
    dataset.header.note_count = 2U;
    dataset.header.label_words_per_row = 1U;
    dataset.header.midi_min = 40;
    dataset.header.midi_max = 41;
    dataset.header.sample_rate = 22050U;
    dataset.header.hop_size = 256U;
    dataset.feature_words.resize(4U, 0U);
    dataset.activity_words.resize(4U, 0U);
    dataset.onset_words.resize(4U, 0U);
    dataset.set_activity(0U, 0U, true);
    dataset.set_activity(1U, 0U, true);
    dataset.set_activity(2U, 1U, true);
    dataset.set_onset(2U, 1U, true);
    dataset.validate();
    return dataset;
}

void test_activity_calibration() {
    const auto dataset = make_dataset();
    const std::vector<std::int32_t> scores{
        5, 4,
        3, 2,
        1, 4,
        0, -1,
    };
    const auto result = tmgm::native::calibrate_score_threshold(
        dataset, tmgm::native::TmModelHead::activity, scores, 1.5);
    require(result.threshold == 3, "activity threshold is not the best F1 cut");
    require(result.true_positives == 3U, "activity true positives are wrong");
    require(result.false_positives == 1U, "activity false positives are wrong");
    require(result.false_negatives == 0U, "activity false negatives are wrong");
    require(std::abs(result.f1 - 6.0 / 7.0) < 1.0e-12,
            "activity F1 is wrong");

    std::vector<std::uint8_t> predictions;
    tmgm::native::apply_score_threshold(scores, result.threshold, predictions);
    require(predictions == std::vector<std::uint8_t>({1, 1, 1, 0, 0, 1, 0, 0}),
            "thresholded activity predictions are wrong");
}

void test_head_selects_onset_truth() {
    const auto dataset = make_dataset();
    const std::vector<std::int32_t> scores{
        5, 4,
        3, 2,
        1, 8,
        0, -1,
    };
    const auto result = tmgm::native::calibrate_score_threshold(
        dataset, tmgm::native::TmModelHead::onset, scores, 4.0);
    require(result.threshold == 8, "onset truth was not selected by model head");
    require(result.true_positives == 1U && result.false_positives == 0U,
            "onset calibration counts are wrong");
    require(result.f1 == 1.0, "onset calibration should be perfect");
}

void test_zero_positive_and_validation() {
    auto dataset = make_dataset();
    std::fill(dataset.onset_words.begin(), dataset.onset_words.end(), 0U);
    const std::vector<std::int32_t> scores(8U, 7);
    const auto result = tmgm::native::calibrate_score_threshold(
        dataset, tmgm::native::TmModelHead::onset, scores, 4.0);
    require(result.threshold == 8, "zero-positive calibration should predict nothing");
    require(result.predicted_mean_polyphony == 0.0,
            "zero-positive calibration emitted predictions");

    require_throws(
        [&] {
            static_cast<void>(tmgm::native::calibrate_score_threshold(
                dataset,
                tmgm::native::TmModelHead::onset,
                std::vector<std::int32_t>(7U),
                4.0));
        },
        "wrong score dimensions were accepted");
    require_throws(
        [&] {
            static_cast<void>(tmgm::native::calibrate_score_threshold(
                dataset,
                tmgm::native::TmModelHead::onset,
                scores,
                0.25));
        },
        "invalid polyphony ratio was accepted");
}

}  // namespace

int main() {
    test_activity_calibration();
    test_head_selects_onset_truth();
    test_zero_positive_and_validation();
    return 0;
}
