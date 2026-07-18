#include "tmgm/tm_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace tmgm::native {
namespace {

constexpr std::uint64_t kMaximumScoreRange = 10'000'000U;

[[nodiscard]] bool packed_label(
    const std::vector<std::uint64_t>& labels,
    const NativeDatasetHeader& header,
    const std::uint64_t row,
    const std::uint32_t output) {
    const auto word = labels[
        row * header.label_words_per_row + output / kDatasetWordBits];
    return ((word >> (output % kDatasetWordBits)) & 1ULL) != 0ULL;
}

[[nodiscard]] bool is_better(
    const TmScoreCalibration& candidate,
    const TmScoreCalibration& current) {
    return candidate.f1 > current.f1 ||
        (candidate.f1 == current.f1 && candidate.threshold > current.threshold);
}

}  // namespace

double default_maximum_polyphony_ratio(const TmModelHead head) {
    switch (head) {
    case TmModelHead::activity:
        return 1.5;
    case TmModelHead::onset:
        return 4.0;
    }
    throw std::invalid_argument("unsupported native TM model head");
}

TmScoreCalibration calibrate_score_threshold(
    const NativeDataset& dataset,
    const TmModelHead head,
    const std::vector<std::int32_t>& scores,
    const double maximum_polyphony_ratio) {
    dataset.validate();
    if (!std::isfinite(maximum_polyphony_ratio) ||
        maximum_polyphony_ratio < 0.5) {
        throw std::invalid_argument(
            "maximum polyphony ratio must be finite and at least 0.5");
    }
    if (dataset.header.frame_count == 0U || dataset.header.note_count == 0U) {
        throw std::invalid_argument("cannot calibrate on an empty native dataset");
    }
    if (dataset.header.frame_count >
        std::numeric_limits<std::size_t>::max() / dataset.header.note_count) {
        throw std::invalid_argument("calibration score dimensions overflow size_t");
    }
    const auto expected_scores = static_cast<std::size_t>(
        dataset.header.frame_count * dataset.header.note_count);
    if (scores.size() != expected_scores) {
        throw std::invalid_argument(
            "score count does not match calibration dataset dimensions");
    }

    const std::vector<std::uint64_t>* labels = nullptr;
    switch (head) {
    case TmModelHead::activity:
        labels = &dataset.activity_words;
        break;
    case TmModelHead::onset:
        labels = &dataset.onset_words;
        break;
    default:
        throw std::invalid_argument("unsupported native TM model head");
    }

    std::uint64_t positives = 0U;
    auto minimum = std::numeric_limits<std::int32_t>::max();
    auto maximum = std::numeric_limits<std::int32_t>::min();
    for (std::uint64_t row = 0; row < dataset.header.frame_count; ++row) {
        for (std::uint32_t output = 0; output < dataset.header.note_count; ++output) {
            positives += packed_label(*labels, dataset.header, row, output) ? 1U : 0U;
            const auto score = scores[static_cast<std::size_t>(
                row * dataset.header.note_count + output)];
            minimum = std::min(minimum, score);
            maximum = std::max(maximum, score);
        }
    }

    const auto score_range = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(maximum) - static_cast<std::int64_t>(minimum)) +
        1U;
    if (score_range > kMaximumScoreRange) {
        throw std::runtime_error("TM score range is unexpectedly large");
    }
    std::vector<std::uint64_t> true_histogram(
        static_cast<std::size_t>(score_range), 0U);
    std::vector<std::uint64_t> false_histogram(
        static_cast<std::size_t>(score_range), 0U);
    for (std::size_t index = 0; index < scores.size(); ++index) {
        const auto row = index / dataset.header.note_count;
        const auto output = static_cast<std::uint32_t>(
            index % dataset.header.note_count);
        const auto bucket = static_cast<std::size_t>(
            static_cast<std::int64_t>(scores[index]) - minimum);
        if (packed_label(*labels, dataset.header, row, output)) {
            ++true_histogram[bucket];
        } else {
            ++false_histogram[bucket];
        }
    }

    const auto target_mean = static_cast<double>(positives) /
        static_cast<double>(dataset.header.frame_count);
    TmScoreCalibration best;
    best.f1 = -1.0;
    TmScoreCalibration best_unconstrained;
    best_unconstrained.f1 = -1.0;
    std::uint64_t true_positives = 0U;
    std::uint64_t false_positives = 0U;

    auto consider = [&](const std::int32_t threshold) {
        const auto false_negatives = positives - true_positives;
        const auto precision = static_cast<double>(true_positives) /
            static_cast<double>(
                std::max<std::uint64_t>(true_positives + false_positives, 1U));
        const auto recall = static_cast<double>(true_positives) /
            static_cast<double>(
                std::max<std::uint64_t>(true_positives + false_negatives, 1U));
        const auto f1 = 2.0 * precision * recall /
            std::max(precision + recall, 1.0e-12);
        const auto predicted_mean =
            static_cast<double>(true_positives + false_positives) /
            static_cast<double>(dataset.header.frame_count);
        const TmScoreCalibration candidate{
            threshold,
            true_positives,
            false_positives,
            false_negatives,
            precision,
            recall,
            f1,
            predicted_mean,
            target_mean,
        };
        if (is_better(candidate, best_unconstrained)) {
            best_unconstrained = candidate;
        }
        const bool allowed =
            predicted_mean >= 0.5 * target_mean &&
            predicted_mean <= maximum_polyphony_ratio * target_mean;
        if (allowed && is_better(candidate, best)) {
            best = candidate;
        }
    };

    // Include the all-negative prediction when it is representable.
    if (maximum < std::numeric_limits<std::int32_t>::max()) {
        consider(maximum + 1);
    }
    for (std::int64_t threshold = maximum; threshold >= minimum; --threshold) {
        const auto bucket = static_cast<std::size_t>(threshold - minimum);
        true_positives += true_histogram[bucket];
        false_positives += false_histogram[bucket];
        consider(static_cast<std::int32_t>(threshold));
    }
    return best.f1 >= 0.0 ? best : best_unconstrained;
}

void apply_score_threshold(
    const std::vector<std::int32_t>& scores,
    const std::int32_t threshold,
    std::vector<std::uint8_t>& predictions) {
    predictions.resize(scores.size());
    for (std::size_t index = 0; index < scores.size(); ++index) {
        predictions[index] = scores[index] >= threshold ? 1U : 0U;
    }
}

}  // namespace tmgm::native
