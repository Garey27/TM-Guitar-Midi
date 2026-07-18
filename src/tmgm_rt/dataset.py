from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib

import numpy as np

from .audio import load_audio_mono_channel_zero
from .config import ContextConfig, FrontendConfig, TargetConfig
from .context import stack_causal_context
from .nnpg import event_targets, guitar_teacher_slice, read_nnpg
from .stft_plus import extract_stft_plus


@dataclass(frozen=True)
class CorpusEntry:
    split: str
    source: str
    identifier: str
    input_path: Path
    output_relative: Path
    group: str


@dataclass
class DatasetSplit:
    features: np.ndarray
    targets: np.ndarray
    soft_targets: np.ndarray
    track_ids: np.ndarray
    polyphony: np.ndarray
    category_counts: dict[str, int]


def read_corpus(
    path: str | Path,
    split: str,
    limit: int | None = None,
    seed: int = 42,
) -> list[CorpusEntry]:
    available: list[CorpusEntry] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["split"] != split:
                continue
            available.append(
                CorpusEntry(
                    split=row["split"],
                    source=row["source"],
                    identifier=row["id"],
                    input_path=Path(row["input"]),
                    output_relative=Path(row["output_rel"]),
                    group=row["group"],
                )
            )
    if not available:
        raise ValueError(f"no corpus rows for split={split}")
    if limit is None or limit >= len(available):
        return available

    # The manifest is grouped by source (GOAT comes first), so taking its first
    # N rows silently produces single-corpus experiments. Hash-sort each source
    # and draw round-robin for a deterministic, source-stratified track pool.
    by_source: dict[str, list[CorpusEntry]] = {}
    for entry in available:
        by_source.setdefault(entry.source, []).append(entry)
    # Within a source, rotate through performer/task groups as well. This keeps
    # GuitarSet performers and Guitar-TECHS playing tasks diverse even in a
    # twelve-track engineering ablation.
    for source, source_entries in by_source.items():
        by_group: dict[str, list[CorpusEntry]] = {}
        for entry in source_entries:
            by_group.setdefault(entry.group, []).append(entry)
        for group_entries in by_group.values():
            group_entries.sort(
                key=lambda entry: _stable_seed(
                    f"{split}/{source}/{entry.identifier}", seed
                )
            )
        groups = sorted(
            by_group,
            key=lambda group: _stable_seed(f"{split}/{source}/{group}", seed),
        )
        positions = {group: 0 for group in groups}
        interleaved: list[CorpusEntry] = []
        while len(interleaved) < len(source_entries):
            for group in groups:
                position = positions[group]
                entries = by_group[group]
                if position < len(entries):
                    interleaved.append(entries[position])
                    positions[group] += 1
        by_source[source] = interleaved

    result: list[CorpusEntry] = []
    positions = {source: 0 for source in by_source}
    sources = sorted(by_source)
    while len(result) < limit:
        made_progress = False
        for source in sources:
            position = positions[source]
            entries = by_source[source]
            if position >= len(entries):
                continue
            result.append(entries[position])
            positions[source] += 1
            made_progress = True
            if len(result) == limit:
                break
        if not made_progress:
            break
    return result


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") ^ seed


def _balanced_indices(
    activity: np.ndarray,
    onset: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    onset_count = onset.sum(axis=1)
    active_count = activity.sum(axis=1)
    masks = {
        "chord_onset": onset_count >= 2,
        "single_onset": onset_count == 1,
        "poly_sustain": (onset_count == 0) & (active_count >= 2),
        "single_sustain": (onset_count == 0) & (active_count == 1),
        "silence": active_count == 0,
    }
    proportions = {
        "chord_onset": 0.30,
        "single_onset": 0.20,
        "poly_sustain": 0.30,
        "single_sustain": 0.15,
        "silence": 0.05,
    }
    chosen: list[np.ndarray] = []
    counts: dict[str, int] = {}
    remaining = count
    available_names = [name for name, mask in masks.items() if mask.any()]
    for index, name in enumerate(available_names):
        candidates = np.flatnonzero(masks[name])
        if index == len(available_names) - 1:
            wanted = remaining
        else:
            wanted = min(remaining, int(round(count * proportions[name])))
        if wanted <= 0:
            counts[name] = 0
            continue
        selection = rng.choice(candidates, size=wanted, replace=candidates.size < wanted)
        chosen.append(selection)
        counts[name] = int(wanted)
        remaining -= wanted
    if remaining > 0:
        extra = rng.choice(activity.shape[0], size=remaining, replace=activity.shape[0] < remaining)
        chosen.append(extra)
        counts["fallback"] = int(remaining)
    indices = np.concatenate(chosen) if chosen else np.arange(activity.shape[0])
    rng.shuffle(indices)
    return indices.astype(np.int64), counts


def build_track_examples(
    entry: CorpusEntry,
    teacher_root: str | Path,
    frontend: FrontendConfig,
    context: ContextConfig,
    targets: TargetConfig,
    frames_per_track: int,
    seed: int,
    balanced_sampling: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    teacher_base = Path(teacher_root) / entry.output_relative
    posterior = read_nnpg(teacher_base.with_suffix(".nnpg"))
    if posterior.header.sample_rate != frontend.sample_rate:
        raise ValueError("teacher/frontend sample rate mismatch")
    if posterior.header.hop_size != frontend.hop_size:
        raise ValueError("teacher/frontend hop mismatch")

    audio = load_audio_mono_channel_zero(entry.input_path, frontend.sample_rate)
    spectral = extract_stft_plus(audio, frontend)
    features = stack_causal_context(spectral, context)
    event = event_targets(
        teacher_base.with_suffix(".events.tsv"),
        posterior.header.frame_count,
        frontend.midi_min,
        frontend.midi_max,
        targets.onset_width_frames,
        targets.onset_delay_frames,
    )
    soft_activity, soft_onset = guitar_teacher_slice(
        posterior, frontend.midi_min, frontend.midi_max
    )
    frame_count = min(
        features.shape[0], event.activity.shape[0], soft_activity.shape[0]
    )
    if frame_count < 8:
        raise ValueError(f"track is too short: {entry.identifier}")
    features = features[:frame_count]
    activity = event.activity[:frame_count]
    onset = event.onset[:frame_count]
    output_parts: list[np.ndarray] = []
    soft_parts: list[np.ndarray] = []
    if targets.activity_outputs:
        output_parts.append(activity)
        soft_parts.append(soft_activity[:frame_count])
    if targets.onset_outputs:
        output_parts.append(onset)
        soft_parts.append(soft_onset[:frame_count])
    y = np.concatenate(output_parts, axis=1).astype(np.uint32)
    soft = np.concatenate(soft_parts, axis=1).astype(np.float32)

    rng = np.random.default_rng(_stable_seed(entry.identifier, seed))
    if balanced_sampling:
        # Sampling with replacement is deliberate for rare chord-onset categories.
        indices, category_counts = _balanced_indices(
            activity, onset, frames_per_track, rng
        )
    else:
        # Validation must preserve the natural temporal/class prior.
        count = min(frames_per_track, frame_count)
        indices = np.linspace(0, frame_count - 1, count, dtype=np.int64)
        category_counts = {"natural_uniform": int(count)}
    return (
        features[indices],
        y[indices],
        soft[indices],
        activity[indices].sum(axis=1).astype(np.uint8),
        category_counts,
    )


def build_split(
    corpus_path: str | Path,
    teacher_root: str | Path,
    split: str,
    track_limit: int,
    frames_per_track: int,
    frontend: FrontendConfig,
    context: ContextConfig,
    targets: TargetConfig,
    seed: int = 42,
) -> DatasetSplit:
    entries = read_corpus(corpus_path, split, track_limit, seed)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    soft: list[np.ndarray] = []
    track_ids: list[np.ndarray] = []
    polyphony: list[np.ndarray] = []
    totals: dict[str, int] = {}
    for track_index, entry in enumerate(entries):
        print(f"[{split} {track_index + 1}/{len(entries)}] {entry.source}/{entry.identifier}")
        x, y, teacher_soft, poly, counts = build_track_examples(
            entry,
            teacher_root,
            frontend,
            context,
            targets,
            frames_per_track,
            seed,
            balanced_sampling=(split == "train"),
        )
        features.append(x)
        labels.append(y)
        soft.append(teacher_soft)
        polyphony.append(poly)
        track_ids.append(np.full(x.shape[0], track_index, dtype=np.uint16))
        for name, value in counts.items():
            totals[name] = totals.get(name, 0) + value
    return DatasetSplit(
        features=np.concatenate(features),
        targets=np.concatenate(labels),
        soft_targets=np.concatenate(soft),
        track_ids=np.concatenate(track_ids),
        polyphony=np.concatenate(polyphony),
        category_counts=totals,
    )
