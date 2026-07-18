from pathlib import Path

from tmgm_rt.dataset import read_corpus


def test_read_corpus_limit_is_source_stratified(tmp_path: Path):
    manifest = tmp_path / "corpus.tsv"
    manifest.write_text(
        "split\tsource\tid\tinput\toutput_rel\tgroup\n"
        + "".join(
            f"train\talpha\ta{i}\ta{i}.wav\ta{i}\tga{i % 2}\n" for i in range(6)
        )
        + "".join(
            f"train\tbeta\tb{i}\tb{i}.wav\tb{i}\tgb{i}\n" for i in range(6)
        ),
        encoding="utf-8",
    )

    first = read_corpus(manifest, "train", limit=4, seed=9)
    second = read_corpus(manifest, "train", limit=4, seed=9)
    assert [entry.identifier for entry in first] == [
        entry.identifier for entry in second
    ]
    assert {entry.source for entry in first} == {"alpha", "beta"}
    assert sum(entry.source == "alpha" for entry in first) == 2
    assert sum(entry.source == "beta" for entry in first) == 2
    assert len({entry.group for entry in first if entry.source == "alpha"}) == 2
