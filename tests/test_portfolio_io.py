"""Lossless, reproducible and atomic compressed dataframe exports."""

import gzip

import pandas as pd
import pytest

from portfolio_core.io import atomic_write_dataframe


@pytest.mark.parametrize("index", [False, True])
def test_gzip_matches_plain_csv_without_filename_or_timestamp(tmp_path, index):
    frame = pd.DataFrame(
        {"Value": [1 / 3, float("nan")], "Label": ['Liège, "A"', "two\nlines"]},
        index=pd.date_range("2023-01-31", periods=2, freq="ME", name="Date"),
    )
    plain = tmp_path / "plain.csv"
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "different" / "second.csv.gz"
    for path in (plain, first, second):
        atomic_write_dataframe(frame, path, index=index)
    payload = first.read_bytes()
    assert payload == second.read_bytes()
    assert payload[3] & 8 == 0  # No original filename in the gzip header.
    assert payload[4:8] == b"\0\0\0\0"  # No wall-clock timestamp.
    assert gzip.decompress(payload) == plain.read_bytes()
    pd.testing.assert_frame_equal(pd.read_csv(first), pd.read_csv(plain))
    assert sorted(p.name for p in first.parent.iterdir()) == [
        "different", "first.csv.gz", "plain.csv"
    ]


@pytest.mark.parametrize("frame", [pd.DataFrame(), pd.DataFrame(columns=["Value"])])
def test_empty_gzip_exports_preserve_plain_csv_bytes(tmp_path, frame):
    plain, compressed = tmp_path / "empty.csv", tmp_path / "empty.csv.gz"
    atomic_write_dataframe(frame, plain)
    atomic_write_dataframe(frame, compressed)
    assert gzip.decompress(compressed.read_bytes()) == plain.read_bytes()


def test_failed_gzip_write_preserves_existing_target_and_removes_temporary(tmp_path):
    target = tmp_path / "results.csv.gz"
    atomic_write_dataframe(pd.DataFrame({"Value": [1]}), target)
    original = target.read_bytes()

    class BrokenFrame:
        def to_csv(self, stream, **kwargs):
            stream.write(b"partial data")
            raise RuntimeError("interrupted export")

    with pytest.raises(RuntimeError, match="interrupted export"):
        atomic_write_dataframe(BrokenFrame(), target)
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]
