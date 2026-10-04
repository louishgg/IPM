"""Atomic filesystem-writing helpers shared by repository workflows."""

from __future__ import annotations

import csv
import gzip
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def atomic_output_path(target: Path) -> Iterator[Path]:
    """Yield a same-directory temporary path and atomically promote on success."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_bytes(content: bytes, target: Path) -> None:
    with atomic_output_path(target) as temporary:
        temporary.write_bytes(content)


def save_figure(
    fig: Any,
    path: Path,
    *,
    dpi: int = 160,
    atomic: bool = False,
) -> None:
    """Save one PNG, optionally through a sibling atomic replacement."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not atomic:
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return
    try:
        with atomic_output_path(path) as temporary:
            fig.savefig(temporary, dpi=dpi, format="png")
    finally:
        plt.close(fig)


def atomic_write_text(
    content: str,
    target: Path,
    *,
    encoding: str = "utf-8",
) -> None:
    with atomic_output_path(target) as temporary:
        temporary.write_text(content, encoding=encoding)


def atomic_write_csv_rows(
    rows: Iterable[Mapping[str, object]],
    columns: Sequence[str],
    target: Path,
) -> None:
    """Write a deterministic RFC-4180 CSV through a same-directory replace."""
    fieldnames = tuple(str(column) for column in columns)
    if not fieldnames or len(set(fieldnames)) != len(fieldnames):
        raise ValueError("CSV columns must be nonempty and unique")
    with atomic_output_path(target) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                missing = set(fieldnames) - set(row)
                if missing:
                    raise ValueError(f"CSV row is missing fields: {sorted(missing)}")
                writer.writerow({column: row[column] for column in fieldnames})
            stream.flush()
            os.fsync(stream.fileno())


def atomic_write_dataframe(frame, target: Path, **to_csv_kwargs) -> None:
    """Atomically write CSV, using deterministic gzip for .csv.gz targets."""
    target = Path(target)
    options = {
        "index": False,
        "date_format": "%Y-%m-%d",
        "float_format": "%.17g",
        "lineterminator": "\n",
        **to_csv_kwargs,
    }
    with atomic_output_path(target) as temporary:
        if target.name.endswith(".csv.gz"):
            # Stream compressed bytes directly to the atomic temporary file.
            # Neither its random filename nor the clock belongs in the header.
            with temporary.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0
                ) as compressed:
                    frame.to_csv(compressed, **options)
        else:
            frame.to_csv(temporary, **options)


__all__ = [
    "atomic_output_path",
    "atomic_write_bytes",
    "atomic_write_csv_rows",
    "atomic_write_dataframe",
    "atomic_write_text",
    "save_figure",
]
