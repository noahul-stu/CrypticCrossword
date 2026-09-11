"""JSONL / JSON helpers and run-directory bookkeeping."""

from __future__ import annotations

import gc
import gzip
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger("cryptic_star")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def open_text(path: str | Path):
    """Open a text file, transparently handling `.gz`.

    The united dataset ships gzipped (`train.jsonl` is 205 MB plain, over
    GitHub's file limit), so every reader in the project goes through this.
    """
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_jsonl(path: str | Path, limit: int | None = None) -> Iterator[dict]:
    with open_text(path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read a JSON file, returning `default` if it is missing or truncated.

    Used to reload run state (`history.json`, the data-prep fingerprint) when a
    preempted job is resubmitted, where a half-written file is a real
    possibility and is not worth crashing over.
    """
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        log.warning("%s is not readable JSON, ignoring it", path)
        return default


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def free_gpu() -> None:
    """Hand cached GPU memory back to the driver.

    `del model` is not enough on its own: the tensors stay in torch's caching
    allocator, so the generator that was just dropped still occupies the card
    when the Trainer starts and a run that fits comfortably OOMs anyway. Call
    this right after deleting the last reference. Harmless on CPU-only hosts.
    """
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        mps.empty_cache()


def pick_device(requested: str = "auto") -> str:
    """`auto` -> cuda if present, else Apple MPS, else CPU."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
