"""Live run tracking: one JSONL event stream, TensorBoard scalars, trace dumps.

Three artefacts, all under `star.out_dir`, all written as the run progresses
rather than at the end - a preemptible job that dies at hour 23 should still
leave behind everything it learned:

    metrics.jsonl        append-only, one JSON object per event. `tail -f` it on
                         the cluster; `pd.read_json(..., lines=True)` later.
    tb/                  TensorBoard event files: `tensorboard --logdir runs/x/tb`
    iterK/trace_samples.md   solved and failed reasoning traces, side by side
                         with the gold answer, for reading with your eyes

No Weights & Biases. Cluster compute nodes have no outbound network, so a
tracker that phones home would stall the job or need to be disabled exactly
when it matters; TensorBoard reads these files after the fact over an ssh
tunnel and costs nothing at run time.

Every method is a no-op when `tracking.enabled` is false, and the TensorBoard
half degrades to a warning if the `tensorboard` package is missing, so nothing
here can be the reason a training run fails.
"""

from __future__ import annotations

import json
import logging
import time
from numbers import Number
from pathlib import Path
from typing import Any, Sequence

from .config import Config

log = logging.getLogger(__name__)


def _summary_writer(tb_dir: Path):
    """A TensorBoard SummaryWriter, or None if tensorboard is not installed."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        log.warning(
            "tracking.tensorboard is on but the `tensorboard` package is missing "
            "(pip install tensorboard). Metrics still go to the jsonl stream."
        )
        return None
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir))


def _flatten(prefix: str, obj: Any) -> dict[str, float]:
    """Numeric leaves of a nested dict, as `a/b/c -> value`.

    Lets callers hand over a whole stats block (`summarise()`'s output, with its
    nested `reject_reasons`) and get every number plotted, without listing them.
    Booleans are treated as numbers on purpose: `answer_disjoint` as a 0/1 line
    is exactly what you want to see.
    """
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten(f"{prefix}/{key}" if prefix else str(key), value))
    elif isinstance(obj, bool):
        out[prefix] = float(obj)
    elif isinstance(obj, Number):
        out[prefix] = float(obj)
    return out


class RunTracker:
    """Writes the three artefacts above. Safe to use as a context manager."""

    def __init__(self, run_dir: str | Path, cfg: Config):
        self.cfg = cfg.tracking
        self.run_dir = Path(run_dir)
        self.enabled = bool(self.cfg.enabled)
        self._t0 = time.time()
        self._fh = None
        self._writer = None
        if not self.enabled:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Append, not truncate: a resubmitted job after a preemption adds to the
        # same history instead of erasing what the previous attempt recorded.
        self._fh = (self.run_dir / self.cfg.metrics_file).open("a", encoding="utf-8")
        if self.cfg.tensorboard:
            self._writer = _summary_writer(self.run_dir / self.cfg.tb_dir)

    # -- the event stream -------------------------------------------------
    def event(
        self,
        name: str,
        *,
        iteration: int | None = None,
        step: int | None = None,
        scalars: dict | None = None,
        **fields: Any,
    ) -> None:
        """Record one thing that happened.

        `scalars` (nested dicts fine) is both written to the jsonl row and
        plotted to TensorBoard under `name/...`; `fields` is jsonl-only, for
        strings and paths that make the log readable but do not plot.
        """
        if not self.enabled:
            return
        row: dict[str, Any] = {
            "event": name,
            "elapsed_s": round(time.time() - self._t0, 2),
        }
        if iteration is not None:
            row["iteration"] = iteration
        if step is not None:
            row["step"] = step
        if scalars:
            row.update(scalars)
        row.update(fields)
        self._write(row)
        if scalars:
            # Iteration number is the natural x-axis for loop-level metrics;
            # per-step training logs pass their own global step.
            self.scalars(scalars, step=step if step is not None else iteration, prefix=name)

    def scalars(self, values: dict, *, step: int | None, prefix: str = "") -> None:
        if not self.enabled or self._writer is None:
            return
        for tag, value in _flatten(prefix, values).items():
            self._writer.add_scalar(tag, value, global_step=step or 0)
        self._writer.flush()

    def config(self, cfg: Config) -> None:
        """Pin the run's settings into both artefacts as the first entry."""
        if not self.enabled:
            return
        self.event("config", **{"config": cfg.to_dict()})
        if self._writer is not None:
            self._writer.add_text(
                "config", f"```json\n{json.dumps(cfg.to_dict(), indent=2)}\n```", 0
            )

    def _write(self, row: dict) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        # Flush every line: on a preemptible partition the process can be killed
        # without warning, and a buffered log is an empty log.
        self._fh.flush()

    # -- qualitative output -----------------------------------------------
    def trace_samples(
        self,
        candidate_rows: Sequence[dict],
        out_file: str | Path,
        *,
        iteration: int,
        hinted: bool = False,
    ) -> Path | None:
        """Dump solved and failed traces to markdown for a human to read.

        The quantitative side (pass@N, EM) says whether the loop is working; this
        says *how* - whether the model is decomposing clues or has learned to
        emit a plausible-looking wordplay sentence and guess. Half the samples
        are solved clues and half are failures, since the failures are where the
        interesting behaviour is.
        """
        limit = int(self.cfg.trace_samples or 0)
        if not self.enabled or limit <= 0 or not candidate_rows:
            return None

        solved = [r for r in candidate_rows if r.get("n_correct")]
        failed = [r for r in candidate_rows if not r.get("n_correct")]
        half = max(limit // 2, 1)
        picked = [("solved", r) for r in solved[:half]] + [
            ("failed", r) for r in failed[: limit - min(len(solved), half)]
        ]

        lines = [
            f"# Iteration {iteration} trace samples"
            + (" (rationalisation, answer given as a hint)" if hinted else ""),
            "",
            f"{len(solved)}/{len(candidate_rows)} clues solved. "
            f"Showing {len(picked)} of them.",
            "",
        ]
        for kind, row in picked:
            lines += [
                f"## [{kind}] {row.get('clue', '')} {row.get('enumeration', '')}".rstrip(),
                "",
                f"- gold: `{row.get('letters', '')}`",
                f"- human rationale: {row.get('wordplay') or '_none_'}",
                "",
            ]
            for cand in row.get("candidates", [])[:4]:
                mark = "correct" if cand.get("label") == 1 else cand.get("reason", "rejected")
                lines += [
                    f"- **{mark}** -> `{cand.get('predicted') or ''}`",
                    f"  - {(cand.get('trace') or '').strip()}",
                ]
            lines.append("")

        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("\n".join(lines), encoding="utf-8")
        log.info("wrote %d sample traces -> %s", len(picked), out_file)
        return out_file

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "RunTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
