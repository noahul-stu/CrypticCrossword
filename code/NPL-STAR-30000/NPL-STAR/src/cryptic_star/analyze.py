"""Post-training analysis: how well does the generator do on clues it has
never seen, and *where* does it fail.

    python -m cryptic_star.analyze --predictions runs/x/test_predictions.jsonl \
                                   --history runs/x/history.json \
                                   --out runs/x/analysis

Reads the `predictions.jsonl` that `evaluate.py` writes, so it costs no GPU time
and can be re-run with different slicing as often as you like.

A note on "unseen": the Cryptonite official split is answer-disjoint, so no
answer in the test split appears anywhere in training. These numbers are
therefore about *generalising the wordplay skill*, not about recalling answers
seen during fine-tuning - which is the whole point of evaluating on that split,
and worth stating in the report.

Headline metrics
    top1_em            beam-search Exact Match. The number to compare against
                       the 7.64% T5-Large baseline (Efrat et al., 2021).
    pass_at_n          any of N sampled traces correct = the ceiling a perfect
                       re-ranker could reach. Bounds the Best-of-N result.
    oracle_gap         pass_at_n - top1_em = head-room available to re-ranking.
                       If this is ~0, Best-of-N cannot help, and that is a
                       finding rather than a bug.
    majority_vote_em   self-consistency: pick the answer most sampled traces
                       agree on. A discriminator-free Best-of-N baseline; the
                       DeBERTa re-ranker has to beat THIS, not just top-1.
    trace_precision    of all traces generated, the fraction that were correct.
                       Per-trace rather than per-clue.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence

from .data.formatting import DEF_KEY, WORDPLAY_KEY
from .io_utils import read_jsonl, setup_logging, write_json
from .verify import extract_answer, extract_field

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def candidate_answers(row: dict) -> list[str]:
    return [extract_answer(t) for t in row.get("candidates", []) or []]


def majority_vote(row: dict) -> tuple[str, int, int]:
    """(most-agreed answer, its vote count, total parseable votes).

    Blank predictions do not vote. Ties break toward the beam-search answer,
    which is the sensible default when the samples disagree completely.
    """
    votes = [a for a in candidate_answers(row) if a]
    if not votes:
        return row.get("predicted", ""), 0, 0
    counts = Counter(votes)
    top_n = max(counts.values())
    tied = [a for a, c in counts.items() if c == top_n]
    if len(tied) > 1 and row.get("predicted") in tied:
        return row["predicted"], top_n, len(votes)
    return counts.most_common(1)[0][0], top_n, len(votes)


def definition_is_grounded(row: dict) -> bool | None:
    """Is the predicted `definition:` span actually a substring of the clue?

    A cheap, annotation-free proxy for reasoning quality: in a real cryptic
    clue the definition is always literally present in the surface text. `None`
    when the model emitted no definition, so it can be excluded from the rate
    rather than counted as a failure.
    """
    trace = row.get("trace", "")
    definition = extract_field(trace, DEF_KEY)
    if not definition or definition.lower() == "unknown":
        return None
    return _norm(definition) in _norm(row.get("clue", ""))


def _rate(rows: Sequence[dict], key: str) -> float:
    return sum(1 for r in rows if r.get(key)) / len(rows) if rows else 0.0


def bucket(
    rows: Sequence[dict],
    keyfn: Callable[[dict], str | None],
    min_n: int = 20,
) -> dict[str, dict]:
    """Group rows and report EM per group. Groups smaller than `min_n` are
    merged into `"(small groups)"` - a 3-clue bucket at 33% EM is noise, and
    reporting it as a finding would be wrong."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        k = keyfn(r)
        if k is not None:
            groups[str(k)].append(r)

    out: dict[str, dict] = {}
    small: list[dict] = []
    for k, rs in groups.items():
        if len(rs) < min_n:
            small.extend(rs)
            continue
        out[k] = {
            "n": len(rs),
            "top1_em": _rate(rs, "top1_correct"),
            "pass_at_n": _rate(rs, "any_correct"),
        }
    if small:
        out["(small groups)"] = {
            "n": len(small),
            "top1_em": _rate(small, "top1_correct"),
            "pass_at_n": _rate(small, "any_correct"),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))


# --------------------------------------------------------------------------
# bucketing keys
# --------------------------------------------------------------------------

def _answer_length(row: dict) -> str:
    n = len(row.get("gold", ""))
    if n <= 4:
        return "1-4 letters"
    if n <= 6:
        return "5-6 letters"
    if n <= 8:
        return "7-8 letters"
    if n <= 11:
        return "9-11 letters"
    return "12+ letters"


def _clue_length(row: dict) -> str:
    n = len((row.get("clue") or "").split())
    if n <= 4:
        return "1-4 words"
    if n <= 7:
        return "5-7 words"
    if n <= 10:
        return "8-10 words"
    return "11+ words"


def _multiword(row: dict) -> str:
    sig = re.findall(r"\d+", row.get("enumeration", "") or "")
    return "single word" if len(sig) <= 1 else f"{len(sig)} words"


def _confidence(row: dict) -> str | None:
    """Bucket by how much the sampled traces agreed on their answer.

    This is the calibration table: if EM rises steeply with agreement, then
    sample-agreement alone is a usable confidence signal, and the discriminator
    has to add value on top of it.
    """
    _, top_votes, total = majority_vote(row)
    if not total:
        return None
    share = top_votes / total
    if share >= 0.8:
        return "4. strong agreement (>=80%)"
    if share >= 0.5:
        return "3. majority (50-80%)"
    if share >= 0.25:
        return "2. weak (25-50%)"
    return "1. scattered (<25%)"


# --------------------------------------------------------------------------
# main analysis
# --------------------------------------------------------------------------

def analyse(rows: Sequence[dict], n_error_examples: int = 25) -> dict:
    if not rows:
        raise ValueError("No prediction rows to analyse.")
    n = len(rows)

    # Self-consistency baseline.
    mv_correct = 0
    for r in rows:
        answer, _, total = majority_vote(r)
        r["_mv_correct"] = bool(total) and answer == r.get("gold")
        mv_correct += int(r["_mv_correct"])

    # Trace-level precision across every sampled candidate.
    total_cands = correct_cands = 0
    for r in rows:
        for a in candidate_answers(r):
            total_cands += 1
            correct_cands += int(a == r.get("gold"))

    # Definition grounding, excluding rows with no definition emitted.
    grounded = [definition_is_grounded(r) for r in rows]
    graded = [g for g in grounded if g is not None]

    # Length agreement: did the answer at least fit the enumeration?
    def _length_ok(r: dict) -> bool:
        sig = [int(x) for x in re.findall(r"\d+", r.get("enumeration", "") or "")]
        pred = r.get("predicted", "")
        return bool(pred) and bool(sig) and len(pred) == sum(sig)

    top1_em = _rate(rows, "top1_correct")
    pass_at_n = _rate(rows, "any_correct")

    headline = {
        "n_clues": n,
        "top1_em": top1_em,
        "pass_at_n": pass_at_n,
        "oracle_gap": pass_at_n - top1_em,
        "majority_vote_em": mv_correct / n,
        "majority_vote_gain_over_top1": mv_correct / n - top1_em,
        "trace_precision": correct_cands / total_cands if total_cands else None,
        "n_traces_scored": total_cands,
        "unparseable_rate": sum(1 for r in rows if not r.get("predicted")) / n,
        # Answers that match a documented alternative for the same clue. Counted
        # as errors above (the baseline is scored the same way); reported here so
        # the write-up can say how much of the error is arguable.
        "alt_answer_rate": sum(1 for r in rows if r.get("alt_answer_match")) / n,
        "length_valid_rate": sum(1 for r in rows if _length_ok(r)) / n,
        "definition_grounded_rate": (sum(graded) / len(graded)) if graded else None,
        "definition_emitted_rate": len(graded) / n,
    }

    # Why the wrong ones were wrong.
    failures = [r for r in rows if not r.get("top1_correct")]
    reasons = Counter(r.get("reason", "unknown") for r in failures)

    # Clues no sample got right - the genuinely hard residue.
    unsolved = [r for r in rows if not r.get("any_correct")]

    report = {
        "headline": headline,
        "failure_reasons": {
            "n_failures": len(failures),
            "distribution": dict(reasons.most_common()),
            "share": {k: v / len(failures) for k, v in reasons.most_common()} if failures else {},
        },
        "by_answer_length": bucket(rows, _answer_length),
        "by_clue_length": bucket(rows, _clue_length),
        "by_answer_words": bucket(rows, _multiword),
        "by_publisher": bucket(rows, lambda r: r.get("publisher") or None),
        "by_sample_agreement": dict(sorted(bucket(rows, _confidence).items())),
        "hardest": {
            "n_unsolved_by_any_sample": len(unsolved),
            "share": len(unsolved) / n,
        },
        "error_examples": [
            {
                "clue": r.get("clue"),
                "enumeration": r.get("enumeration"),
                "gold": r.get("gold"),
                "predicted": r.get("predicted"),
                "reason": r.get("reason"),
                "wordplay": extract_field(r.get("trace", ""), WORDPLAY_KEY)[:200],
            }
            for r in failures[:n_error_examples]
        ],
        "correct_examples": [
            {
                "clue": r.get("clue"),
                "gold": r.get("gold"),
                "wordplay": extract_field(r.get("trace", ""), WORDPLAY_KEY)[:200],
            }
            for r in rows
            if r.get("top1_correct")
        ][:10],
    }
    for r in rows:
        r.pop("_mv_correct", None)
    return report


def learning_curve(history: Sequence[dict]) -> list[dict]:
    """Per-iteration table: is the STaR loop actually improving anything?

    `train_examples` growing while `val EM` is flat means self-generated data
    has stopped adding information - the signal to stop iterating.
    """
    out = []
    for h in history:
        gen = h.get("generation", {}) or {}
        rat = h.get("rationalization", {}) or {}
        val = h.get("val", {}) or {}
        out.append(
            {
                "iteration": h.get("iteration"),
                "train_examples": h.get("train_examples"),
                "new_examples": h.get("new_examples"),
                "train_clues_solved": gen.get("clues_with_at_least_one_correct"),
                "train_pass_at_n": gen.get("pass_at_n"),
                "rationalized_recovered": rat.get("clues_with_at_least_one_correct"),
                "val_top1_em": val.get("top1_em"),
                "val_pass_at_n": val.get("pass_at_n"),
            }
        )
    return out


# --------------------------------------------------------------------------
# markdown rendering
# --------------------------------------------------------------------------

def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.2f}%"


def _table(title: str, data: dict[str, dict]) -> list[str]:
    if not data:
        return []
    lines = [f"### {title}", "", "| group | n | top-1 EM | pass@N |", "|---|---:|---:|---:|"]
    for k, v in data.items():
        lines.append(f"| {k} | {v['n']} | {_pct(v['top1_em'])} | {_pct(v['pass_at_n'])} |")
    lines.append("")
    return lines


def to_markdown(report: dict, curve: Sequence[dict] | None = None) -> str:
    h = report["headline"]
    L: list[str] = [
        "# Generator analysis",
        "",
        f"Evaluated on **{h['n_clues']} unseen clues** from the Cryptonite "
        "answer-disjoint split: none of these answers appear anywhere in training.",
        "",
        "## Headline",
        "",
        "| metric | value | what it means |",
        "|---|---:|---|",
        f"| top-1 EM | {_pct(h['top1_em'])} | accuracy on a new clue (vs 7.64% T5-Large baseline) |",
        f"| pass@N | {_pct(h['pass_at_n'])} | ceiling for a perfect re-ranker |",
        f"| oracle gap | {_pct(h['oracle_gap'])} | head-room available to Best-of-N |",
        f"| majority-vote EM | {_pct(h['majority_vote_em'])} | discriminator-free baseline to beat |",
        f"| majority-vote gain | {_pct(h['majority_vote_gain_over_top1'])} | what self-consistency alone buys |",
        f"| trace precision | {_pct(h['trace_precision'])} | correct traces / all {h['n_traces_scored']} traces |",
        f"| length-valid rate | {_pct(h['length_valid_rate'])} | answers that at least fit the enumeration |",
        f"| definition grounded | {_pct(h['definition_grounded_rate'])} | predicted definition really is in the clue |",
        f"| unparseable | {_pct(h['unparseable_rate'])} | no answer field emitted at all |",
        f"| alt-answer hits | {_pct(h.get('alt_answer_rate'))} | scored wrong, but matched a documented alternative |",
        "",
        f"No sampled trace solved **{report['hardest']['n_unsolved_by_any_sample']}** clues "
        f"({_pct(report['hardest']['share'])}) - the residue that Best-of-N cannot fix.",
        "",
    ]

    fr = report["failure_reasons"]
    if fr["distribution"]:
        L += ["## Why top-1 was wrong", "", "| reason | count | share |", "|---|---:|---:|"]
        for k, v in fr["distribution"].items():
            L.append(f"| {k} | {v} | {_pct(fr['share'].get(k))} |")
        L.append("")

    L += ["## Breakdowns", ""]
    L += _table("By answer length", report["by_answer_length"])
    L += _table("By clue length", report["by_clue_length"])
    L += _table("By answer word count", report["by_answer_words"])
    L += _table("By sample agreement (calibration)", report["by_sample_agreement"])
    L += _table("By publisher", report["by_publisher"])

    if curve:
        L += [
            "## STaR learning curve",
            "",
            "| iter | train examples | new | train pass@N | val top-1 EM | val pass@N |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in curve:
            L.append(
                f"| {r['iteration']} | {r['train_examples']} | {r['new_examples']} | "
                f"{_pct(r['train_pass_at_n'])} | {_pct(r['val_top1_em'])} | "
                f"{_pct(r['val_pass_at_n'])} |"
            )
        L += [
            "",
            "Train examples rising while val EM stays flat means self-generated data has "
            "stopped adding information - stop iterating.",
            "",
        ]

    if report["correct_examples"]:
        L += ["## Solved examples", ""]
        for e in report["correct_examples"][:5]:
            L.append(f"- **{e['clue']}** -> `{e['gold']}`  \n  _{e['wordplay']}_")
        L.append("")

    if report["error_examples"]:
        L += ["## Error examples", ""]
        for e in report["error_examples"][:10]:
            L.append(
                f"- **{e['clue']}** ({e['enumeration']}) gold `{e['gold']}`, "
                f"got `{e['predicted']}` [{e['reason']}]  \n  _{e['wordplay']}_"
            )
        L.append("")

    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        help="predictions.jsonl from evaluate.py (or runs/x/test_predictions.jsonl)",
    )
    parser.add_argument("--history", default=None, help="runs/x/history.json, optional")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--error-examples", type=int, default=25)
    args = parser.parse_args(argv)

    setup_logging()
    rows = list(read_jsonl(args.predictions))
    report = analyse(rows, n_error_examples=args.error_examples)

    curve = None
    if args.history and Path(args.history).exists():
        curve = learning_curve(json.loads(Path(args.history).read_text(encoding="utf-8")))
        report["learning_curve"] = curve

    out = Path(args.out)
    write_json(out / "analysis.json", report)
    (out / "analysis.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "analysis.md").write_text(to_markdown(report, curve), encoding="utf-8")

    h = report["headline"]
    log.info("=" * 64)
    log.info("clues (unseen)      %d", h["n_clues"])
    log.info("top-1 EM            %s", _pct(h["top1_em"]))
    log.info("pass@N              %s", _pct(h["pass_at_n"]))
    log.info("oracle gap          %s  <- head-room for Best-of-N", _pct(h["oracle_gap"]))
    log.info("majority-vote EM    %s  <- baseline the discriminator must beat",
             _pct(h["majority_vote_em"]))
    log.info("trace precision     %s", _pct(h["trace_precision"]))
    log.info("=" * 64)
    log.info("wrote %s and %s", out / "analysis.json", out / "analysis.md")


if __name__ == "__main__":
    main()
