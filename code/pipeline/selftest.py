"""End-to-end test of everything except the two models.

torch and transformers are stubbed, and the generator/scorer are replaced with
deterministic fakes, so this needs no GPU stack and downloads nothing. A few
seconds of CPU; wall clock varies with filesystem state. It exercises the parts
most likely to be wrong and hardest to notice: filtering, selection, resume,
metrics arithmetic, the scorer verification, and the artifact writers.

    python code/pipeline/selftest.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
import types
from pathlib import Path

# --------------------------------------------------------------------------
# Stub torch / transformers so adapters.py imports without a GPU stack.
# --------------------------------------------------------------------------
if "torch" not in sys.modules:
    torch = types.ModuleType("torch")

    class _Device:
        def __init__(self, kind="cpu"):
            self.type = kind

        def __repr__(self):
            return f"device({self.type})"

    torch.device = _Device
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, is_bf16_supported=lambda: False, get_device_name=lambda i: ""
    )
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.no_grad = lambda: (lambda fn: fn)
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.__version__ = "stub"
    sys.modules["torch"] = torch

    transformers = types.ModuleType("transformers")

    class _Auto:
        @staticmethod
        def from_pretrained(*a, **k):
            raise AssertionError("selftest must not load a real model")

    transformers.AutoTokenizer = _Auto
    transformers.AutoModelForSeq2SeqLM = _Auto
    transformers.AutoModelForSequenceClassification = _Auto
    transformers.__version__ = "5.17.0"   # matches what setup_env.sh installs
    sys.modules["transformers"] = transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config_io, data, metrics as metrics_mod, pipeline  # noqa: E402
from pipeline.adapters import Candidate, ClueRecord, matches_enumeration, normalize_answer  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        FAILURES.append(label)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class FakeGenerator:
    """Emits the gold answer at a controlled rank, plus plausible distractors.

    `gold_rank` drives the whole test: at 0 the baseline is already right (the
    reranker can only lose), at 2 the gold answer is present but not top-1 (the
    reranker can win), at None it is absent (oracle@k must equal 0).
    """

    description = "fake-generator"

    def __init__(self, gold_rank: int | None = 2, k: int = 5, wrong_length: bool = False):
        self.gold_rank = gold_rank
        self.k = k
        self.wrong_length = wrong_length
        self.stats = {"decoded": 0, "reason_parsed": 0, "unparsed": 0}

    def generate(self, records):
        for record in records:
            cands = []
            for i in range(self.k):
                if self.gold_rank is not None and i == self.gold_rank and record.gold_answer:
                    answer = record.gold_answer
                elif self.wrong_length and i == 0:
                    answer = "z" * 39  # must be killed by the enumeration filter
                else:
                    answer = f"wrong{i}"
                cands.append(
                    Candidate(
                        answer=answer,
                        reason=f"reason for {answer}",
                        raw=answer,
                        gen_rank=i,
                        gen_score=-1.0 - i * 0.1,
                    )
                )
            record.candidates = cands
            self.stats["decoded"] += self.k
            self.stats["reason_parsed"] += self.k


class OracleScorer:
    """A perfect scorer: 0.99 for the gold answer, low noise otherwise."""

    description = "fake-oracle-scorer"

    def score(self, records):
        rng = random.Random(0)
        for record in records:
            for cand in record.candidates:
                if not cand.alive:
                    continue
                right = record.gold_answer and normalize_answer(cand.answer) == normalize_answer(
                    record.gold_answer
                )
                cand.score = 0.99 if right else rng.uniform(0.0, 0.4)


class AdversarialScorer:
    """A perfectly wrong scorer: always ranks the gold answer last."""

    description = "fake-adversarial-scorer"

    def score(self, records):
        for record in records:
            for cand in record.candidates:
                if not cand.alive:
                    continue
                right = record.gold_answer and normalize_answer(cand.answer) == normalize_answer(
                    record.gold_answer
                )
                cand.score = 0.01 if right else 0.9


def fake_records(n: int = 40) -> list[ClueRecord]:
    return [
        ClueRecord(
            id=f"t-{i:03d}",
            clue=f"some cryptic clue number {i} (6)",
            enumeration="(6)",
            gold_answer="charge",
            gold_reason="CH(chapter) + ARGE",
        )
        for i in range(n)
    ]


def run_pipeline_with(gen, scorer, cfg, records, tmp: Path, **kw):
    pipeline.build_generator = lambda c, d=None: gen
    pipeline.build_scorer = lambda c, d=None: scorer
    return pipeline.run(records, cfg, tmp, print_first=0, progress_every=0, stream=_Quiet(), **kw)


class _Quiet:
    def write(self, *_a):
        pass

    def flush(self):
        pass


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_helpers():
    print("\nhelpers")
    check(normalize_answer("Running Buffet.") == "runningbuffet", "normalize_answer collapses case/space/punct")
    check(normalize_answer("hard-wood") == normalize_answer("hard wood"), "hyphen == space")
    check(matches_enumeration("charge", "(6)"), "6-letter answer fits (6)")
    check(not matches_enumeration("charger", "(6)"), "7-letter answer rejected by (6)")
    check(matches_enumeration("running buffet", "(7,6)"), "multiword fits (7,6)")
    check(matches_enumeration("runningbuffet", "(7,6)"), "total-length match accepted (data artifact)")
    check(matches_enumeration("anything", None), "no enumeration => no filtering")
    check(not matches_enumeration("", "(6)"), "empty answer never fits")


def test_config():
    print("\nconfig")
    cfg = config_io.load_config("default")
    check(cfg["generator"]["strategy"] == "beam_sample_union", "default config parses (with // comments)")
    child = config_io.load_config("t5large_deberta_large")
    check("hugTAU" in child["generator"]["model"], "extends resolves the generator override")
    check(
        child["scorer"]["template"] == cfg["scorer"]["template"],
        "extends inherits un-overridden fields",
    )
    over = config_io.apply_overrides(child, ["generator.num_candidates=20", "selection.combine=weighted"])
    check(over["generator"]["num_candidates"] == 20, "--set parses an int")
    check(over["selection"]["combine"] == "weighted", "--set parses a string")
    check(child["generator"]["num_candidates"] == 12, "--set does not mutate the input config")
    for name in ("default", "t5large_deberta_large", "smoke"):
        config_io.load_config(name)
    check(True, "every shipped config loads")

    # The delivered checkpoint is deberta-v3-large, which needs a smaller batch
    # than the small model it replaced, fp32 to avoid perturbing near-tied
    # candidate scores, and NO base-model fallback (a random head scores noise).
    sc = child["scorer"]
    check("trained_deberta" in sc["model"], "scorer points at the delivered checkpoint")
    check("deberta-v3-small" not in sc["model"] and "deberta-v3-large" not in sc["model"],
          "no bare base-model fallback in the scorer spec", sc["model"])
    check(sc["batch_size"] <= 16, "batch size reduced for a large scorer", str(sc["batch_size"]))
    check(sc["dtype"] == "float32", "scorer pinned to fp32", str(sc["dtype"]))
    check(sc["max_length"] <= 512, "max_length within the checkpoint's position embeddings")
    check(sc["template"] == "CLUE: {clue}\nANSWER: {answer}\nREASONING: {reason}",
          "scorer template inherited unchanged from the fine-tuning layout")
    check(sc.get("positive_label") is None, "positive class read from the checkpoint's label2id")

    from pipeline.paths import resolve_model_path
    resolved = resolve_model_path(sc["model"])
    check(Path(resolved).is_dir(), "scorer path resolves to a real directory on disk", resolved)
    check((Path(resolved) / "config.json").exists(), "resolved checkpoint has a config.json")


def test_dataset():
    print("\ndataset")
    try:
        recs = data.load_split("test", limit=25, seed=1)
    except FileNotFoundError as exc:
        check(False, "test split loads", str(exc))
        return
    check(len(recs) == 25, "limit honoured", f"got {len(recs)}")
    check(all(r.gold_answer for r in recs), "every row has a gold answer")
    check(all(r.clue for r in recs), "every row has clue text")

    a = [r.id for r in data.load_split("test", limit=25, seed=1)]
    b = [r.id for r in data.load_split("test", limit=25, seed=1)]
    c = [r.id for r in data.load_split("test", limit=25, seed=2)]
    check(a == b, "sampling is reproducible for a fixed seed")
    check(a != c, "a different seed draws a different subset")

    shards = [
        {r.id for r in data.load_split("test", limit=40, seed=1, shard=(i, 4))} for i in range(4)
    ]
    union = set().union(*shards)
    check(sum(len(s) for s in shards) == 40, "shards partition without loss")
    check(len(union) == 40, "shards do not overlap")
    check(union == set(a) | {r.id for r in data.load_split("test", limit=40, seed=1)} - set(a),
          "sharded coverage equals unsharded coverage")

    adhoc = data.single_clue("attack general at end of month (6)")
    check(adhoc.enumeration == "(6)", "enumeration inferred from clue text", adhoc.enumeration or "")

    try:
        data.load_split("test", limit=5, require_reason=True)
        check(False, "require_reason on test raises")
    except ValueError:
        check(True, "require_reason on test raises a clear error (no annotations there)")


def test_filters():
    print("\nfilters")
    cfg = config_io.load_config("default")
    stats = pipeline.FilterStats()
    rec = ClueRecord(id="x", clue="c (6)", enumeration="(6)", gold_answer="charge")
    rec.candidates = [
        Candidate("charge", "r", "charge", 0),
        Candidate("charger", "r", "charger", 1),
        Candidate("", "r", "", 2),
        Candidate("ch", "r", "ch", 3),
    ]
    pipeline.apply_filters(rec, cfg["filters"], stats)
    alive = [c.answer for c in rec.candidates if c.alive]
    check(alive == ["charge"], "wrong-length, empty and short answers dropped", str(alive))
    check(stats.dropped_enumeration == 2, "enumeration drops counted", str(stats.as_dict()))

    # Every candidate bad -> must rescue, or the clue vanishes from the metrics.
    stats2 = pipeline.FilterStats()
    rec2 = ClueRecord(id="y", clue="c (6)", enumeration="(6)", gold_answer="charge")
    rec2.candidates = [Candidate("toolongforsix", "r", "", 0), Candidate("ab", "r", "", 1)]
    pipeline.apply_filters(rec2, cfg["filters"], stats2)
    check(all(c.alive for c in rec2.candidates), "all-dropped clue is rescued, never left unanswered")
    check(stats2.rescued_clues == 1, "rescue is counted, not hidden")
    check(rec2.rescued is True, "rescue is flagged on the record, so the report can say so")
    check(ClueRecord.from_dict(rec2.to_dict()).rescued is True, "rescue flag survives JSONL round-trip")
    check(rec.rescued is False, "a clue with survivors is not marked rescued")


def test_selection():
    print("\nselection")
    rec = ClueRecord(id="z", clue="c", gold_answer="charge")
    rec.candidates = [
        Candidate("wrong", "r", "", 0, gen_score=-0.5, score=0.30),
        Candidate("charge", "r", "", 1, gen_score=-3.0, score=0.95),
    ]
    best = pipeline.select_best(rec, {"combine": "scorer"})
    check(best.answer == "charge", "scorer mode picks the highest score")

    best = pipeline.select_best(rec, {"combine": "generator"})
    check(best.answer == "wrong", "generator mode reproduces the baseline (ablation)")

    best = pipeline.select_best(rec, {"combine": "weighted", "scorer_weight": 0.7})
    check(best.answer == "charge", "weighted mode leans on the scorer at w=0.7")
    best = pipeline.select_best(rec, {"combine": "weighted", "scorer_weight": 0.0})
    check(best.answer == "wrong", "weighted mode at w=0 equals the generator")

    tie = ClueRecord(id="t", clue="c")
    tie.candidates = [
        Candidate("second", "r", "", 3, gen_score=-1.0, score=0.5),
        Candidate("first", "r", "", 1, gen_score=-1.0, score=0.5),
    ]
    check(
        pipeline.select_best(tie, {"combine": "scorer"}).answer == "first",
        "ties break toward the lower generator rank",
    )

    empty = ClueRecord(id="e", clue="c")
    check(pipeline.select_best(empty, {}) is None, "no candidates => no selection, no crash")


def test_metrics_arithmetic():
    print("\nmetrics")
    cfg = config_io.load_config("default")
    tmp = Path(tempfile.mkdtemp())

    try:
        # Gold at rank 2 with a perfect scorer: baseline wrong, pipeline right.
        out = run_pipeline_with(FakeGenerator(gold_rank=2), OracleScorer(), cfg, fake_records(40), tmp / "a")
        m = metrics_mod.compute_metrics(out.records, cfg["selection"])
        check(m["accuracy"]["baseline_generator_top1"] == 0.0, "baseline is 0 when gold is never rank 0")
        check(m["accuracy"]["pipeline"] == 1.0, "perfect scorer recovers every clue")
        check(m["accuracy"]["oracle_at_k"] == 1.0, "oracle@k is 1 when gold is always present")
        check(m["improvement"]["rerank_wins"] == 40, "all 40 disagreements are wins")
        check(m["improvement"]["rerank_losses"] == 0, "no losses")
        check(m["improvement"]["mcnemar_p_value"] < 1e-6, "40-0 is significant")
        check(abs(m["improvement"]["headroom_captured"] - 1.0) < 1e-9, "all headroom captured")
        check(m["scorer"]["candidate_auc"] == 1.0, "AUC is 1 for a perfect scorer")
        check(m["recall_at_k"][1] == 0.0 and m["recall_at_k"][3] == 1.0, "recall@k steps up at k=3")

        # Adversarial scorer: strictly worse than baseline, and must be reported so.
        out = run_pipeline_with(FakeGenerator(gold_rank=0), AdversarialScorer(), cfg, fake_records(20), tmp / "b")
        m = metrics_mod.compute_metrics(out.records, cfg["selection"])
        check(m["accuracy"]["baseline_generator_top1"] == 1.0, "baseline right when gold is rank 0")
        check(m["accuracy"]["pipeline"] == 0.0, "adversarial scorer breaks every clue")
        check(m["improvement"]["delta_accuracy"] == -1.0, "delta is negative, not hidden")
        check(m["improvement"]["rerank_losses"] == 20, "losses counted")
        check(m["scorer"]["candidate_auc"] == 0.0, "AUC is 0 for an inverted scorer")

        # Gold absent: the ceiling itself must be 0.
        out = run_pipeline_with(FakeGenerator(gold_rank=None), OracleScorer(), cfg, fake_records(10), tmp / "c")
        m = metrics_mod.compute_metrics(out.records, cfg["selection"])
        check(m["accuracy"]["oracle_at_k"] == 0.0, "oracle@k is 0 when gold is never generated")
        check(m["accuracy"]["pipeline"] == 0.0, "cannot pick an answer that was never proposed")

        # No gold at all (the --clue case) must not divide by zero.
        plain = [ClueRecord(id="n1", clue="a clue (6)", enumeration="(6)")]
        out = run_pipeline_with(FakeGenerator(gold_rank=None), OracleScorer(), cfg, plain, tmp / "d")
        m = metrics_mod.compute_metrics(out.records, cfg["selection"])
        check(m["n_with_gold"] == 0, "no-gold run reports n_with_gold=0 instead of crashing")
        check("no gold" in metrics_mod.format_metrics(m).lower(), "no-gold summary says so")

        check(metrics_mod.mcnemar_exact(0, 0) == 1.0, "McNemar with no disagreements is p=1")
        check(abs(metrics_mod.mcnemar_exact(5, 5) - 1.0) < 1e-9, "McNemar symmetric case is p=1")
        check(metrics_mod.mcnemar_exact(10, 0) < 0.01, "10-0 is significant")
        lo, hi = metrics_mod.wilson_interval(50, 100)
        check(lo < 0.5 < hi and 0 <= lo and hi <= 1, "Wilson interval brackets the estimate")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume_and_artifacts():
    print("\nresume + artifacts")
    cfg = config_io.load_config("default")
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir = tmp / "run"
        records = fake_records(20)

        # First pass over half the clues, then resume over all of them.
        run_pipeline_with(FakeGenerator(gold_rank=1), OracleScorer(), cfg, records[:10], run_dir)
        first_lines = (run_dir / "records.jsonl").read_text().count("\n")

        gen = FakeGenerator(gold_rank=1)
        out = run_pipeline_with(gen, OracleScorer(), cfg, fake_records(20), run_dir)
        check(first_lines == 10, "first pass wrote 10 records", str(first_lines))
        check(out.resumed == 10, "resume skipped the 10 finished clues", str(out.resumed))
        check(gen.stats["decoded"] == 10 * 5, "resume regenerated only the missing 10 clues")
        check(len(out.records) == 20, "resumed run returns all 20 records")
        check([r.id for r in out.records] == [f"t-{i:03d}" for i in range(20)],
              "resumed records come back in the caller's order")

        # A truncated final line (a job killed mid-write) must not poison resume.
        with open(run_dir / "records.jsonl", "a") as fh:
            fh.write('{"id": "t-999", "clue": "trunc')
        out2 = run_pipeline_with(FakeGenerator(gold_rank=1), OracleScorer(), cfg, fake_records(20), run_dir)
        check(len(out2.records) == 20, "truncated trailing line is ignored on resume")

        # --candidates-from: score an existing generation without regenerating.
        rescore_dir = tmp / "rescore"
        gen2 = FakeGenerator(gold_rank=1)
        out3 = run_pipeline_with(
            gen2, AdversarialScorer(), cfg, fake_records(20), rescore_dir,
            stage="score", candidates_from=run_dir,
        )
        check(gen2.stats["decoded"] == 0, "stage=score never invokes the generator")
        check(len(out3.records) == 20, "rescored every clue from the cached candidates")
        m = metrics_mod.compute_metrics(out3.records, cfg["selection"])
        check(m["accuracy"]["pipeline"] == 0.0, "rescoring applied the NEW scorer, not the cached scores")

        m = metrics_mod.compute_metrics(out.records, cfg["selection"])
        metrics_mod.write_metrics_json(m, run_dir / "metrics.json")
        reloaded = json.loads((run_dir / "metrics.json").read_text())
        check(reloaded["n_with_gold"] == 20, "metrics.json round-trips")
        check(
            "nan" not in (run_dir / "metrics.json").read_text().lower(),
            "metrics.json contains no bare NaN (would be invalid JSON)",
        )

        metrics_mod.write_predictions_csv(out.records, run_dir / "predictions.csv", cfg["selection"])
        rows = (run_dir / "predictions.csv").read_text().strip().split("\n")
        check(len(rows) == 21, "predictions.csv has a header plus one row per clue", str(len(rows)))

        figs = metrics_mod.write_plots(out.records, m, run_dir / "figures")
        check(len(figs) == 3, "three figures written", str([f.name for f in figs]))
        check(all(f.exists() and f.stat().st_size > 1000 for f in figs), "figures are non-trivial files")

        # The stage report must render without exploding on odd input.
        import io

        buf = io.StringIO()
        rec = out.records[0]
        pipeline.print_stage_report(rec, pipeline.select_best(rec, cfg["selection"]),
                                    pipeline.baseline_candidate(rec), buf)
        text = buf.getvalue()
        check("stage 1" in text and "stage 2" in text and "stage 3" in text, "report shows all three stages")
        check("RERANK WIN" in text, "report flags a rerank win")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_decode_passes():
    """The generate() kwargs, without loading a model.

    These are pure config->kwargs translation, and getting them wrong fails only
    at generation time -- i.e. after a job has queued. Worth pinning down here.
    """
    print("\ndecode passes")
    from pipeline.adapters import Seq2SeqGenerator

    def gen(**over):
        g = Seq2SeqGenerator.__new__(Seq2SeqGenerator)
        g.cfg = {**over}
        g.num_candidates = over.get("num_candidates", 12)
        g.max_new_tokens = 32
        g.strategy = over.get("strategy", "beam_sample_union")
        g.device = types.SimpleNamespace(type=over.get("device", "cuda"))
        return g

    passes = gen(num_candidates=12, beam_candidates=6)._decode_passes()
    labels = [p[0] for p in passes]
    check(labels == ["beam", "sample"], "beam_sample_union runs beam then sample", str(labels))
    check(
        sum(int(p[1]["num_return_sequences"]) for p in passes) == 12,
        "union pass sizes sum to num_candidates",
    )
    check(passes[0][1]["do_sample"] is False, "the beam pass does not sample")
    check(passes[1][1]["do_sample"] is True, "the sample pass does sample")
    check(
        all(p[1]["return_dict_in_generate"] and p[1]["output_scores"] for p in passes),
        "every pass asks for scores (they define the baseline)",
    )

    single = gen(strategy="beam", num_candidates=7)._decode_passes()
    check(len(single) == 1 and single[0][1]["num_beams"] == 7, "beam strategy is one pass")
    single = gen(strategy="sample", num_candidates=7)._decode_passes()
    check(len(single) == 1 and single[0][1]["do_sample"] is True, "sample strategy is one pass")

    # num_beams must be a multiple of num_beam_groups; an awkward config must be
    # snapped to a divisor rather than raising at generation time.
    g = gen(strategy="diverse_beam", num_candidates=12, num_beam_groups=5, allow_remote_code=True)
    kw = g._decode_passes()[0][1]
    check(kw["num_beams"] % kw["num_beam_groups"] == 0,
          "diverse_beam group count snapped to a divisor", f"{kw['num_beams']}/{kw['num_beam_groups']}")
    check(kw.get("trust_remote_code") is True, "diverse_beam opts into remote code when allowed")

    # ...and without that opt-in it must refuse, with instructions.
    try:
        gen(strategy="diverse_beam", num_candidates=12)._decode_passes()
        check(False, "diverse_beam refuses without allow_remote_code")
    except ValueError as exc:
        check("beam_sample_union" in str(exc),
              "diverse_beam refusal names the alternative", str(exc)[:60])

    try:
        gen(strategy="nonsense")._decode_passes()
        check(False, "unknown strategy rejected")
    except ValueError:
        check(True, "unknown strategy rejected with a clear error")

    # The MPS-only transformers bug: non-beam paths need use_cache=False there.
    g = gen(device="mps")
    beam_kw = g._apply_device_workarounds(g._decode_passes()[0][1])
    samp_kw = g._apply_device_workarounds(g._decode_passes()[1][1])
    check("use_cache" not in beam_kw, "mps workaround leaves beam passes alone")
    check(samp_kw.get("use_cache") is False, "mps workaround disables cache for sampling")
    g = gen(device="cuda")
    check("use_cache" not in g._apply_device_workarounds(g._decode_passes()[1][1]),
          "no workaround applied off mps")


def test_verify_scorer():
    print("\nverify-scorer")
    from pipeline import adapters, cli

    cfg = config_io.load_config("t5large_deberta_large")

    class FakeVerifyScorer:
        """Separates training-style negatives well, rerank-style poorly.

        That asymmetry is the realistic case and the one the verdict text must
        call out, so it is what the test asserts on.
        """

        description = "fake-verify-scorer"

        def score(self, records):
            for r in records:
                r.candidates[0].score = 0.90   # human reasoning
                r.candidates[1].score = 0.10   # other clue's answer + reasoning
                r.candidates[2].score = 0.88   # gold answer, other reasoning: nearly tied

    real = adapters.build_scorer
    adapters.build_scorer = lambda c, d=None: FakeVerifyScorer()
    try:
        import contextlib, io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = cli.verify_scorer(cfg, count=30, seed=0)
        text = buf.getvalue()
    finally:
        adapters.build_scorer = real

    check(res["n_clues"] == 30, "verify-scorer used the requested clue count", str(res["n_clues"]))
    check(res["training_style_negative"]["pairwise_accuracy"] == 1.0,
          "training-style separation reported")
    check(res["training_style_negative"]["auc"] == 1.0, "training-style AUC computed")
    check(res["rerank_style_negative"]["pairwise_accuracy"] == 1.0,
          "rerank-style comparison reported separately")
    check("VERDICT" in text, "verify-scorer prints a verdict")
    check("training-style" in text and "rerank-style" in text, "both negative types shown")

    p = cli.build_parser()
    check(p.parse_args(["--verify-scorer"]).verify_scorer == 200, "--verify-scorer defaults to 200")
    check(p.parse_args(["--verify-scorer", "50"]).verify_scorer == 50, "--verify-scorer takes a count")
    check(p.parse_args(["--clue", "x"]).verify_scorer is None, "--verify-scorer off by default")


def test_parse_and_cli():
    print("\nparsing + CLI")
    from pipeline import cli
    from pipeline.adapters import Seq2SeqGenerator

    cfg = config_io.load_config("default")

    # Exercise parse_output without constructing a real model.
    gen = Seq2SeqGenerator.__new__(Seq2SeqGenerator)
    gen.parse_patterns = [__import__("re").compile(p) for p in cfg["generator"]["parse"]["patterns"]]
    gen.parse_fallback = "answer_only"
    gen.strip_enumeration = True
    gen.stats = {"decoded": 0, "reason_parsed": 0, "unparsed": 0}

    cases = [
        ("Answer: charge | Reason: CH + ARGE", ("charge", "CH + ARGE")),
        ("charge | CH(chapter) + ARGE", ("charge", "CH(chapter) + ARGE")),
        ("charge", ("charge", "")),
        ("CHARGE (6)", ("charge", "")),
        ("  Charge.  ", ("charge", "")),
    ]
    for raw, expected in cases:
        got = gen.parse_output(raw, "(6)")
        check(got == expected, f"parse {raw!r}", f"got {got}")
    check(gen.stats["reason_parsed"] == 2, "reason-parse counter tracks real matches", str(gen.stats))

    p = cli.build_parser()
    args = p.parse_args(["--clue", "x (6)"])
    check(args.clue == ["x (6)"] and args.stage == "all", "CLI parses a single clue")
    check(cli.parse_shard("2/8") == (2, 8), "--shard parses")
    for bad in (["--clue", "a", "--split", "test"], []):
        try:
            cli.resolve_records(p.parse_args(bad))
            check(False, f"rejects {bad}")
        except SystemExit:
            check(True, f"rejects {bad or '(no input)'}")
    check(len(cli.resolve_records(p.parse_args(["--clue", "a (4)", "--clue", "b (5)"]))) == 2,
          "multiple --clue flags accepted")
    snap = cli.environment_snapshot()
    check("python" in snap and "timestamp" in snap, "environment snapshot populated")


def main() -> int:
    print("=" * 70)
    print("pipeline selftest (torch/transformers stubbed, models faked)")
    print("=" * 70)
    for test in (
        test_helpers,
        test_config,
        test_dataset,
        test_filters,
        test_selection,
        test_metrics_arithmetic,
        test_resume_and_artifacts,
        test_decode_passes,
        test_verify_scorer,
        test_parse_and_cli,
    ):
        test()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
