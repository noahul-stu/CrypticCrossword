"""The two swappable pipeline stages, behind two narrow interfaces.

    Generator.generate(clues)   -> list[list[Candidate]]     stage 1
    Scorer.score(clue, cands)   -> fills Candidate.score     stage 2

Everything model-specific -- prompt text, decode strategy, output parsing, the
scorer's input template, which logit is the positive class -- is a config field,
never a code path. Swapping deberta-v3-small for deberta-v3-large, or flan-t5
for a different seq2seq, is editing JSON. Adding a genuinely different *kind* of
model (a causal LM, an ensemble scorer) means one new class here plus a "kind"
entry in the registry at the bottom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Protocol

from . import paths

paths.setup_environment()  # must precede the transformers import -- see paths.py

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# --------------------------------------------------------------------------
# The unit of data that flows between the stages
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    """One (answer, reason) proposal for a clue, and everything we know about it.

    `gen_score` is the generator's own length-normalized log-probability. It is
    what defines the BASELINE we have to beat: `gen_rank == 0` is exactly what a
    plain T5 with the same decode settings would have returned. Keeping it on
    the candidate means the baseline costs no extra compute.
    """

    answer: str
    reason: str
    raw: str
    gen_rank: int
    gen_score: float | None = None
    score: float | None = None  # scorer's P(correct); None until stage 2
    final_score: float | None = None  # what selection actually sorts on
    dropped: str | None = None  # non-None => filtered out, with the reason why

    @property
    def alive(self) -> bool:
        return self.dropped is None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ClueRecord:
    """One clue plus its candidates: the pipeline's per-example state."""

    id: str
    clue: str
    enumeration: str | None = None
    gold_answer: str | None = None
    gold_reason: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    # Set when every candidate failed the filters and they were all reinstated so
    # the clue still gets an answer. Without surfacing this, the stage report says
    # "N survived filters" for candidates that in fact all failed -- which reads
    # as a passing filter rather than a rescue.
    rescued: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidates"] = [c.to_dict() for c in self.candidates]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ClueRecord":
        cands = [Candidate.from_dict(c) for c in d.get("candidates", [])]
        known = {f for f in cls.__dataclass_fields__} - {"candidates"}
        return cls(candidates=cands, **{k: v for k, v in d.items() if k in known})


class Generator(Protocol):
    def generate(self, records: list[ClueRecord]) -> None:
        """Populate `record.candidates` in place for every record."""


class Scorer(Protocol):
    def score(self, records: list[ClueRecord]) -> None:
        """Set `candidate.score` in place for every surviving candidate."""


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS is what a Mac smoke test gets. Correct, just slow.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dtype_kwargs(device: torch.device, dtype: str | None) -> dict:
    """Build the from_pretrained dtype kwarg, spelled for the installed version.

    transformers >= 5 renamed `torch_dtype=` to `dtype=`. setup_env.sh installs
    >= 5, but a Colab or laptop may well have v4, and passing the wrong spelling
    is either a TypeError or -- worse in v4 -- a silently ignored kwarg that
    leaves the model in fp32 and OOMs. Inspect rather than guess.
    """
    if not dtype or dtype == "auto":
        if device.type != "cuda":
            return {}  # fp32 everywhere but CUDA; bf16 on CPU is slower, not faster
        resolved = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif dtype == "float32":
        return {}
    else:
        resolved = getattr(torch, dtype)

    import inspect

    params = inspect.signature(AutoModelForSeq2SeqLM.from_pretrained).parameters
    key = "dtype" if "dtype" in params else "torch_dtype"
    return {key: resolved}


def normalize_answer(text: str) -> str:
    """Comparison key for answers: letters and digits only, lowercased.

    Gold answers in the united dataset are lowercase and may contain spaces
    ("running buffet"). A generator may emit "RUNNING BUFFET", "running-buffet"
    or "Running Buffet." -- all four must compare equal, or the accuracy number
    is measuring formatting rather than solving.
    """
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def enumeration_lengths(enumeration: str | None) -> list[int] | None:
    """"(4,2)" -> [4, 2]. Returns None when unparseable, meaning "do not filter".

    Commas and hyphens are both word separators for length purposes: "(4-2)" is
    one hyphenated word of 4+2 letters, "(4,2)" is two words -- different
    surface forms, identical letter counts, and letter count is all we check.
    """
    if not enumeration:
        return None
    digits = re.findall(r"\d+", str(enumeration))
    if not digits:
        return None
    return [int(d) for d in digits]


def answer_lengths(answer: str) -> list[int]:
    return [len(w) for w in re.findall(r"[a-z0-9]+", str(answer).lower())]


def matches_enumeration(answer: str, enumeration: str | None) -> bool:
    """Does the answer fit the clue's letter pattern?

    A cheap, model-free filter: an answer of the wrong length is definitionally
    wrong -- it will not fit the grid. Compares the multiset-as-sequence of word
    lengths, but also accepts a total-length match, because "(4,2)" vs "(6)"
    disagreements are a known data artifact (59 rows, see the dataset README's
    enumeration_conflicts) and we would rather keep a right answer than enforce
    a word split the source data itself is inconsistent about.
    """
    expected = enumeration_lengths(enumeration)
    if expected is None:
        return True
    got = answer_lengths(answer)
    if not got:
        return False
    return got == expected or sum(got) == sum(expected)


# --------------------------------------------------------------------------
# Stage 1 -- generation
# --------------------------------------------------------------------------
class Seq2SeqGenerator:
    """Encoder-decoder candidate generator (flan-t5 and friends).

    Produces k candidates per clue. A reranker can only fix a mistake if the right
    answer is somewhere in the list, so the decode strategy is the single biggest
    lever on the pipeline's ceiling -- more so than the scorer. Compare oracle@k
    across strategies before tuning anything else.

      beam_sample_union  DEFAULT. A beam pass then a sampling pass, merged. The
                         two things we need are in tension: the baseline must be
                         what a plain T5 returns (beam), while the pool must be
                         diverse (sampling). Separate passes get both, and beam
                         goes first so the pool's rank 0 is the true beam top-1.
      beam               num_beams=k, return all k. Best single answers, worst
                         diversity -- beams share prefixes, so the k answers are
                         often variants of one guess. Poor oracle@k.
      sample             top-p sampling only. Diverse, noisy, and no
                         well-defined top-1 to use as a baseline.
      diverse_beam       grouped beam search. REMOVED FROM transformers 5 core;
                         needs allow_remote_code=true there, which fetches and
                         executes code from hf.co at generation time.
    """

    def __init__(self, cfg: dict, device: torch.device | None = None):
        self.cfg = cfg
        self.device = device or pick_device(cfg.get("device", "auto"))
        self.model_path = paths.resolve_model_path(cfg["model"])

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path, **_dtype_kwargs(self.device, cfg.get("dtype", "auto"))
        )
        self.model.to(self.device)
        self.model.eval()

        self.prompt_template: str = cfg.get("prompt", "{clue}")
        self.num_candidates: int = int(cfg.get("num_candidates", 12))
        self.batch_size: int = int(cfg.get("batch_size", 8))
        self.max_input_tokens: int = int(cfg.get("max_input_tokens", 128))
        self.max_new_tokens: int = int(cfg.get("max_new_tokens", 64))
        self.strategy: str = cfg.get("strategy", "diverse_beam")

        parse_cfg = cfg.get("parse", {})
        self.parse_patterns = [re.compile(p) for p in parse_cfg.get("patterns", [])]
        self.parse_fallback: str = parse_cfg.get("fallback", "answer_only")
        self.strip_enumeration: bool = bool(parse_cfg.get("strip_enumeration", True))
        # Diagnostic counters, surfaced by the CLI. A high unparsed count on a
        # model you believe emits reasons means the patterns are wrong -- run
        # `--probe` to see the raw decodes.
        self.stats = {"decoded": 0, "reason_parsed": 0, "unparsed": 0}
        self._warned_dropped_inputs = False

        self.description = f"{self.model_path} [{self.strategy}, k={self.num_candidates}]"

    # ---------------------------------------------------------------- prompts
    def build_prompt(self, record: ClueRecord) -> str:
        return self.prompt_template.format(
            clue=record.clue,
            enumeration=record.enumeration or "",
            lengths=",".join(str(n) for n in (enumeration_lengths(record.enumeration) or [])),
        )

    # ------------------------------------------------------------ decode args
    def _diverse_beam_kwargs(self, k: int) -> dict:
        """Grouped beam search kwargs, plus the transformers-5 remote-code dance.

        transformers 5 REMOVED group beam search from core: it now lives in the
        `transformers-community/group-beam-search` custom_generate repo and needs
        `trust_remote_code=True`, which downloads and executes code from hf.co at
        generation time. That is a bad default here -- a compute node may have no
        outbound route (the repo's own sbatch warns about exactly this), so the
        failure would land mid-job after queueing.

        So on transformers >= 5 this raises unless the config opts in explicitly
        with `allow_remote_code: true`. Use `beam_sample_union` instead; it gets
        comparable diversity from core-only features.
        """
        # Hard constraints from generate(): num_beams must be a multiple of
        # num_beam_groups, and num_return_sequences <= num_beams. Snap the group
        # count to a divisor rather than letting transformers raise -- a config of
        # k=12, groups=5 is a reasonable thing to write and should not be a crash.
        groups = int(self.cfg.get("num_beam_groups", 0)) or max(2, k // 2)
        groups = min(groups, k)
        while k % groups:
            groups -= 1

        kwargs = dict(
            num_beams=k,
            num_beam_groups=groups,
            diversity_penalty=float(self.cfg.get("diversity_penalty", 0.7)),
            do_sample=False,
            early_stopping=True,
        )

        import transformers

        # Assume the restrictive (>=5) behaviour when the version is unparseable:
        # erroring with instructions beats sending trust_remote_code=True to a
        # transformers that does not expect it.
        try:
            major = int(str(transformers.__version__).split(".")[0])
        except (TypeError, ValueError):
            major = 5
        if major >= 5:
            if not self.cfg.get("allow_remote_code", False):
                raise ValueError(
                    f"strategy='diverse_beam' is unavailable: transformers "
                    f"{transformers.__version__} moved group beam search out of core into "
                    "the remote `transformers-community/group-beam-search` repo, which "
                    "requires trust_remote_code=True and network access to hf.co at "
                    "generation time.\n"
                    "  Recommended:  --set generator.strategy=beam_sample_union\n"
                    "                (core-only, and gives a cleaner baseline anyway)\n"
                    "  Or opt in:    --set generator.allow_remote_code=true\n"
                    "                (needs an outbound route from the compute node)"
                )
            kwargs["custom_generate"] = "transformers-community/group-beam-search"
            kwargs["trust_remote_code"] = True

        return kwargs

    def _apply_device_workarounds(self, kwargs: dict) -> dict:
        """Work around a transformers-on-MPS bug in the sampling/greedy paths.

        On Apple MPS, transformers 5.17 raises
            AttributeError: 'EncoderDecoderCache' object has no attribute 'layers'
        from its own generate() internals whenever `return_dict_in_generate=True`
        is combined with a non-beam decoding path. Verified matrix:

            device   sample+dict   greedy+dict   beam+dict
            cpu      ok            ok            ok
            mps      FAIL          FAIL          ok

        `use_cache=False` avoids it. This is MPS-only -- CUDA behaves like CPU --
        so the cluster is unaffected and only the laptop smoke-test path pays the
        (irrelevant there) speed cost. We need `return_dict_in_generate` because
        that is how generation log-probs come back, and those define the baseline.
        """
        if self.device.type == "mps" and int(kwargs.get("num_beams", 1)) <= 1:
            kwargs = dict(kwargs, use_cache=False)
        return kwargs

    def _decode_passes(self) -> list[tuple[str, dict]]:
        """One or more generate() calls whose results are merged into one pool.

        Returning a LIST is what lets `beam_sample_union` exist: beam search
        supplies an honest rank-0 baseline, and a separate sampling pass supplies
        the diversity a reranker needs. Those two goals conflict inside a single
        decode call, which is why splitting them is better design than a lucky
        choice of diversity_penalty -- not merely a workaround for the
        transformers-5 removal above.
        """
        k = self.num_candidates
        common = dict(
            max_new_tokens=self.max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )
        sampling = dict(
            do_sample=True,
            top_p=float(self.cfg.get("top_p", 0.95)),
            temperature=float(self.cfg.get("temperature", 1.0)),
        )

        if self.strategy == "sample":
            return [("sample", dict(common, num_return_sequences=k, **sampling))]

        if self.strategy == "beam":
            return [
                (
                    "beam",
                    dict(
                        common,
                        num_return_sequences=k,
                        num_beams=k,
                        do_sample=False,
                        early_stopping=True,
                    ),
                )
            ]

        if self.strategy == "diverse_beam":
            return [
                ("diverse_beam", dict(common, num_return_sequences=k, **self._diverse_beam_kwargs(k)))
            ]

        if self.strategy == "beam_sample_union":
            # Beam first, so the merged pool's rank 0 is the beam top-1 -- exactly
            # what a plain T5 would return, which is the baseline every metric is
            # measured against. Sampling after, for candidates beam search would
            # never surface because its beams share prefixes.
            beam_n = int(self.cfg.get("beam_candidates", 0)) or max(2, (k + 1) // 2)
            beam_n = max(1, min(beam_n, k))
            sample_n = max(0, k - beam_n)
            passes = [
                (
                    "beam",
                    dict(
                        common,
                        num_return_sequences=beam_n,
                        num_beams=max(beam_n, 2),
                        do_sample=False,
                        early_stopping=True,
                    ),
                )
            ]
            if sample_n:
                passes.append(("sample", dict(common, num_return_sequences=sample_n, **sampling)))
            return passes

        raise ValueError(
            f"unknown generation strategy {self.strategy!r} (expected "
            "'beam_sample_union', 'beam', 'sample' or 'diverse_beam')"
        )

    # ------------------------------------------------------------------ parse
    def parse_output(self, text: str, enumeration: str | None) -> tuple[str, str]:
        """Split one raw decode into (answer, reason).

        Tries each configured regex in order, expecting named groups `answer`
        and `reason`. If none match, `parse_fallback` decides:

          answer_only  the whole decode is the answer, reason "". Correct for a
                       clue->answer checkpoint, and the safe default: an empty
                       reason still gives the scorer a valid (clue, answer) pair.
          first_line   line 1 is the answer, the rest is the reason.
        """
        text = text.strip()
        self.stats["decoded"] += 1

        for pattern in self.parse_patterns:
            m = pattern.search(text)
            if m:
                answer = (m.groupdict().get("answer") or "").strip()
                reason = (m.groupdict().get("reason") or "").strip()
                if answer:
                    self.stats["reason_parsed"] += 1
                    return self._clean_answer(answer, enumeration), reason

        self.stats["unparsed"] += 1

        if self.parse_fallback == "first_line":
            head, _, tail = text.partition("\n")
            return self._clean_answer(head, enumeration), tail.strip()

        return self._clean_answer(text, enumeration), ""

    def _clean_answer(self, answer: str, enumeration: str | None) -> str:
        answer = answer.strip()
        if self.strip_enumeration:
            # The prompt contains the enumeration, and seq2seq models echo it.
            # Left in place it breaks both the length filter and exact match.
            answer = re.sub(r"\s*\(\s*[\d,\-\s]+\s*\)\s*$", "", answer)
        answer = answer.strip().strip('."\'`,;:')
        # Gold answers in this dataset are lowercase; normalize so the printed
        # prediction and the compared prediction are the same string.
        return answer.lower()

    # ----------------------------------------------------------------- decode
    def _sequence_scores(self, outputs, num_sequences: int) -> list[float | None]:
        """Length-normalized log-prob per returned sequence.

        Beam methods hand this back directly as `sequences_scores`. Sampling
        does not, so reconstruct it from transition scores. Either way the value
        must be length-normalized, otherwise it ranks by brevity and the
        "baseline" we compare against is not the baseline anyone would ship.
        """
        seq_scores = getattr(outputs, "sequences_scores", None)
        if seq_scores is not None:
            return [float(s) for s in seq_scores.detach().float().cpu()]

        try:
            transitions = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
        except Exception:  # noqa: BLE001 - scoring is a nicety, never fatal
            return [None] * num_sequences

        transitions = transitions.detach().float().cpu()
        finite = torch.isfinite(transitions)
        totals = torch.where(finite, transitions, torch.zeros_like(transitions)).sum(dim=-1)
        counts = finite.sum(dim=-1).clamp(min=1)
        return [float(v) for v in (totals / counts)]

    def _encode(self, records: list[ClueRecord]):
        """Tokenize a batch, keeping only the fields the model's forward accepts.

        Some tokenizers (anything BERT-lineage, including DebertaV2) return
        `token_type_ids`. T5 does not accept it, and generate() does not ignore
        unknown kwargs -- it raises "The following `model_kwargs` are not used by
        the model". Filtering against the real signature makes the generator
        tolerant of a tokenizer that emits more than its model consumes, instead
        of dying at the first batch.
        """
        encoded = self.tokenizer(
            [self.build_prompt(r) for r in records],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )

        import inspect

        accepted = set(inspect.signature(self.model.forward).parameters)
        dropped = [key for key in encoded if key not in accepted]
        for key in dropped:
            encoded.pop(key)
        if dropped and not self._warned_dropped_inputs:
            self._warned_dropped_inputs = True
            print(
                f"[generator] tokenizer emitted {dropped}, which "
                f"{type(self.model).__name__}.forward does not accept -- dropping. "
                "Usually harmless, but check the tokenizer matches the model."
            )

        return encoded.to(self.device)

    @torch.no_grad()
    def _run_passes(self, batch: list[ClueRecord]) -> list[list[tuple[str, float | None]]]:
        """Run every decode pass over one batch; return per-record (decode, score).

        Results are concatenated in pass order, so with `beam_sample_union` the
        beam decodes come first and the merged pool's rank 0 is the beam top-1 --
        the honest baseline.
        """
        inputs = self._encode(batch)
        collected: list[list[tuple[str, float | None]]] = [[] for _ in batch]

        for _label, kwargs in self._decode_passes():
            kwargs = self._apply_device_workarounds(kwargs)
            n = int(kwargs["num_return_sequences"])
            outputs = self.model.generate(**inputs, **kwargs)
            decoded = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            scores = self._sequence_scores(outputs, len(decoded))

            # generate() returns n sequences per input, flattened and ordered
            # best-first within each input's contiguous block.
            for i in range(len(batch)):
                collected[i].extend(zip(decoded[i * n : (i + 1) * n], scores[i * n : (i + 1) * n]))

        return collected

    @torch.no_grad()
    def generate(self, records: list[ClueRecord]) -> None:
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            for record, pairs in zip(batch, self._run_passes(batch)):
                decodes = [d for d, _ in pairs]
                scores = [s for _, s in pairs]
                record.candidates = self._to_candidates(decodes, scores, record)

    def _to_candidates(
        self, decodes: Iterable[str], scores: list[float | None], record: ClueRecord
    ) -> list[Candidate]:
        """Parse, then collapse duplicates while preserving generator rank.

        Deduping matters for more than tidiness: beam search routinely returns
        the same answer several times with different phrasing, and without a
        collapse the scorer burns forward passes on repeats and the reported
        "k candidates" overstates the real diversity.
        """
        seen: dict[str, Candidate] = {}
        ordered: list[Candidate] = []

        for raw, gen_score in zip(decodes, scores):
            answer, reason = self.parse_output(raw, record.enumeration)
            key = (normalize_answer(answer), " ".join(reason.lower().split()))

            if key in seen:
                # Keep the better-scoring instance, but never let a duplicate
                # improve its rank -- rank is the baseline's ordering.
                existing = seen[key]
                if gen_score is not None and (
                    existing.gen_score is None or gen_score > existing.gen_score
                ):
                    existing.gen_score = gen_score
                    existing.reason = reason or existing.reason
                continue

            cand = Candidate(
                answer=answer,
                reason=reason,
                raw=raw.strip(),
                gen_rank=len(ordered),
                gen_score=gen_score,
            )
            seen[key] = cand
            ordered.append(cand)

        return ordered

    @torch.no_grad()
    def probe(self, records: list[ClueRecord]) -> list[tuple[ClueRecord, list[str]]]:
        """Raw decodes with no parsing, for working out the output format.

        Use this the first time you point the pipeline at an unfamiliar
        checkpoint: it shows exactly what the model emits, which is what you
        need in order to write `parse.patterns`.
        """
        out = []
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            for record, pairs in zip(batch, self._run_passes(batch)):
                out.append((record, [d for d, _ in pairs]))
        return out


# --------------------------------------------------------------------------
# Stage 2 -- scoring
# --------------------------------------------------------------------------
class SequenceClassificationScorer:
    """Cross-encoder verifier: P(reasoning is correct | clue, answer, reason).

    Wraps the fine-tuned deberta-v3 classifier from
    `code/DeBERTa_small_Wordplay_Reasoning_Scorer_with_finetuning.ipynb`, using
    the same `CLUE:/ANSWER:/REASONING:` template it was trained on. That
    template is a config field, and it MUST match training -- a scorer fed a
    layout it never saw produces confident nonsense, not an error.
    """

    def __init__(self, cfg: dict, device: torch.device | None = None):
        self.cfg = cfg
        self.device = device or pick_device(cfg.get("device", "auto"))
        self.model_path = paths.resolve_model_path(cfg["model"])

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, **_dtype_kwargs(self.device, cfg.get("dtype", "auto"))
        )
        self.model.to(self.device)
        self.model.eval()

        self.template: str = cfg.get(
            "template", "CLUE: {clue}\nANSWER: {answer}\nREASONING: {reason}"
        )
        self.max_length: int = int(cfg.get("max_length", 256))
        self.batch_size: int = int(cfg.get("batch_size", 32))
        self.positive_index = self._resolve_positive_index(cfg.get("positive_label"))
        self._warn_if_head_untrained()

        self.description = f"{self.model_path} [positive logit {self.positive_index}]"

    def _warn_if_head_untrained(self) -> None:
        """Shout if this looks like a base model with a random classifier head.

        The single most expensive failure mode in this pipeline: pointing at
        `microsoft/deberta-v3-small` instead of the fine-tuned scorer loads fine,
        runs fine, and produces scores that are pure noise. Nothing errors, and
        the resulting "the reranker doesn't help" conclusion is about the config,
        not the method. A checkpoint fine-tuned by the notebook carries
        id2label={0: INCORRECT, 1: CORRECT}; the untouched base model carries the
        LABEL_n placeholders, which is what we detect.

        The verdict is also recorded on the instance so the run's metrics.json
        carries it -- a warning scrolls out of a Slurm log, a JSON field does not.
        """
        id2label = getattr(self.model.config, "id2label", None) or {}
        placeholder = all(
            str(name).upper().startswith("LABEL_") for name in id2label.values()
        )
        self.head_looks_untrained = bool(placeholder and not Path(self.model_path).is_dir())
        if self.head_looks_untrained:
            import warnings

            warnings.warn(
                f"\n{'!' * 78}\n"
                f"Scorer {self.model_path!r} has placeholder labels {sorted(id2label.values())},\n"
                "which means its classification head is RANDOMLY INITIALIZED and its scores\n"
                "are noise. This is almost certainly the fine-tuned checkpoint not being in\n"
                "place yet, not an intentional choice. Drop it in per models/README.md, or\n"
                "pass --set scorer.model=<path-or-hub-id>, before reporting any number from\n"
                f"this run.\n{'!' * 78}",
                RuntimeWarning,
                stacklevel=3,
            )

    def _resolve_positive_index(self, configured: str | int | None) -> int:
        """Find which logit means "correct".

        Reading it from the checkpoint's own label2id rather than hardcoding 1
        is what stops a silently inverted score when a future scorer is trained
        with the labels the other way round -- a bug that looks like "the
        reranker is slightly worse than baseline" rather than like a bug.
        """
        label2id = getattr(self.model.config, "label2id", None) or {}
        if isinstance(configured, int):
            return configured
        if isinstance(configured, str):
            if configured in label2id:
                return int(label2id[configured])
            raise ValueError(
                f"positive_label {configured!r} not in checkpoint labels {sorted(label2id)}"
            )
        for name in ("CORRECT", "correct", "LABEL_1", "entailment"):
            if name in label2id:
                return int(label2id[name])
        num_labels = int(getattr(self.model.config, "num_labels", 2))
        return 1 if num_labels > 1 else 0

    def build_input(self, record: ClueRecord, cand: Candidate) -> str:
        return self.template.format(
            clue=record.clue,
            answer=cand.answer,
            reason=cand.reason,
            enumeration=record.enumeration or "",
        )

    @torch.no_grad()
    def score(self, records: list[ClueRecord]) -> None:
        # Flatten across clues before batching. Scoring per-clue would leave
        # most batches at k rows (~12), which underuses the GPU by an order of
        # magnitude at 10k-clue scale.
        flat: list[tuple[Candidate, str]] = [
            (cand, self.build_input(record, cand))
            for record in records
            for cand in record.candidates
            if cand.alive
        ]
        if not flat:
            return

        for start in range(0, len(flat), self.batch_size):
            chunk = flat[start : start + self.batch_size]
            inputs = self.tokenizer(
                [text for _, text in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            logits = self.model(**inputs).logits.float()
            if logits.shape[-1] == 1:  # a regression-head scorer
                probs = torch.sigmoid(logits[:, 0])
            else:
                probs = torch.softmax(logits, dim=-1)[:, self.positive_index]

            for (cand, _), p in zip(chunk, probs.cpu().tolist()):
                cand.score = float(p)


# --------------------------------------------------------------------------
# Registry -- the seam a new model kind plugs into
# --------------------------------------------------------------------------
GENERATORS = {"seq2seq": Seq2SeqGenerator}
SCORERS = {"sequence_classification": SequenceClassificationScorer}


def build_generator(cfg: dict, device: torch.device | None = None) -> Generator:
    kind = cfg.get("kind", "seq2seq")
    if kind not in GENERATORS:
        raise ValueError(f"unknown generator kind {kind!r}; have {sorted(GENERATORS)}")
    return GENERATORS[kind](cfg, device)


def build_scorer(cfg: dict, device: torch.device | None = None) -> Scorer:
    kind = cfg.get("kind", "sequence_classification")
    if kind not in SCORERS:
        raise ValueError(f"unknown scorer kind {kind!r}; have {sorted(SCORERS)}")
    return SCORERS[kind](cfg, device)
