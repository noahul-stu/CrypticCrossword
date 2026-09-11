"""Stage 1 of STaR: sample N candidate traces per clue.

Loads a T5 checkpoint (the warm-started one on iteration 0, the previous
iteration's fine-tune afterwards), samples `generate.n_samples` traces per clue,
labels each with `verify.py`, and writes one jsonl row per clue holding every
candidate with its verdict.

That file is used twice: the accepted traces become fine-tuning data, and the
full candidate list with labels is exactly what the DeBERTa discriminator needs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

from .config import Config, add_config_args, config_from_args
from .data.align import clue_to_row, row_to_clue
from .data.cryptonite import Clue
from .data.formatting import render_source
from .io_utils import pick_device, read_jsonl, seed_everything, setup_logging, write_json, write_jsonl
from .verify import Reason, verify_trace

log = logging.getLogger(__name__)


class TraceGenerator:
    """Thin wrapper over a HF seq2seq model, shared by generate/rationalize/eval."""

    def __init__(self, model_path: str, cfg: Config, device: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.cfg = cfg
        self.device = device or pick_device(cfg.device)
        self.torch = torch
        log.info("loading %s onto %s", model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.bfloat16 if (cfg.train.bf16 and self.device == "cuda") else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype)
        self.model.to(self.device).eval()

    def _decode_batch(self, sources: Sequence[str], **gen_kwargs) -> list[list[str]]:
        enc = self.tokenizer(
            list(sources),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.format.max_source_len,
        ).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        n_per = gen_kwargs.get("num_return_sequences", 1)
        texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return [texts[i * n_per : (i + 1) * n_per] for i in range(len(sources))]

    def sample(self, sources: Sequence[str], n: int) -> list[list[str]]:
        return self._decode_batch(
            sources,
            do_sample=True,
            num_return_sequences=n,
            temperature=self.cfg.generate.temperature,
            top_p=self.cfg.generate.top_p,
            top_k=self.cfg.generate.top_k or 0,
            max_new_tokens=self.cfg.generate.max_new_tokens,
        )

    def greedy(self, sources: Sequence[str], num_beams: int = 1) -> list[str]:
        batches = self._decode_batch(
            sources,
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=1,
            max_new_tokens=self.cfg.eval.max_new_tokens,
        )
        return [b[0] for b in batches]


def _chunks(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def generate_for_clues(
    gen: TraceGenerator,
    clues: Sequence[Clue],
    cfg: Config,
    hints: bool = False,
) -> list[dict]:
    """Sample and label. `hints=True` runs the rationalisation prompt."""
    rows: list[dict] = []
    total = len(clues)
    for bi, batch in enumerate(_chunks(list(clues), cfg.generate.batch_size)):
        sources = [
            render_source(c, cfg.format, hint=c.answer if hints else None) for c in batch
        ]
        samples = gen.sample(sources, cfg.generate.n_samples)
        for clue, source, traces in zip(batch, sources, samples):
            if cfg.generate.dedup:
                traces = list(dict.fromkeys(traces))
            candidates = []
            for t in traces:
                v = verify_trace(
                    clue,
                    t,
                    require_reasoning=cfg.verify.require_reasoning,
                    min_wordplay_chars=cfg.verify.min_wordplay_chars,
                    accept_alt_answers=cfg.verify.accept_alt_answers,
                )
                candidates.append(
                    {
                        "trace": v.trace,
                        "label": v.label,
                        "reason": v.reason,
                        "predicted": v.predicted,
                        "wordplay": v.wordplay,
                        "definition": v.definition,
                    }
                )
            rows.append(
                {
                    **clue_to_row(clue),
                    "source": source,
                    "hinted": hints,
                    "candidates": candidates,
                    "n_correct": sum(c["label"] for c in candidates),
                }
            )
        if bi % 20 == 0:
            done = min((bi + 1) * cfg.generate.batch_size, total)
            solved = sum(1 for r in rows if r["n_correct"])
            log.info("generated %d/%d clues | %d solved so far", done, total, solved)
    return rows


def summarise(rows: Sequence[dict]) -> dict:
    n = len(rows) or 1
    reasons: dict[str, int] = {}
    n_cand = 0
    for r in rows:
        for c in r["candidates"]:
            reasons[c["reason"]] = reasons.get(c["reason"], 0) + 1
            n_cand += 1
    solved = sum(1 for r in rows if r["n_correct"])
    return {
        "clues": len(rows),
        "candidates": n_cand,
        "clues_with_at_least_one_correct": solved,
        "pass_at_n": solved / n,
        "accepted_traces": reasons.get(Reason.OK, 0),
        "reject_reasons": reasons,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--model", required=True, help="checkpoint dir or HF model id")
    parser.add_argument("--clues", required=True, help="jsonl of clues to generate for")
    parser.add_argument("--out", required=True, help="output jsonl of candidates")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hints", action="store_true", help="rationalisation prompt")
    args = parser.parse_args(argv)

    cfg: Config = config_from_args(args)
    setup_logging(cfg.log_level)
    seed_everything(cfg.star.seed)

    clues = [row_to_clue(r) for r in read_jsonl(args.clues, limit=args.limit)]
    gen = TraceGenerator(args.model, cfg)
    rows = generate_for_clues(gen, clues, cfg, hints=args.hints)
    write_jsonl(args.out, rows)
    stats = summarise(rows)
    write_json(Path(args.out).with_suffix(".stats.json"), stats)
    log.info("wrote %d clues -> %s | pass@%d = %.3f", len(rows), args.out,
             cfg.generate.n_samples, stats["pass_at_n"])


if __name__ == "__main__":
    main()
