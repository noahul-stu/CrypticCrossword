import os

# Must be set before transformers/datasets/huggingface_hub are imported --
# they read HF_HOME at import time to resolve cache paths. Redirects the
# ~3GB model cache off your (often quota-limited) $HOME onto project storage.
os.environ.setdefault("HF_HOME", "/home/morg/NLP_2526b/hullernoa/hf_cache")

import random
import tempfile
import urllib.request
import zipfile

import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

# ==========================================
# TOGGLE THIS TO FALSE FOR THE REAL RUN
# ==========================================
DEBUG_MODE = True
# ==========================================

print(f"Starting Script. DEBUG_MODE is {DEBUG_MODE}")

MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ------------------------------------------------------------------
# Dataset: download the paper's official answer-split directly
# ------------------------------------------------------------------
ZIP_URL = "https://github.com/aviaefrat/cryptonite/blob/main/data/cryptonite-official-split.zip?raw=true"
DATA_DIR = os.environ.get("CRYPTONITE_DATA_DIR", "/home/morg/NLP_2526b/hullernoa/cryptonite_data")
EXPECTED_FILES = ["cryptonite-train.jsonl", "cryptonite-val.jsonl", "cryptonite-test.jsonl"]


def data_is_complete(data_dir):
    return all(os.path.exists(os.path.join(data_dir, f)) for f in EXPECTED_FILES)


if not data_is_complete(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading cryptonite-official-split.zip ...")
    fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".zip.tmp")
    os.close(fd)
    try:
        urllib.request.urlretrieve(ZIP_URL, tmp_path)
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(DATA_DIR)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    assert data_is_complete(DATA_DIR), "Download/extraction finished but expected files are missing."

import json as _json
from datasets import Dataset, DatasetDict


def _load_jsonl_split(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _json.loads(line)
            records.append({
                "clue": obj["clue"],
                "answer": obj["answer"],
                "enumeration": obj["enumeration"],
                "quick": bool(obj.get("quick", False)),
            })
    return Dataset.from_list(records)


dataset = DatasetDict({
    "train": _load_jsonl_split(os.path.join(DATA_DIR, "cryptonite-train.jsonl")),
    "validation": _load_jsonl_split(os.path.join(DATA_DIR, "cryptonite-val.jsonl")),
    "test": _load_jsonl_split(os.path.join(DATA_DIR, "cryptonite-test.jsonl")),
})


def preprocess_function(examples):
    # examples["clue"] already contains the enumeration text, e.g.
    # "make progress socially in stated region (5)" -- do NOT re-append
    # examples["enumeration"], that would duplicate it.
    inputs = examples["clue"]
    targets = examples["answer"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=32, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if DEBUG_MODE:
    print("Truncating dataset for sanity check...")
    train_dataset = dataset["train"].select(range(50)).map(preprocess_function, batched=True)
    eval_dataset = dataset["validation"].select(range(50)).map(preprocess_function, batched=True)
else:
    train_dataset = dataset["train"].map(preprocess_function, batched=True)
    eval_dataset = dataset["validation"].map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    exact_matches = sum(
        1 for p, l in zip(decoded_preds, decoded_labels) if p.strip().lower() == l.strip().lower()
    )
    return {"exact_match_accuracy": exact_matches / len(decoded_preds)}


# ------------------------------------------------------------------
# Token-budget batch sampler, approximating the paper's "batch size
# of 7000 tokens"
# ------------------------------------------------------------------
class TokenBudgetBatchSampler(Sampler):
    def __init__(self, lengths, token_budget=7000, shuffle=True):
        self.lengths = lengths
        self.token_budget = token_budget
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)
        batch, running = [], 0
        for idx in indices:
            n = self.lengths[idx]
            if batch and running + n > self.token_budget:
                yield batch
                batch, running = [], 0
            batch.append(idx)
            running += n
        if batch:
            yield batch

    def __len__(self):
        avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        per_batch = max(int(self.token_budget / avg_len), 1)
        return max(len(self.lengths) // per_batch, 1)


class TokenBudgetSeq2SeqTrainer(Seq2SeqTrainer):
    def get_train_dataloader(self):
        lengths = [len(ex["input_ids"]) + len(ex["labels"]) for ex in self.train_dataset]
        sampler = TokenBudgetBatchSampler(lengths, token_budget=7000, shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
        )


output_directory = "/home/morg/NLP_2526b/hullernoa/output_model"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_directory,
    optim="adafactor",
    learning_rate=0.001,
    lr_scheduler_type="constant",

    max_steps=5 if DEBUG_MODE else -1,
    num_train_epochs=1 if DEBUG_MODE else 1000,
    eval_strategy="steps" if DEBUG_MODE else "epoch",
    eval_steps=5 if DEBUG_MODE else None,
    save_strategy="steps" if DEBUG_MODE else "epoch",
    save_total_limit=3,  # cap checkpoints on disk -- these add up fast on shared quota

    load_best_model_at_end=not DEBUG_MODE,
    metric_for_best_model="exact_match_accuracy",
    greater_is_better=True,

    predict_with_generate=True,
    generation_num_beams=5,
    generation_max_length=32,
    report_to="none" if DEBUG_MODE else "wandb",
    bf16=torch.cuda.is_bf16_supported() if not DEBUG_MODE else False,  # safer than fp16 for T5
)

trainer_cls = Seq2SeqTrainer if DEBUG_MODE else TokenBudgetSeq2SeqTrainer
trainer = trainer_cls(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=None if DEBUG_MODE else [EarlyStoppingCallback(early_stopping_patience=10)],
)

last_checkpoint = get_last_checkpoint(training_args.output_dir)
if last_checkpoint is not None and not DEBUG_MODE:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("Starting training from scratch...")
    trainer.train()

print("Script completed!")