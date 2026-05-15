"""
src/data/dataset.py
-------------------
Load, split và format dataset cho SFT.

Các FIX so với notebook gốc:
1. raw_test được giữ nguyên (không map) → dùng được ở inference (tránh KeyError)
2. format_qwen_template thêm system_prompt nhất quán vào mọi split
3. apply_chat_template tokenize=False → SFTTrainer tự tokenize
"""
from __future__ import annotations

from typing import Tuple

from datasets import Dataset, DatasetDict, load_dataset

from src.utils.config import DataConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Build functions
# ─────────────────────────────────────────────

def _build_format_fn(system_prompt: str):
    """(instruction, output) → messages list với system prompt."""
    def _fn(examples: dict) -> dict:
        formatted = []
        for instr, out in zip(examples["instruction"], examples["output"]):
            formatted.append([
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": instr},
                {"role": "assistant", "content": out},
            ])
        return {"messages": formatted}
    return _fn


def _build_template_fn(tokenizer):
    """messages list → text string theo Qwen2.5 chat template."""
    def _fn(examples: dict) -> dict:
        texts = [
            tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False
            )
            for msg in examples["messages"]
        ]
        return {"text": texts}
    return _fn


def _pipeline(ds: Dataset, format_fn, template_fn, desc: str) -> Dataset:
    ds = ds.map(format_fn,   batched=True, remove_columns=["instruction", "output"])
    ds = ds.map(template_fn, batched=True)
    logger.info(f"{desc}: {len(ds):,} mẫu đã được format.")
    return ds


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def prepare_datasets(
    cfg: DataConfig,
    tokenizer,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Trả về (train_dataset, eval_dataset, raw_test_dataset).

    - train_dataset  : đã format, dùng để train.
    - eval_dataset   : subset nhỏ đã format, dùng để eval trong training loop.
    - raw_test_dataset: CHƯA format, giữ nguyên cột instruction/output.
                        Dùng ở inference để lấy câu hỏi gốc → tránh KeyError.
    """
    logger.info(f"Đang load dataset: {cfg.dataset_name}")
    full = load_dataset(cfg.dataset_name, split="train")
    split: DatasetDict = full.train_test_split(test_size=cfg.test_size, seed=cfg.seed)

    raw_train = split["train"]
    raw_test  = split["test"]   # ← Không map ở đây — dùng nguyên bản cho inference

    logger.info(f"Train: {len(raw_train):,} | Test: {len(raw_test):,}")

    format_fn   = _build_format_fn(cfg.system_prompt)
    template_fn = _build_template_fn(tokenizer)

    train_ds = _pipeline(raw_train, format_fn, template_fn, "Train")

    eval_raw = raw_test.shuffle(seed=cfg.seed).select(range(cfg.eval_sample_size))
    eval_ds  = _pipeline(eval_raw, format_fn, template_fn, "Eval")

    return train_ds, eval_ds, raw_test


def formatting_func(batch: dict) -> list:
    """Hàm truyền vào SFTTrainer.formatting_func."""
    return batch["text"]
