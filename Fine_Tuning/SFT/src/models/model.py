"""
src/models/model.py
-------------------
Load base model và áp dụng LoRA adapter qua Unsloth.

FIX so với notebook gốc:
- peft_config KHÔNG truyền vào SFTTrainer (model đã là PEFT model sau get_peft_model)
- Expose riêng hàm load_for_inference để dùng ở script inference
"""
from __future__ import annotations

from typing import Tuple

from unsloth import FastLanguageModel

from src.utils.config import AppConfig, LoraConfig, ModelConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_base_model(cfg: ModelConfig) -> Tuple:
    """Load pretrained model + tokenizer."""
    logger.info(f"Đang load model: {cfg.name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.name,
        max_seq_length=cfg.max_seq_length,
        dtype=cfg.dtype,
        load_in_4bit=cfg.load_in_4bit,
    )
    logger.info("Load model thành công.")
    return model, tokenizer


def apply_lora(model, cfg: LoraConfig):
    """Wrap model với LoRA adapter."""
    logger.info(f"Áp dụng LoRA: r={cfg.r}, alpha={cfg.lora_alpha}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.r,
        target_modules=cfg.target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.random_state,
    )
    logger.info("LoRA adapter đã được áp dụng.")
    return model


def load_model_for_training(cfg: AppConfig) -> Tuple:
    """
    Trả về (model_peft, tokenizer) sẵn sàng để train.
    KHÔNG trả về peft_config vì không cần truyền vào SFTTrainer.
    """
    model, tokenizer = load_base_model(cfg.model)
    model = apply_lora(model, cfg.lora)
    return model, tokenizer


def load_model_for_inference(model_dir: str, cfg: ModelConfig) -> Tuple:
    """
    Load LoRA model đã lưu và switch sang inference mode.
    """
    logger.info(f"Đang load model từ: {model_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=cfg.max_seq_length,
        dtype=cfg.dtype,
        load_in_4bit=cfg.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    logger.info("Model sẵn sàng để inference.")
    return model, tokenizer


def save_lora(model, tokenizer, save_dir: str) -> None:
    """Lưu LoRA adapter (không merge vào base weights)."""
    logger.info(f"Đang lưu LoRA adapter → {save_dir}")
    model.save_pretrained_merged(save_dir, tokenizer, save_method="lora")
    logger.info("Lưu thành công.")


def save_merged(model, tokenizer, save_dir: str) -> None:
    """Merge LoRA vào base model và lưu full model (dùng để deploy)."""
    logger.info(f"Đang merge + lưu full model → {save_dir}")
    model.save_pretrained_merged(save_dir, tokenizer, save_method="merged_16bit")
    logger.info("Merge và lưu thành công.")
