"""
scripts/train.py
================
Entry point chính để chạy toàn bộ GRPO training pipeline.

Cách sử dụng:
  python scripts/train.py
  python scripts/train.py --config configs/grpo_config.yaml
  python scripts/train.py --config configs/grpo_config.yaml --sft_model_dir /path/to/model

Pipeline:
  1. Parse arguments và load config YAML
  2. Load SFT model đã fine-tune + áp dụng LoRA
  3. Load dataset Evol-Instruct-Code-80k-v1 + format cho GRPO
  4. Xây dựng reward functions
  5. Xây dựng validation callback + early stopping
  6. Khởi tạo GRPOConfig và GRPOTrainer
  7. Chạy trainer.train()
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Setup path — đảm bảo import src.* hoạt động khi chạy scripts/train.py
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.callbacks.early_stopping import CodeQualityCallback
from src.data.dataset import build_grpo_dataset, load_and_split_dataset
from src.model.loader import apply_lora, load_model
from src.rewards.code_rewards import (
    reward_code_executable,
    reward_code_format,
    reward_no_placeholder,
    reward_syntax_valid,
)
from src.rewards.combined import compute_reward
from src.training.trainer import build_grpo_config, build_trainer

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("grpo_training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    """Load YAML config file và trả về dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file config: {config_path}\n"
            f"Hãy đảm bảo file configs/grpo_config.yaml tồn tại."
        )
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Đã load config từ: %s", config_path)
    return cfg


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GRPO Fine-Tuning — Evol-Instruct-Code-80k-v1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_ROOT / "configs" / "grpo_config.yaml"),
        help="Đường dẫn tới file config YAML. "
             "Dùng configs/grpo_config_vllm.yaml để bật tối ưu vLLM.",
    )
    parser.add_argument(
        "--sft_model_dir",
        type=str,
        default=None,
        help="Override SFT model dir từ config. "
             "Ví dụ: /content/drive/.../model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output dir từ config.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=None,
        help="Override số epochs từ config.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        choices=["none", "wandb", "tensorboard"],
        help="Platform để log metrics.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        default=None,
        help="Override use_vllm=True từ CLI (không cần sửa config YAML).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Chạy toàn bộ GRPO training pipeline."""
    args = _parse_args()

    # ── 1. Load config ────────────────────────────────────────────────────
    cfg = _load_config(args.config)

    model_cfg    = cfg["model"]
    lora_cfg     = cfg["lora"]
    dataset_cfg  = cfg["dataset"]
    training_cfg = cfg["training"]
    reward_cfg   = cfg["rewards"]
    callback_cfg = cfg["callback"]

    # Override config với CLI arguments (nếu có)
    if args.sft_model_dir:
        model_cfg["sft_model_dir"] = args.sft_model_dir
    if args.output_dir:
        training_cfg["output_dir"] = args.output_dir
    if args.num_train_epochs:
        training_cfg["num_train_epochs"] = args.num_train_epochs
    if args.report_to:
        training_cfg["report_to"] = args.report_to
    # --use_vllm CLI flag override (bật vLLM mà không cần sửa YAML)
    if getattr(args, "use_vllm", None):
        training_cfg["use_vllm"] = True

    # ── Đọc vLLM settings (có default an toàn nếu key không tồn tại) ──────
    use_vllm                   = training_cfg.get("use_vllm", False)
    vllm_gpu_memory_utilization = training_cfg.get("vllm_gpu_memory_utilization", 0.40)
    vllm_max_model_len          = training_cfg.get("vllm_max_model_len", None)
    vllm_dtype                  = training_cfg.get("vllm_dtype", "float16")
    vllm_max_num_seqs           = training_cfg.get("vllm_max_num_seqs", 64)

    logger.info("=" * 60)
    logger.info("GRPO Fine-Tuning Pipeline bắt đầu")
    logger.info("SFT Model : %s", model_cfg["sft_model_dir"])
    logger.info("Output dir: %s", training_cfg["output_dir"])
    if use_vllm:
        logger.info("Generation: vLLM (gpu_memory_utilization=%.2f, max_len=%s, dtype=%s)",
                    vllm_gpu_memory_utilization, vllm_max_model_len, vllm_dtype)
    else:
        logger.info("Generation: Eager (không dùng vLLM)")
    logger.info("=" * 60)

    # ── 2. Load SFT model đã fine-tune + áp dụng LoRA ────────────────────
    logger.info("[Bước 2/7] Load SFT model và áp dụng LoRA...")
    model, tokenizer = load_model(
        sft_model_dir=model_cfg["sft_model_dir"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg.get("dtype"),          # None = auto-detect
        load_in_4bit=model_cfg["load_in_4bit"],
    )
    model = apply_lora(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
        target_modules=lora_cfg["target_modules"],
    )

    # ── 3. Load dataset + format cho GRPO ────────────────────────────────
    logger.info("[Bước 3/7] Load và format dataset...")
    dataset_train, dataset_test = load_and_split_dataset(
        dataset_name=dataset_cfg["name"],
        split=dataset_cfg["split"],
        test_size=dataset_cfg["test_size"],
        seed=dataset_cfg["seed"],
    )
    grpo_dataset = build_grpo_dataset(
        dataset_train=dataset_train,
        tokenizer=tokenizer,
        system_prompt=dataset_cfg["system_prompt"],
    )

    # ── 4. Reward functions ───────────────────────────────────────────────
    logger.info("[Bước 4/7] Chuẩn bị reward functions...")
    reward_weights    = reward_cfg["weights"]
    execution_timeout = reward_cfg.get("execution_timeout", 5)

    # Bind weights và timeout vào compute_reward bằng partial
    from functools import partial
    _compute_reward = partial(
        compute_reward,
        weights=reward_weights,
        execution_timeout=execution_timeout,
    )
    _compute_reward.__name__ = "compute_reward"  # Để GRPOTrainer log đúng tên

    reward_funcs = [
        _compute_reward,
        reward_code_format,
        reward_syntax_valid,
        reward_code_executable,
        reward_no_placeholder,
    ]
    logger.info("Reward functions: %s", [f.__name__ for f in reward_funcs])

    # ── 5. Validation callback + early stopping ───────────────────────────
    logger.info("[Bước 5/7] Khởi tạo CodeQualityCallback...")
    val_ds = grpo_dataset.shuffle(seed=42).select(
        range(min(100, len(grpo_dataset)))
    )
    log_file_value = callback_cfg.get("log_file", "logs/eval_metrics.log")
    log_file_path = Path(log_file_value)
    if not log_file_path.is_absolute():
        log_file_path = (_ROOT / log_file_path).resolve()
    log_file = str(log_file_path)
    callback = CodeQualityCallback(
        val_dataset=val_ds,
        tokenizer=tokenizer,
        num_samples=callback_cfg["num_samples"],
        eval_steps=callback_cfg.get("eval_steps", 4),
        patience=callback_cfg["patience"],
        min_delta=callback_cfg["min_delta"],
        monitor=callback_cfg["monitor"],
        save_best=callback_cfg.get("save_best", True),
        best_model_dir=callback_cfg.get("best_model_dir", "best_grpo_checkpoint"),
        execution_timeout=execution_timeout,
        log_file=log_file,
    )

    # ── 6. GRPOConfig + GRPOTrainer ───────────────────────────────────────
    logger.info("[Bước 6/7] Khởi tạo GRPOConfig và GRPOTrainer...")
    config = build_grpo_config(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        num_generations=training_cfg["num_generations"],
        max_completion_length=training_cfg["max_completion_length"],
        max_prompt_length=training_cfg["max_prompt_length"],
        learning_rate=training_cfg["learning_rate"],
        warmup_ratio=training_cfg["warmup_ratio"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        optim=training_cfg["optim"],
        logging_steps=training_cfg["logging_steps"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],
        report_to=training_cfg["report_to"],
        # ── vLLM settings ────────────────────────────────────────────────
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_model_len=vllm_max_model_len,
        vllm_dtype=vllm_dtype,
        vllm_max_num_seqs=vllm_max_num_seqs,
    )
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=grpo_dataset,
        reward_funcs=reward_funcs,
        callbacks=[callback],
    )

    # ── 7. Training ───────────────────────────────────────────────────────
    logger.info("[Bước 7/7] Bắt đầu GRPO training...")
    logger.info("=" * 60)
    trainer.train()
    logger.info("=" * 60)
    logger.info("Training hoàn tất!")
    logger.info(
        "Best model đã được lưu tại: %s",
        callback_cfg.get("best_model_dir", "best_grpo_checkpoint"),
    )


if __name__ == "__main__":
    main()
