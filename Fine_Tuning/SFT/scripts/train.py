"""
scripts/train.py
----------------
Entry point chạy toàn bộ pipeline fine-tuning SFT.

Cách dùng:
    python scripts/train.py --config configs/training_config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Thêm project root vào sys.path để import src.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import prepare_datasets
from src.models.model import load_model_for_training, save_lora
from src.training.trainer import build_trainer
from src.utils.config import AppConfig
from src.utils.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT Fine-tuning script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Đường dẫn đến file YAML config",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = AppConfig.from_yaml(args.config)

    logger = get_logger("train", log_file=cfg.paths.log_file)
    logger.info("=" * 60)
    logger.info("Bắt đầu SFT fine-tuning")
    logger.info(f"Config: {args.config}")
    logger.info(f"Model : {cfg.model.name}")
    logger.info(f"Data  : {cfg.data.dataset_name}")
    logger.info("=" * 60)

    # 1. Load model + LoRA
    model, tokenizer = load_model_for_training(cfg)

    # 2. Chuẩn bị dataset
    #    raw_test được giữ nguyên → dùng ở scripts/inference.py
    train_ds, eval_ds, _ = prepare_datasets(cfg.data, tokenizer)

    # 3. Build trainer
    trainer = build_trainer(cfg, model, tokenizer, train_ds, eval_ds)

    # 4. Train
    logger.info("Bắt đầu training...")
    trainer.train()
    logger.info("Training hoàn tất.")

    # 5. Lưu model
    save_lora(model, tokenizer, cfg.paths.final_model_dir)
    logger.info(f"Model đã lưu tại: {cfg.paths.final_model_dir}")


if __name__ == "__main__":
    main()
