"""
scripts/export_model.py
========================
Load best GRPO checkpoint và merge LoRA weights thành full model float16.

Map từ Notebook:
  - Cell 35 : FastLanguageModel.from_pretrained() + save_pretrained_merged()

Cách sử dụng:
  python scripts/export_model.py
  python scripts/export_model.py --best_model_dir best_grpo_checkpoint
  python scripts/export_model.py --output_dir exported_model --save_method merged_16bit

Khi nào dùng:
  - Sau khi training hoàn tất (hoặc dừng bởi early stopping)
  - Để tạo model hoàn chỉnh (không còn LoRA adapter riêng) để:
    * Deploy lên HuggingFace Hub
    * Dùng với vLLM / LlamaCpp / Ollama
    * Chia sẻ hoặc backup
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Setup path
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export best GRPO checkpoint — merge LoRA sang float16.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_ROOT / "configs" / "grpo_config.yaml"),
        help="Đường dẫn tới file config YAML.",
    )
    parser.add_argument(
        "--best_model_dir",
        type=str,
        default=None,
        help="Override best_model_dir từ config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Thư mục lưu model sau khi merge. "
             "Mặc định: lưu tại thư mục hiện tại.",
    )
    parser.add_argument(
        "--save_method",
        type=str,
        default=None,
        choices=["merged_16bit", "merged_4bit", "lora"],
        help="Phương thức lưu model.\n"
             "  merged_16bit : Merge LoRA → float16 full model (recommended)\n"
             "  merged_4bit  : Merge LoRA → 4-bit quantized\n"
             "  lora         : Chỉ lưu LoRA adapter (không merge)\n",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Override max_seq_length khi load checkpoint.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load best checkpoint và export thành full merged model."""
    args = _parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        export_cfg = cfg.get("export", {})
        model_cfg  = cfg.get("model", {})
    else:
        logger.warning("Không tìm thấy config, dùng giá trị mặc định.")
        export_cfg = {}
        model_cfg  = {}

    # Ưu tiên CLI args > config
    best_model_dir  = args.best_model_dir  or export_cfg.get("best_model_dir", "best_grpo_checkpoint")
    output_dir      = args.output_dir      or export_cfg.get("output_dir", "")
    save_method     = args.save_method     or export_cfg.get("save_method", "merged_16bit")
    max_seq_length  = args.max_seq_length  or model_cfg.get("max_seq_length", 2048)

    logger.info("=" * 60)
    logger.info("Export Model Pipeline")
    logger.info("  best_model_dir : %s", best_model_dir)
    logger.info("  output_dir     : %s", output_dir if output_dir else "(thư mục hiện tại)")
    logger.info("  save_method    : %s", save_method)
    logger.info("=" * 60)

    # ── Kiểm tra thư mục best checkpoint tồn tại ─────────────────────────
    if not Path(best_model_dir).exists():
        logger.error(
            "Không tìm thấy best checkpoint tại: %s\n"
            "Hãy chạy scripts/train.py trước.",
            best_model_dir,
        )
        sys.exit(1)

    # ── Import Unsloth ────────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Thư viện 'unsloth' chưa được cài đặt.\n"
            "Chạy: pip install unsloth"
        ) from exc

    # ── Load best checkpoint ──────────────────────────────────────────────
    logger.info("Đang load best checkpoint từ: %s", best_model_dir)
    best_model, best_tokenizer = FastLanguageModel.from_pretrained(
        model_name=best_model_dir,
        max_seq_length=max_seq_length,
        dtype=None,         # auto-detect
        load_in_4bit=True,  # Load lại ở 4-bit để tiết kiệm RAM trong quá trình merge
    )
    logger.info("Best checkpoint đã được load thành công.")

    # ── Merge LoRA và lưu full model ─────────────────────────────────────
    logger.info("Đang merge LoRA weights và lưu model...")
    logger.info("  save_method : %s", save_method)

    best_model.save_pretrained_merged(
        output_dir,             # Thư mục đích (rỗng = thư mục hiện tại)
        best_tokenizer,
        save_method=save_method,
    )

    final_dir = output_dir if output_dir else Path.cwd()
    logger.info("=" * 60)
    logger.info("✅ Export hoàn tất!")
    logger.info("Final model đã được lưu tại: %s", final_dir)
    logger.info("=" * 60)
    print(f"\n✅ Đã lưu final model (best checkpoint) → {final_dir}")


if __name__ == "__main__":
    main()
