"""
src/model/loader.py
===================
Chịu trách nhiệm nạp SFT model đã fine-tune và áp dụng LoRA adapter.

Map từ Notebook:
  - Cell 5  : FastLanguageModel.from_pretrained()  → load_model()
  - Cell 6  : device check                          → load_model() (log)
  - Cell 8  : FastLanguageModel.get_peft_model()   → apply_lora()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    sft_model_dir: str,
    max_seq_length: int = 2048,
    dtype: Optional[Any] = None,
    load_in_4bit: bool = True,
) -> Tuple[Any, Any]:
    """
    Nạp SFT model đã được fine-tune trước đó bằng Unsloth FastLanguageModel.

    Model được load theo chế độ 4-bit (QLoRA) để tiết kiệm VRAM, phù hợp
    với GPU T4 trên Google Colab.

    Parameters
    ----------
    sft_model_dir : str
        Đường dẫn đến thư mục chứa SFT model đã fine-tune.
        Có thể là:
          - Đường dẫn local (e.g. "/content/drive/.../model")
          - HuggingFace Hub ID (e.g. "meta-llama/Meta-Llama-3-8B-Instruct")
    max_seq_length : int
        Độ dài sequence tối đa. Mặc định 2048.
    dtype : optional
        Kiểu dữ liệu tensor. None = auto-detect:
          - bfloat16 trên Ampere GPU (A100, A10G, ...)
          - float16  trên Turing GPU (T4, V100, ...)
    load_in_4bit : bool
        True = tải weights ở định dạng 4-bit (NF4) để tiết kiệm VRAM.

    Returns
    -------
    (model, tokenizer) : tuple
        model     – FastLanguageModel đã load, chạy ở chế độ inference.
        tokenizer – Tokenizer tương ứng.

    Raises
    ------
    ImportError
        Nếu thư viện `unsloth` chưa được cài đặt.
    ValueError
        Nếu `sft_model_dir` không tồn tại hoặc không hợp lệ.

    Example
    -------
    >>> model, tokenizer = load_model(
    ...     sft_model_dir="/content/drive/.../model",
    ...     max_seq_length=2048,
    ...     load_in_4bit=True,
    ... )
    >>> print(f"Device: {next(model.parameters()).device}")
    """
    try:
        from unsloth import FastLanguageModel  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Thư viện 'unsloth' chưa được cài đặt.\n"
            "Chạy: pip install unsloth"
        ) from exc

    logger.info("Đang tải SFT model từ: %s", sft_model_dir)
    logger.info("  max_seq_length = %d", max_seq_length)
    logger.info("  load_in_4bit   = %s", load_in_4bit)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_model_dir,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Kiểm tra và log device
    try:
        import torch  # noqa: PLC0415
        device = next(model.parameters()).device
        logger.info("Model đã được tải lên: %s", device)
        print(f"[Model] Device: {device} | dtype: {next(model.parameters()).dtype}")
    except StopIteration:
        logger.warning("Không thể xác định device của model.")

    return model, tokenizer


def apply_lora(
    model: Any,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    bias: str = "none",
    use_gradient_checkpointing: str = "unsloth",
    random_state: int = 42,
    target_modules: Optional[list] = None,
) -> Any:
    """
    Áp dụng LoRA adapter lên model đã load để chuẩn bị fine-tune với GRPO.

    Sử dụng PEFT (Parameter-Efficient Fine-Tuning) thông qua Unsloth để
    chỉ cập nhật một phần nhỏ trọng số, giúp tiết kiệm bộ nhớ đáng kể.

    Parameters
    ----------
    model : FastLanguageModel
        Model đã được load bởi `load_model()`.
    r : int
        LoRA rank — số chiều của ma trận low-rank.
        r càng lớn → nhiều tham số trainable hơn, nhưng cần nhiều VRAM hơn.
        Giá trị thông thường: 8, 16, 32, 64.
    lora_alpha : int
        Hệ số scaling cho LoRA weights. Thường đặt = 2×r hoặc bằng r.
    lora_dropout : float
        Dropout rate cho LoRA layers. GRPO thường dùng 0.0.
    bias : str
        Cách xử lý bias. "none" = không train bias (tiết kiệm bộ nhớ).
    use_gradient_checkpointing : str
        "unsloth" = dùng gradient checkpointing tối ưu của Unsloth,
        cho phép train batch lớn hơn với ít VRAM hơn.
    random_state : int
        Seed để đảm bảo tái lặp kết quả.
    target_modules : list, optional
        Danh sách các module được áp dụng LoRA.
        Mặc định: tất cả projection layers của attention + MLP.

    Returns
    -------
    model : FastLanguageModel
        Model đã được gắn LoRA adapter, sẵn sàng để train.

    Example
    -------
    >>> model = apply_lora(model, r=16, lora_alpha=32)
    """
    try:
        from unsloth import FastLanguageModel  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Thư viện 'unsloth' chưa được cài đặt.\n"
            "Chạy: pip install unsloth"
        ) from exc

    if target_modules is None:
        target_modules = [
            "q_proj",    # Query projection
            "k_proj",    # Key projection
            "v_proj",    # Value projection
            "o_proj",    # Output projection
            "gate_proj", # MLP gate
            "up_proj",   # MLP up
            "down_proj", # MLP down
        ]

    logger.info("Áp dụng LoRA adapter:")
    logger.info("  r              = %d", r)
    logger.info("  lora_alpha     = %d", lora_alpha)
    logger.info("  lora_dropout   = %.2f", lora_dropout)
    logger.info("  target_modules = %s", target_modules)

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
    )

    # Đếm và log số trainable parameters
    _log_trainable_params(model)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (private)
# ─────────────────────────────────────────────────────────────────────────────

def _log_trainable_params(model: Any) -> None:
    """In ra số lượng parameters trainable so với tổng số parameters."""
    try:
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pct = 100 * trainable_params / total_params if total_params > 0 else 0.0

        logger.info(
            "Trainable params: %s / %s (%.2f%%)",
            f"{trainable_params:,}",
            f"{total_params:,}",
            pct,
        )
        print(
            f"[LoRA] Trainable params: {trainable_params:,} / {total_params:,} "
            f"({pct:.2f}%)"
        )
    except Exception:  # noqa: BLE001
        logger.warning("Không thể đếm số lượng parameters.")
