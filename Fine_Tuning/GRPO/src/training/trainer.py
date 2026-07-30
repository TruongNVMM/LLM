"""
src/training/trainer.py
========================
Khởi tạo GRPOConfig và GRPOTrainer từ cấu hình.

Map từ Notebook:
  - Cell 30 : GRPOConfig(...)               → build_grpo_config()
  - Cell 32 : warnings.filterwarnings(...)  → _suppress_warnings()
  - Cell 33 : GRPOTrainer(...) + callback   → build_trainer()

vLLM Integration (TRL >= 0.15.0):
  Khi use_vllm=True, TRL tự động khởi động một vLLM subprocess chia sẻ
  cùng GPU. vLLM dùng PagedAttention để quản lý KV cache hiệu quả hơn,
  giúp giảm peak VRAM và tăng throughput generation.

  Các tham số vLLM trong build_grpo_config():
    use_vllm                  – bật/tắt vLLM backend
    vllm_gpu_memory_utilization – % VRAM dành cho vLLM KV cache
    vllm_max_model_len        – context window tối đa của vLLM server
    vllm_dtype                – dtype của vLLM ("float16", "bfloat16", "auto")
    vllm_max_num_seqs         – số sequences xử lý song song tối đa
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_grpo_config(
    output_dir: str = "outputs_grpo",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_generations: int = 4,
    max_completion_length: int = 512,
    max_prompt_length: int = 512,
    learning_rate: float = 5e-6,
    warmup_ratio: float = 0.05,
    lr_scheduler_type: str = "cosine",
    optim: str = "adamw_8bit",
    logging_steps: int = 10,
    save_steps: int = 20,
    save_total_limit: int = 2,
    report_to: str = "none",
    # ── vLLM parameters ───────────────────────────────────────────────────────
    use_vllm: bool = False,
    vllm_gpu_memory_utilization: float = 0.40,
    vllm_max_model_len: Optional[int] = None,
    vllm_dtype: str = "float16",
    vllm_max_num_seqs: int = 64,
) -> Any:
    """
    Khởi tạo GRPOConfig với các hyperparameters cho GRPO training.

    GRPO (Group Relative Policy Optimization) khác với PPO ở chỗ không
    cần mô hình Critic riêng biệt mà tính reward relative trong nhóm
    `num_generations` responses được sinh ra cho cùng 1 prompt.

    Parameters
    ----------
    output_dir : str
        Thư mục lưu checkpoints và model sau training.
    num_train_epochs : int
        Số epochs huấn luyện. Thường chỉ cần 1 epoch với GRPO.
    per_device_train_batch_size : int
        Batch size trên mỗi GPU. GRPO cần nhiều VRAM hơn SFT vì phải
        sinh `num_generations` responses cho mỗi sample.
    gradient_accumulation_steps : int
        Số steps tích lũy gradient trước khi update.
        Effective batch size = per_device_train_batch_size × gradient_accumulation_steps.
    num_generations : int
        Số responses sinh ra cho mỗi prompt để GRPO so sánh reward.
        Nhiều hơn → ổn định hơn nhưng tốn nhiều VRAM hơn. Thường dùng 4-8.
    max_completion_length : int
        Độ dài tối đa của response được sinh ra (tokens). Mặc định 512.
    max_prompt_length : int
        Độ dài tối đa của prompt đầu vào (tokens). Mặc định 512.
    learning_rate : float
        Learning rate. GRPO thường dùng LR nhỏ hơn SFT (5e-6 vs 2e-4).
    warmup_ratio : float
        Tỉ lệ steps dùng cho linear warmup. Mặc định 0.05 (5%).
    lr_scheduler_type : str
        Loại learning rate scheduler. "cosine" cho GRPO.
    optim : str
        Optimizer. "adamw_8bit" tiết kiệm VRAM hơn "adamw_torch".
    logging_steps : int
        Log metrics mỗi bao nhiêu steps. Mặc định 10.
    save_steps : int
        Lưu checkpoint mỗi bao nhiêu steps. Mặc định 20.
    save_total_limit : int
        Số lượng checkpoints tối đa giữ lại. Mặc định 2.
    report_to : str
        Platform để log metrics. "none", "wandb", "tensorboard".
    use_vllm : bool
        True = dùng vLLM subprocess để sinh completions thay vì eager generation.
        Yêu cầu TRL >= 0.15.0 và Linux/WSL2. Mặc định False.
        Lợi ích: PagedAttention → KV cache hiệu quả hơn → giảm peak VRAM,
        tăng throughput generation, cho phép num_generations lớn hơn.
    vllm_gpu_memory_utilization : float
        Tỉ lệ VRAM tối đa dành cho vLLM KV cache (0.0 – 1.0).
        Mặc định 0.40 (~4.4 GB trên 11 GB) — phù hợp RTX 2080 Ti + Qwen2.5 3B.
        Tăng nếu còn VRAM; giảm nếu vẫn OOM.
    vllm_max_model_len : int, optional
        Context window tối đa mà vLLM server xử lý.
        None = dùng max_seq_length của model. Đặt thấp hơn để tiết kiệm KV cache.
    vllm_dtype : str
        Kiểu dữ liệu cho vLLM. "float16" với RTX 2080 Ti (không hỗ trợ bfloat16 HW).
        Hoặc "auto" để vLLM tự detect.
    vllm_max_num_seqs : int
        Số sequences tối đa vLLM xử lý song song trong một batch.
        Giảm để tiết kiệm KV cache. Mặc định 64.

    Returns
    -------
    GRPOConfig
        Config object sẵn sàng truyền vào GRPOTrainer.

    Raises
    ------
    ImportError
        Nếu thư viện `trl` chưa được cài đặt.
    RuntimeError
        Nếu use_vllm=True nhưng TRL < 0.15.0.

    Example
    -------
    >>> # Training không dùng vLLM (mặc định)
    >>> config = build_grpo_config(output_dir="outputs/run_1", num_generations=4)

    >>> # Training với vLLM (tối ưu VRAM)
    >>> config = build_grpo_config(
    ...     output_dir="outputs/run_vllm",
    ...     num_generations=6,
    ...     use_vllm=True,
    ...     vllm_gpu_memory_utilization=0.40,
    ...     vllm_max_model_len=768,
    ...     vllm_dtype="float16",
    ... )
    """
    try:
        from trl import GRPOConfig  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Thư viện 'trl' chưa được cài đặt.\n"
            "Chạy: pip install 'trl>=0.15.0'"
        ) from exc

    # ── Kiểm tra phiên bản TRL nếu dùng vLLM ─────────────────────────────────
    if use_vllm:
        _check_trl_vllm_support()

    logger.info("Đang tạo GRPOConfig:")
    logger.info("  output_dir                  = %s", output_dir)
    logger.info("  num_generations             = %d", num_generations)
    logger.info("  per_device_train_batch_size = %d", per_device_train_batch_size)
    logger.info("  learning_rate               = %.2e", learning_rate)
    logger.info("  use_vllm                    = %s", use_vllm)

    if use_vllm:
        logger.info("  vllm_gpu_memory_utilization = %.2f", vllm_gpu_memory_utilization)
        logger.info("  vllm_max_model_len          = %s", vllm_max_model_len)
        logger.info("  vllm_dtype                  = %s", vllm_dtype)
        logger.info("  vllm_max_num_seqs           = %d", vllm_max_num_seqs)

    # ── Xây dựng kwargs base ───────────────────────────────────────────────────
    config_kwargs: Dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to=report_to,
    )

    # ── Thêm vLLM kwargs nếu được bật ─────────────────────────────────────────
    if use_vllm:
        config_kwargs["use_vllm"] = True
        config_kwargs["vllm_gpu_memory_utilization"] = vllm_gpu_memory_utilization
        if vllm_max_model_len is not None:
            config_kwargs["vllm_max_model_len"] = vllm_max_model_len
        config_kwargs["vllm_dtype"] = vllm_dtype
        config_kwargs["vllm_max_num_seqs"] = vllm_max_num_seqs

    config = GRPOConfig(**config_kwargs)
    return config


def build_trainer(
    model: Any,
    tokenizer: Any,
    config: Any,
    train_dataset: Any,
    reward_funcs: List[Any],
    callbacks: Optional[List[Any]] = None,
    suppress_warnings: bool = True,
) -> Any:
    """
    Khởi tạo GRPOTrainer với model, reward functions và callback.

    Parameters
    ----------
    model : FastLanguageModel
        Model đã được áp dụng LoRA, sẵn sàng để train.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer tương ứng.
    config : GRPOConfig
        Config từ `build_grpo_config()`.
    train_dataset : datasets.Dataset
        GRPO dataset với cột "prompt" từ `build_grpo_dataset()`.
    reward_funcs : List[callable]
        Danh sách reward functions. Mỗi hàm nhận `completions: List[str]`
        và trả về `List[float]`. GRPOTrainer sẽ log riêng từng hàm.
        Ví dụ: [compute_reward, reward_code_format, reward_syntax_valid]
    callbacks : List[TrainerCallback], optional
        Danh sách callbacks. Thường truyền [CodeQualityCallback(...)].
    suppress_warnings : bool
        True = ẩn FutureWarning từ transformers.modeling_attn_mask_utils.
        Mặc định True.

    Returns
    -------
    GRPOTrainer
        Trainer đã được khởi tạo, sẵn sàng để gọi `.train()`.

    Raises
    ------
    ImportError
        Nếu thư viện `trl` chưa được cài đặt.

    Example
    -------
    >>> trainer = build_trainer(
    ...     model=model,
    ...     tokenizer=tokenizer,
    ...     config=config,
    ...     train_dataset=grpo_dataset,
    ...     reward_funcs=[compute_reward, reward_code_format],
    ...     callbacks=[callback],
    ... )
    >>> trainer.train()
    """
    try:
        from trl import GRPOTrainer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Thư viện 'trl' chưa được cài đặt.\n"
            "Chạy: pip install 'trl>=0.15.0'"
        ) from exc

    if suppress_warnings:
        _suppress_warnings()

    callbacks = callbacks or []

    logger.info("Đang tạo GRPOTrainer:")
    logger.info("  reward_funcs = %s", [f.__name__ for f in reward_funcs])
    logger.info("  callbacks    = %s", [type(c).__name__ for c in callbacks])
    logger.info("  train_dataset size = %d", len(train_dataset))
    logger.info("  use_vllm     = %s", getattr(config, "use_vllm", False))

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_funcs,  # Mỗi hàm → 1 dòng log riêng trong console
        args=config,
        train_dataset=train_dataset,
        callbacks=callbacks,
    )

    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (private)
# ─────────────────────────────────────────────────────────────────────────────

def _check_trl_vllm_support() -> None:
    """
    Kiểm tra phiên bản TRL có hỗ trợ use_vllm hay không.

    TRL >= 0.15.0 mới có tham số use_vllm trong GRPOConfig.
    Phát hiện sớm để tránh lỗi khó debug khi khởi tạo GRPOConfig.

    Raises
    ------
    RuntimeError
        Nếu TRL < 0.15.0 hoặc GRPOConfig không có tham số use_vllm.
    """
    try:
        import trl  # noqa: PLC0415
        from trl import GRPOConfig  # noqa: PLC0415
        import inspect  # noqa: PLC0415

        trl_version = getattr(trl, "__version__", "unknown")
        grpo_params = inspect.signature(GRPOConfig).parameters

        if "use_vllm" not in grpo_params:
            raise RuntimeError(
                f"TRL version {trl_version} không hỗ trợ use_vllm trong GRPOConfig.\n"
                f"Vui lòng nâng cấp: pip install 'trl>=0.15.0'"
            )

        logger.info("TRL version %s hỗ trợ vLLM ✓", trl_version)

    except ImportError as exc:
        raise ImportError(
            "Thư viện 'trl' chưa được cài đặt.\n"
            "Chạy: pip install 'trl>=0.15.0'"
        ) from exc


def _suppress_warnings() -> None:
    """
    Ẩn các cảnh báo FutureWarning không quan trọng từ transformers.

    Cụ thể: AttentionMaskConverter từ modeling_attn_mask_utils,
    xuất hiện nhiều trong quá trình generate với các model cũ.
    """
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="transformers.modeling_attn_mask_utils",
    )
    logger.debug("Đã thiết lập bộ lọc FutureWarning từ transformers.")
