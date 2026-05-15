"""
src/training/trainer.py
-----------------------
Build SFTTrainer với các fix đã được áp dụng.

Các FIX so với notebook gốc:
1. Bỏ compute_metrics + preprocess_logits_for_metrics (token_accuracy không phù hợp)
2. Dùng eval_loss làm metric chính (prediction_loss_only=True)
3. peft_config KHÔNG truyền vào SFTTrainer (model đã là PEFT model)
4. train_on_responses_only dùng "\n" đúng chuẩn Qwen2.5 template
5. warmup_ratio thay cho warmup_steps cứng
6. num_train_epochs thay cho max_steps=200
"""
from __future__ import annotations

from transformers import EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from src.data.dataset import formatting_func
from src.utils.config import AppConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_training_args(cfg: AppConfig) -> TrainingArguments:
    tc = cfg.training
    return TrainingArguments(
        output_dir=tc.output_dir,
        # Batch & accumulation
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        # Epochs / steps
        # FIX: dùng num_train_epochs=1 thay vì max_steps=200
        num_train_epochs=tc.num_train_epochs,
        max_steps=tc.max_steps,           # -1 → bị override bởi num_train_epochs
        # Learning rate schedule
        warmup_ratio=tc.warmup_ratio,     # FIX: scale theo dataset, không hardcode
        learning_rate=tc.learning_rate,
        lr_scheduler_type=tc.lr_scheduler_type,
        # Optimizer & precision
        optim=tc.optim,
        weight_decay=tc.weight_decay,
        fp16=tc.fp16,
        # Logging
        logging_steps=tc.logging_steps,
        report_to=tc.report_to,
        # Eval & save
        eval_strategy=tc.eval_strategy,
        eval_steps=tc.eval_steps,
        save_strategy=tc.save_strategy,
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        # Best model selection
        load_best_model_at_end=tc.load_best_model_at_end,
        metric_for_best_model=tc.metric_for_best_model,   # FIX: "eval_loss"
        greater_is_better=tc.greater_is_better,           # FIX: False (loss nhỏ = tốt)
        # FIX: True → không generate predictions, chỉ tính loss → nhanh + đúng hơn
        prediction_loss_only=tc.prediction_loss_only,
    )


def build_trainer(
    cfg: AppConfig,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
) -> SFTTrainer:

    training_args = build_training_args(cfg)

    callbacks = []
    if cfg.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping.patience,
                early_stopping_threshold=cfg.early_stopping.threshold,
            )
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # FIX: KHÔNG truyền peft_config — model đã được wrap bởi get_peft_model
        formatting_func=formatting_func,
        max_seq_length=cfg.model.max_seq_length,
        packing=cfg.training.packing,
        args=training_args,
        callbacks=callbacks,
        # FIX: compute_metrics & preprocess_logits_for_metrics bị bỏ
        #      → dùng eval_loss làm metric chính
    )

    # FIX: thêm \n vào instruction_part và response_part
    # Qwen2.5 template tạo ra "<|im_start|>user\n" (có newline).
    # Thiếu \n → mask bị lệch → model train trên cả prompt
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    logger.info("SFTTrainer đã được khởi tạo.")
    return trainer
