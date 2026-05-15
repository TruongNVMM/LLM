"""
src/utils/config.py
-------------------
Load YAML config và expose dưới dạng typed dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ─────────────────────────────────────────────
# Dataclasses cho từng section
# ─────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    max_seq_length: int = 2048
    dtype: Optional[str] = None
    load_in_4bit: bool = True


@dataclass
class LoraConfig:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 42
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class DataConfig:
    dataset_name: str
    test_size: float = 0.1
    seed: int = 42
    eval_sample_size: int = 200
    system_prompt: str = (
        "You are a helpful, respectful and honest code assistant. "
        "Always answer as helpfully as possible."
    )


@dataclass
class TrainingConfig:
    output_dir: str = "outputs"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_ratio: float = 0.05
    learning_rate: float = 2e-4
    fp16: bool = True
    optim: str = "adamw_8bit"
    weight_decay: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    prediction_loss_only: bool = True
    packing: bool = True
    report_to: str = "none"


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 3
    threshold: float = 0.001


@dataclass
class InferenceConfig:
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    num_test_samples: int = 3


@dataclass
class PathsConfig:
    final_model_dir: str = "final_lora_model"
    merged_model_dir: str = "final_merged_model"
    log_file: str = "logs/training.log"


@dataclass
class AppConfig:
    model: ModelConfig
    lora: LoraConfig
    data: DataConfig
    training: TrainingConfig
    early_stopping: EarlyStoppingConfig
    inference: InferenceConfig
    paths: PathsConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> "AppConfig":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config không tìm thấy: {config_path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**raw["model"]),
            lora=LoraConfig(**raw["lora"]),
            data=DataConfig(**raw["data"]),
            training=TrainingConfig(**raw["training"]),
            early_stopping=EarlyStoppingConfig(**raw.get("early_stopping", {})),
            inference=InferenceConfig(**raw.get("inference", {})),
            paths=PathsConfig(**raw.get("paths", {})),
        )
