"""
src/callbacks/metrics_logger.py

Custom Trainer Callback để lắng nghe event `on_log` từ GRPOTrainer và ghi toàn bộ
các metric huấn luyện (loss, grad_norm, learning_rate, rewards, completions, kl, epoch, ...)
ra file log cứ sau mỗi N steps (`logging_steps`).
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class TrainingMetricsLoggerCallback(TrainerCallback):
    """
    Callback tự động bắt event `on_log` của GRPOTrainer và ghi thông số
    huấn luyện ra file log (định dạng Text và JSONL).

    Parameters
    ----------
    log_file : str
        Đường dẫn file log để ghi nhận metric huấn luyện.
        Mặc định: "train_metrics.log".
    """

    def __init__(self, log_file: str = "train_metrics.log") -> None:
        super().__init__()
        self.log_file = log_file

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Hook được GRPOTrainer tự động gọi mỗi khi đến `logging_steps` (mặc định cứ 4 steps).
        """
        if not logs:
            return

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        step = state.global_step

        # Tự động tạo thư mục chứa log_file nếu cần
        log_path = Path(self.log_file)
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)

        # ── 1. Ghi dạng Text Log định dạng đẹp ────────────────────────────────
        lines = [
            f"[{now_str}] === TRAINING METRICS AT STEP {step:05d} ===",
        ]

        # Hiển thị các chỉ số cơ bản
        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch")
        kl = logs.get("kl")
        reward = logs.get("reward")
        reward_std = logs.get("reward_std")

        if loss is not None:
            lines.append(f"  • Loss:              {loss:.6f}")
        if grad_norm is not None:
            lines.append(f"  • Grad Norm:         {grad_norm:.6f}")
        if lr is not None:
            lines.append(f"  • Learning Rate:     {lr:.4e}")
        if reward is not None:
            reward_str = f"{reward:.4f}"
            if reward_std is not None:
                reward_str += f" (std: {reward_std:.4f})"
            lines.append(f"  • Total Reward:      {reward_str}")
        if kl is not None:
            lines.append(f"  • KL Divergence:     {kl:.6e}")
        if epoch is not None:
            lines.append(f"  • Epoch:             {epoch:.4f}")

        # Hiển thị phân rã chi tiết Reward functions
        reward_keys = [k for k in sorted(logs.keys()) if k.startswith("rewards/")]
        if reward_keys:
            lines.append("  • Reward Breakdown:")
            for k in reward_keys:
                lines.append(f"      - {k}: {logs[k]:.4f}")

        # Hiển thị thông số Completions (độ dài, clipping...)
        comp_keys = [k for k in sorted(logs.keys()) if k.startswith("completions/")]
        if comp_keys:
            lines.append("  • Completions Stats:")
            for k in comp_keys:
                lines.append(f"      - {k}: {logs[k]:.4f}")

        # Các chỉ số clipping hoặc chỉ số khác
        other_keys = [
            k for k in sorted(logs.keys())
            if k not in {"loss", "grad_norm", "learning_rate", "epoch", "kl", "reward", "reward_std", "num_tokens", "completion_length"}
            and not k.startswith("rewards/")
            and not k.startswith("completions/")
        ]
        if other_keys:
            lines.append("  • Other Stats:")
            for k in other_keys:
                val = logs[k]
                val_str = f"{val:.6e}" if isinstance(val, float) and (abs(val) < 1e-4 and val != 0) else f"{val}"
                lines.append(f"      - {k}: {val_str}")

        lines.append("=" * 65 + "\n")
        log_block = "\n".join(lines)

        # Print ra console để người dùng dễ theo dõi
        print(f"\n{log_block}")

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_block)
        except OSError as exc:
            logger.warning("Không thể ghi file log metrics %s: %s", self.log_file, exc)

        # ── 2. Ghi dạng JSONL (Lưu nguyên bản dictionary) ────────────────────
        jsonl_path = log_path.with_suffix(".jsonl")
        json_entry = {
            "timestamp": now_str,
            "step": step,
            "metrics": logs,
        }
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(json_entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Không thể ghi file JSONL metrics %s: %s", jsonl_path, exc)
