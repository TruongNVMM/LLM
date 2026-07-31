"""
src/callbacks/early_stopping.py

Custom Trainer Callback thực hiện validation định kỳ và early stopping.

Chức năng chính:
  1. Mỗi `eval_steps` steps: lấy mẫu từ val_dataset, sinh response
     và đánh giá theo 4 metrics:
       - val/pass_rate   : tỉ lệ code chạy thành công
       - val/syntax_rate : tỉ lệ code đúng syntax
       - val/format_rate : tỉ lệ code đúng định dạng markdown
       - val/mean_reward : điểm thưởng trung bình

  2. Lưu checkpoint tốt nhất (best_model_dir) khi metric cải thiện.

  3. Dừng training sớm nếu metric không cải thiện sau `patience` lần eval
     liên tiếp (improvement < min_delta).

Thiết kế:
  - State được lưu vào file JSON (callback_state.json) để có thể resume
    nếu training bị gián đoạn.
  - Sử dụng `model.eval()` trong khi eval, `model.train()` sau khi xong.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class CodeQualityCallback(TrainerCallback):
    """
    Callback đánh giá chất lượng code định kỳ và thực hiện early stopping.

    Parameters
    ----------
    val_dataset : datasets.Dataset
        Validation dataset có cột "prompt" (output của `build_grpo_dataset()`).
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer để encode prompt và decode response.
    num_samples : int
        Số mẫu lấy ngẫu nhiên từ val_dataset cho mỗi lần eval.
        Mặc định 50 (~3 phút T4 GPU).
    eval_steps : int
        Eval mỗi bao nhiêu training steps. Mặc định 4 (cứ sau 4 lần chạy).
    patience : int
        Số lần eval liên tiếp không cải thiện trước khi dừng training.
        Mặc định 3.
    min_delta : float
        Ngưỡng cải thiện tối thiểu. Nhỏ hơn ngưỡng này không tính là cải thiện.
        Mặc định 0.005.
    monitor : str
        Tên metric dùng để theo dõi và so sánh.
        Mặc định "val/mean_reward".
    save_best : bool
        True = lưu model khi metric cải thiện. Mặc định True.
    best_model_dir : str
        Thư mục lưu best checkpoint. Mặc định "best_grpo_checkpoint".
    state_file : str
        File JSON để lưu trạng thái callback (hỗ trợ resume).
        Mặc định "callback_state.json".
    execution_timeout : int
        Timeout (giây) cho việc chạy code trong subprocess. Mặc định 5.
    log_file : str
        File log lưu trữ các metric đánh giá qua từng đợt eval.
        Mặc định "eval_metrics.log".
    """

    def __init__(
        self,
        val_dataset: Any,
        tokenizer: Any,
        num_samples: int = 50,
        eval_steps: int = 4,
        patience: int = 3,
        min_delta: float = 0.005,
        monitor: str = "val/mean_reward",
        save_best: bool = True,
        best_model_dir: str = "best_grpo_checkpoint",
        state_file: str = "callback_state.json",
        execution_timeout: int = 5,
        log_file: str = "eval_metrics.log",
    ) -> None:
        super().__init__()
        self.tokenizer         = tokenizer
        self.num_samples       = num_samples
        self.eval_steps        = eval_steps
        self.patience          = patience
        self.min_delta         = min_delta
        self.monitor           = monitor
        self.save_best         = save_best                                                 
        self.best_model_dir    = best_model_dir
        self.state_file        = state_file
        self.execution_timeout = execution_timeout
        self.log_file          = log_file

        # Lấy ngẫu nhiên num_samples mẫu từ val_dataset
        val_dataset  = val_dataset.shuffle(seed=42)
        n            = min(num_samples, len(val_dataset))
        self.val_samples: List[Dict[str, Any]] = [
            val_dataset[i] for i in range(n)
        ]

        # State — sẽ được load lại nếu state_file tồn tại (resume)
        self.best_value: Optional[float] = None
        self.best_step:  int             = 0
        self.wait_count: int             = 0

        self._load_state()
        logger.info(
            "[CodeQualityCallback] Khởi tạo | val_samples=%d | "
            "eval_steps=%d | patience=%d | monitor=%s | log_file=%s",
            len(self.val_samples), eval_steps, patience, monitor, log_file,
        )

    # State persistence
    def _save_state(self) -> None:
        """Lưu trạng thái callback ra file JSON để hỗ trợ resume."""
        state = {
            "best_value": self.best_value,
            "best_step":  self.best_step,
            "wait_count": self.wait_count,
        }
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except OSError as exc:
            logger.warning("Không thể lưu callback state: %s", exc)

    def _load_state(self) -> None:
        """Load trạng thái callback từ file JSON nếu tồn tại."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, encoding="utf-8") as f:
                state = json.load(f)
            self.best_value = state.get("best_value")
            self.best_step  = state.get("best_step", 0)
            self.wait_count = state.get("wait_count", 0)
            logger.info(
                "[CodeQualityCallback] Resume từ state: best=%.4f tại step %d",
                self.best_value or 0.0,
                self.best_step,
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Không thể load callback state: %s", exc)

    # Generation helper
    def _generate(self, model: Any, prompt: str) -> str:
        """
        Sinh response từ model với greedy + temperature decoding.

        Parameters
        ----------
        model : FastLanguageModel
            Model đang train (đã được chuyển về eval mode bởi caller).
        prompt : str
            Prompt đã được format theo chat template.

        Returns
        -------
        str
            Response của model (không bao gồm phần prompt).
        """
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                top_p=0.9,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Chỉ lấy phần tokens được sinh ra (bỏ phần prompt)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── Code extraction helper ─────────────────────────────────────────

    @staticmethod
    def _extract_code(response: str) -> str:
        """Trích xuất code từ markdown code block."""
        pattern = r"```(?:python)?\n?(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return response.strip()

    # ── Execution helper ───────────────────────────────────────────────

    def _is_executable(self, code: str) -> bool:
        """
        Kiểm tra code có thực sự chạy được không (không có runtime error).

        Returns
        -------
        bool
            True nếu subprocess returncode == 0.
        """
        try:
            ast.parse(code)
        except SyntaxError:
            return False

        fname = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(code)
                fname = f.name

            return subprocess.run(
                ["python", fname],
                timeout=self.execution_timeout,
                capture_output=True,
            ).returncode == 0

        except Exception:  # noqa: BLE001
            return False
        finally:
            if fname is not None:
                try:
                    os.unlink(fname)
                except OSError:
                    pass

    # ── Core eval ─────────────────────────────────────────────────────

    def _run_eval(self, model: Any) -> Dict[str, float]:
        """
        Chạy evaluation trên val_samples và trả về metrics dict.

        Returns
        -------
        dict với các keys:
          - "val/pass_rate"   : tỉ lệ code chạy thành công [0, 1]
          - "val/syntax_rate" : tỉ lệ code đúng syntax [0, 1]
          - "val/format_rate" : tỉ lệ code đúng format [0, 1]
          - "val/mean_reward" : reward trung bình [0, 1]
        """
        pass_count = syntax_count = format_count = reward_sum = 0
        total = len(self.val_samples)

        for sample in self.val_samples:
            resp = self._generate(model, sample["prompt"])
            code = self._extract_code(resp)

            has_format = "```python" in resp and resp.count("```") >= 2
            has_syntax = False
            has_exec   = False

            # Kiểm tra syntax
            try:
                ast.parse(code)
                has_syntax = True
            except SyntaxError:
                pass

            # Kiểm tra executable (chỉ nếu syntax đúng)
            if has_syntax:
                has_exec = self._is_executable(code)

            # Tính reward cho mẫu này
            reward = (
                0.40 * float(has_exec)   +
                0.25 * float(has_syntax) +
                0.15 * float(has_format)
            )

            format_count += int(has_format)
            syntax_count += int(has_syntax)
            pass_count   += int(has_exec)
            reward_sum   += reward

        return {
            "val/pass_rate":   round(pass_count   / total, 4),
            "val/syntax_rate": round(syntax_count / total, 4),
            "val/format_rate": round(format_count / total, 4),
            "val/mean_reward": round(reward_sum   / total, 4),
        }

    # ── File Logging helper ───────────────────────────────────────────

    def _log_metrics_to_file(
        self,
        metrics: Dict[str, float],
        state: TrainerState,
        status_msg: str,
        improved: bool,
    ) -> None:
        """
        Ghi log các metric đánh giá và thông tin early stopping ra file log.

        Hỗ trợ ghi đồng thời 2 dạng:
          1. Text format trong self.log_file (mặc định: eval_metrics.log)
          2. JSON Lines format trong <self.log_file>.jsonl (mặc định: eval_metrics.jsonl)
        """
        import datetime

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        step    = state.global_step

        # Tự động tạo thư mục chứa file log nếu chưa tồn tại
        log_path = Path(self.log_file)
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Text log formatted entry
        log_entry = (
            f"[{now_str}] STEP {step:05d} | "
            f"Pass Rate: {metrics['val/pass_rate']:.2%} | "
            f"Syntax Rate: {metrics['val/syntax_rate']:.2%} | "
            f"Format Rate: {metrics['val/format_rate']:.2%} | "
            f"Mean Reward: {metrics['val/mean_reward']:.4f} | "
            f"Status: {status_msg}\n"
        )

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except OSError as exc:
            logger.warning("Không thể ghi log metrics vào file %s: %s", self.log_file, exc)

        # 2. JSONL formatted entry cho việc parse/plot dữ liệu tự động
        jsonl_path = log_path.with_suffix(".jsonl")
        json_entry = {
            "timestamp": now_str,
            "step": step,
            "metrics": metrics,
            "status": status_msg,
            "improved": improved,
            "best_value": self.best_value,
            "best_step": self.best_step,
            "wait_count": self.wait_count,
            "patience": self.patience,
        }
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(json_entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Không thể ghi JSONL metrics vào file %s: %s", jsonl_path, exc)

    # ── Early stopping logic ───────────────────────────────────────────

    def _check_early_stop(
        self,
        metrics: Dict[str, float],
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
    ) -> TrainerControl:
        """
        Kiểm tra điều kiện early stopping, cập nhật best state và ghi file log.

        Parameters
        ----------
        metrics : dict
            Kết quả từ `_run_eval()`.
        state : TrainerState
            Trạng thái training hiện tại.
        control : TrainerControl
            Control object — set `should_training_stop=True` để dừng.
        model : optional
            Model hiện tại để lưu nếu là best.

        Returns
        -------
        TrainerControl
            Control đã được cập nhật.
        """
        current   = metrics[self.monitor]
        old_value = self.best_value  # Lưu lại trước khi cập nhật

        improved = (old_value is None) or (current > old_value + self.min_delta)

        if improved:
            # Cập nhật state trước
            self.best_value = current
            self.best_step  = state.global_step
            self.wait_count = 0

            if old_value is None:
                status_msg = f"Khởi tạo best {self.monitor}={current:.4f}"
            else:
                status_msg = (
                    f"✓ Cải thiện: {old_value:.4f} → {current:.4f} "
                    f"(+{current - old_value:.4f}) | Reset patience (0/{self.patience})"
                )

            if old_value is None:
                print(
                    f"  [EarlyStopping] Khởi tạo "
                    f"best {self.monitor}={current:.4f}"
                )
            else:
                print(f"  [EarlyStopping] {status_msg}")

            # Lưu best model
            if self.save_best and model is not None:
                model.save_pretrained(self.best_model_dir)
                self.tokenizer.save_pretrained(self.best_model_dir)
                print(
                    f"  [EarlyStopping] 💾 Best model saved "
                    f"→ {self.best_model_dir}"
                )
        else:
            self.wait_count += 1
            remaining = self.patience - self.wait_count
            status_msg = (
                f"✗ Không cải thiện ({self.wait_count}/{self.patience}) | "
                f"best={self.best_value:.4f} tại step {self.best_step} | "
                f"còn {remaining} lần"
            )
            print(f"  [EarlyStopping] {status_msg}")

            if self.wait_count >= self.patience:
                print(f"\n{'=' * 55}")
                print(f"  [EarlyStopping] DỪNG tại step {state.global_step}")
                print(
                    f"  Best {self.monitor} = {self.best_value:.4f} "
                    f"tại step {self.best_step}"
                )
                print(f"{'=' * 55}\n")
                control.should_training_stop = True

        # Ghi metric và status ra file log
        self._log_metrics_to_file(metrics, state, status_msg, improved)

        # Lưu state ra file (hỗ trợ resume)
        self._save_state()
        return control

    # ── Trainer hook ───────────────────────────────────────────────────

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        **kwargs,
    ) -> TrainerControl:
        """
        Hook được gọi sau mỗi training step.

        Chỉ thực hiện eval khi step % eval_steps == 0.
        """
        # Chỉ eval tại các bước được chỉ định (và bỏ qua step 0)
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return control

        print(
            f"\n[Callback] Eval {len(self.val_samples)} mẫu "
            f"tại step {state.global_step}..."
        )

        # Switch sang eval mode
        model.eval()
        metrics = self._run_eval(model)
        model.train()

        # Log metrics vào history
        state.log_history.append({"step": state.global_step, **metrics})

        # Log lên W&B nếu được cấu hình
        if "wandb" in (args.report_to or []):
            try:
                import wandb  # noqa: PLC0415
                wandb.log({"step": state.global_step, **metrics})
            except ImportError:
                logger.warning("wandb chưa được cài. Bỏ qua W&B logging.")

        # Print kết quả
        print(
            f"  pass={metrics['val/pass_rate']:.2%} | "
            f"syntax={metrics['val/syntax_rate']:.2%} | "
            f"format={metrics['val/format_rate']:.2%} | "
            f"reward={metrics['val/mean_reward']:.4f}"
        )

        # Kiểm tra early stopping
        control = self._check_early_stop(metrics, state, control, model)

        return control
