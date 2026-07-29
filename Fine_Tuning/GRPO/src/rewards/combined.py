"""
src/rewards/combined.py
========================
Hàm reward tổng hợp — kết hợp 5 reward thành phần với trọng số.

Map từ Notebook:
  - Cell 26 : compute_reward()

Trọng số được thiết kế theo nguyên tắc ưu tiên:
  1. Executable  (0.40) — mục tiêu chính của code generation
  2. Syntax      (0.25) — tiền đề của executable, reward trung gian tốt
  3. No Placeholder (0.20) — chống gian lận / reward hacking
  4. Format      (0.10) — UX, ít quan trọng hơn correctness
  5. Length      (0.05) — heuristic phụ

Tổng = 1.00  → score nằm trong [0.0, 1.0]
"""

from __future__ import annotations

import logging
from typing import Dict, List

from .code_rewards import (
    reward_code_executable,
    reward_code_format,
    reward_length_quality,
    reward_no_placeholder,
    reward_syntax_valid,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: Dict[str, float] = {
    "executable":      0.40,
    "syntax":          0.25,
    "no_placeholder":  0.20,
    "format":          0.10,
    "length":          0.05,
}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    completions: List[str],
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
    execution_timeout: int = 5,
    **kwargs,
) -> List[float]:
    """
    Reward tổng hợp — tính trung bình có trọng số từ 5 reward thành phần.

    Đây là hàm reward chính được truyền vào GRPOTrainer. Kết hợp tất cả
    các khía cạnh chất lượng code vào một điểm số duy nhất trong [0.0, 1.0].

    Parameters
    ----------
    completions : List[str]
        Danh sách responses sinh ra bởi model (num_generations responses
        cho mỗi prompt).
    weights : Dict[str, float]
        Từ điển trọng số cho từng reward component.
        Tổng trọng số nên bằng 1.0.
        Keys: "executable", "syntax", "no_placeholder", "format", "length"
    execution_timeout : int
        Timeout (giây) cho reward_code_executable. Mặc định 5s.
    **kwargs
        Các tham số khác do GRPOTrainer truyền vào (prompts, inputs, ...).

    Returns
    -------
    List[float]
        Danh sách combined scores trong [0.0, 1.0],
        tương ứng với từng completion.

    Examples
    --------
    >>> completions = ["```python\\ndef add(a, b):\\n    return a + b\\n```"]
    >>> scores = compute_reward(completions)
    >>> print(scores)  # [0.9500]

    Notes
    -----
    Thứ tự gọi reward functions được tối ưu để fail-fast:
    - format và syntax nhanh (không cần subprocess)
    - executable chậm hơn (cần spawn subprocess)
    Tuy nhiên, để đảm bảo tính nhất quán, tất cả đều được tính đầy đủ.
    """
    w = weights

    # ── Tính từng reward component ──────────────────────────────────────
    r_exec     = reward_code_executable(completions, timeout=execution_timeout, **kwargs)
    r_syntax   = reward_syntax_valid(completions, **kwargs)
    r_noplace  = reward_no_placeholder(completions, **kwargs)
    r_format   = reward_code_format(completions, **kwargs)
    r_length   = reward_length_quality(completions, **kwargs)

    # ── Tính tổng có trọng số ────────────────────────────────────────────
    combined: List[float] = []
    for i in range(len(completions)):
        score = (
            w["executable"]     * r_exec[i]    +
            w["syntax"]         * r_syntax[i]  +
            w["no_placeholder"] * r_noplace[i] +
            w["format"]         * r_format[i]  +
            w["length"]         * r_length[i]
        )
        combined.append(round(score, 4))

    logger.debug(
        "compute_reward: n=%d | avg=%.4f | min=%.4f | max=%.4f",
        len(combined),
        sum(combined) / len(combined) if combined else 0,
        min(combined) if combined else 0,
        max(combined) if combined else 0,
    )

    return combined
