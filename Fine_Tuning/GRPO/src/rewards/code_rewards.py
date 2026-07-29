"""
src/rewards/code_rewards.py
============================
5 hàm reward thành phần đánh giá chất lượng code Python sinh ra bởi LLM.

Map từ Notebook:
  - Cell 14 : import + _extract_code helper
  - Cell 16 : reward_code_format()      — Reward 1: Format
  - Cell 18 : reward_syntax_valid()     — Reward 2: Syntax
  - Cell 20 : reward_code_executable()  — Reward 3: Executable
  - Cell 22 : reward_no_placeholder()   — Reward 4: No Placeholder
  - Cell 24 : reward_length_quality()   — Reward 5: Length

Thiết kế reward theo nguyên tắc:
  - Mỗi hàm độc lập, nhận List[str] và trả về List[float] trong [0.0, 1.0]
  - Compatible với GRPOTrainer.reward_funcs interface của thư viện TRL
  - Hàm nhẹ nhất (format, syntax) nên được gọi trước để fail-fast

Thứ tự ưu tiên (theo trọng số trong combined reward):
  1. Executable  (0.40) — Mục tiêu chính
  2. Syntax      (0.25) — Tiền đề của executable
  3. No Placeholder (0.20) — Chống reward hacking
  4. Format      (0.10) — UX / usability
  5. Length      (0.05) — Heuristic chất lượng
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import tempfile
from typing import List

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_code(response: str) -> str:
    """
    Trích xuất đoạn code Python từ markdown code block trong response.

    Ưu tiên tìm ``` ```python ... ``` ``` trước,
    fallback về bất kỳ ``` ```...``` ``` nào,
    và cuối cùng trả về toàn bộ response nếu không có code block.

    Parameters
    ----------
    response : str
        Response đầy đủ từ model, có thể chứa text + code block.

    Returns
    -------
    str
        Đoạn code đã được trim, hoặc chuỗi rỗng nếu không có code.

    Examples
    --------
    >>> resp = "Here's the solution:\n```python\ndef foo(): pass\n```"
    >>> _extract_code(resp)
    'def foo(): pass'
    """
    # Pattern: ```python ... ``` hoặc ``` ... ```
    pattern = r"```(?:python)?\n?(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    # Fallback: toàn bộ response
    return response.strip()


# ─────────────────────────────────────────────────────────────────────────────
# REWARD 1: FORMAT
# ─────────────────────────────────────────────────────────────────────────────

def reward_code_format(completions: List[str], **kwargs) -> List[float]:
    """
    Kiểm tra response có wrap code trong markdown code block hay không.

    Dạy model luôn trình bày code trong ``` ```python ... ``` ``` thay vì
    raw text, vì đây là tiêu chuẩn usability quan trọng.

    Score:
      1.0  → có ``` ```python ... ``` ``` đúng chuẩn
      0.5  → có ``` ``` ``` nhưng không chỉ định ngôn ngữ python
      0.0  → không có code block nào

    Parameters
    ----------
    completions : List[str]
        Danh sách responses sinh ra bởi model (num_generations responses).
    **kwargs
        Các tham số khác do GRPOTrainer truyền vào (prompts, inputs, ...).

    Returns
    -------
    List[float]
        Danh sách scores tương ứng với từng completion.
    """
    scores: List[float] = []
    for resp in completions:
        if "```python" in resp and "```" in resp.split("```python", 1)[1]:
            # Có cả opening ```python và closing ```
            scores.append(1.0)
        elif "```" in resp:
            # Có code block nhưng không chỉ định ngôn ngữ
            scores.append(0.5)
        else:
            # Không có code block
            scores.append(0.0)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# REWARD 2: SYNTAX
# ─────────────────────────────────────────────────────────────────────────────

def reward_syntax_valid(completions: List[str], **kwargs) -> List[float]:
    """
    Kiểm tra code Python có cú pháp hợp lệ bằng cách parse AST.

    Nhanh hơn execution (không cần chạy subprocess), dùng như reward
    trung gian giúp model học viết code có syntax đúng trước.

    Score:
      1.0  → ast.parse() thành công — syntax hoàn toàn hợp lệ
      0.3  → có code nhưng syntax sai (ít nhất model đã cố viết code)
      0.0  → không trích xuất được code nào

    Parameters
    ----------
    completions : List[str]
        Danh sách responses sinh ra bởi model.
    **kwargs
        Các tham số khác do GRPOTrainer truyền vào.

    Returns
    -------
    List[float]
        Danh sách scores tương ứng với từng completion.
    """
    scores: List[float] = []
    for resp in completions:
        code = _extract_code(resp)
        if not code:
            scores.append(0.0)
            continue
        try:
            ast.parse(code)
            scores.append(1.0)
        except SyntaxError:
            # Có code nhưng lỗi syntax — vẫn cho điểm một phần
            scores.append(0.3)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# REWARD 3: EXECUTABLE
# ─────────────────────────────────────────────────────────────────────────────

def reward_code_executable(
    completions: List[str],
    timeout: int = 5,
    **kwargs,
) -> List[float]:
    """
    Thực sự chạy code trong subprocess sandbox để kiểm tra executable.

    Quan trọng hơn syntax check vì: import sai, undefined variable,
    kiểu dữ liệu sai đều pass syntax nhưng fail khi chạy thực tế.

    Score:
      1.0  → subprocess.run() returncode == 0 (không có exception)
      0.2  → syntax đúng nhưng có runtime error
      0.1  → code gây infinite loop (bị timeout)
      0.0  → không có code / syntax sai

    Parameters
    ----------
    completions : List[str]
        Danh sách responses sinh ra bởi model.
    timeout : int
        Số giây tối đa cho phép code chạy. Tránh infinite loop.
        Mặc định: 5 giây.
    **kwargs
        Các tham số khác do GRPOTrainer truyền vào.

    Returns
    -------
    List[float]
        Danh sách scores tương ứng với từng completion.

    Notes
    -----
    Mỗi đoạn code được chạy trong một subprocess riêng biệt để đảm bảo
    cô lập — code của model không thể ảnh hưởng đến process chính.
    """
    scores: List[float] = []

    for resp in completions:
        code = _extract_code(resp)
        if not code:
            scores.append(0.0)
            continue

        # Kiểm tra syntax trước (nhanh) để không tốn thời gian spawn subprocess
        try:
            ast.parse(code)
        except SyntaxError:
            scores.append(0.0)
            continue

        # Chạy code trong subprocess sandbox
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

            result = subprocess.run(
                ["python", fname],
                timeout=timeout,        # Tránh infinite loop
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                scores.append(1.0)     # Chạy thành công
            else:
                scores.append(0.2)     # Runtime error nhưng syntax đúng

        except subprocess.TimeoutExpired:
            scores.append(0.1)         # Code có infinite loop
        except Exception as exc:       # noqa: BLE001
            logger.debug("Lỗi khi chạy subprocess: %s", exc)
            scores.append(0.0)
        finally:
            if fname is not None:
                try:
                    os.unlink(fname)
                except OSError:
                    pass

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# REWARD 4: NO PLACEHOLDER
# ─────────────────────────────────────────────────────────────────────────────

def reward_no_placeholder(completions: List[str], **kwargs) -> List[float]:
    """
    Phát hiện và phạt code chứa placeholder — chống reward hacking.

    Model có thể học "gian lận" bằng cách viết code trông hợp lệ nhưng
    thực ra là rỗng (pass, TODO, raise NotImplementedError, ...).
    Reward này ngăn chặn hành vi đó bằng cách phạt nặng.

    Score:
      1.0  → không có placeholder — code thực sự implement logic
      0.0  → phát hiện ít nhất một pattern placeholder

    Các pattern bị phát hiện:
      - `pass`               — statement rỗng
      - `# TODO`             — chưa implement
      - `# FIXME`            — biết là sai nhưng chưa sửa
      - `raise NotImplementedError` — chưa implement
      - `...`                — bare ellipsis thay cho implementation

    Parameters
    ----------
    completions : List[str]
        Danh sách responses sinh ra bởi model.
    **kwargs
        Các tham số khác do GRPOTrainer truyền vào.

    Returns
    -------
    List[float]
        Danh sách scores tương ứng với từng completion.
    """
    # Các pattern regex để phát hiện placeholder
    BAD_PATTERNS = [
        r"\bpass\b",
        r"#\s*TODO",
        r"#\s*FIXME",
        r"raise\s+NotImplementedError",
        r"\.\.\.",                  # Bare ellipsis thay cho implementation
    ]

    scores: List[float] = []
    for resp in completions:
        code = _extract_code(resp)
        has_placeholder = any(re.search(p, code) for p in BAD_PATTERNS)
        scores.append(0.0 if has_placeholder else 1.0)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# REWARD 5: LENGTH QUALITY
# ─────────────────────────────────────────────────────────────────────────────

def reward_length_quality(completions: List[str], **kwargs) -> List[float]:
    """
    Đánh giá chất lượng dựa trên độ dài của code.

    Code quá ngắn (< 3 dòng) thường thiếu logic. Code quá dài (> 100 dòng)
    thường là hallucination hoặc lặp code vô nghĩa.

    Score (dựa trên số dòng code thực tế, không đếm dòng trống):
      1.0  → 3–50 dòng  — độ dài lý tưởng
      0.7  → 51–100 dòng — hơi dài, có thể vẫn tốt
      0.2  → < 3 dòng   — quá ngắn, thiếu logic
      0.3  → > 100 dòng — quá dài, nghi ngờ hallucination

    Parameters
    ----------
    completions : List[str]
        Danh sách responses sinh ra bởi model.
    **kwargs
        Các tham số khác do GRPOTrainer truyền vào.

    Returns
    -------
    List[float]
        Danh sách scores tương ứng với từng completion.
    """
    scores: List[float] = []
    for resp in completions:
        code = _extract_code(resp)
        # Chỉ đếm dòng có nội dung thực sự (bỏ dòng trống)
        non_empty_lines = [line for line in code.split("\n") if line.strip()]
        n = len(non_empty_lines)

        if n < 3:
            scores.append(0.2)   # Quá ngắn
        elif n <= 50:
            scores.append(1.0)   # Lý tưởng
        elif n <= 100:
            scores.append(0.7)   # Hơi dài
        else:
            scores.append(0.3)   # Quá dài

    return scores
