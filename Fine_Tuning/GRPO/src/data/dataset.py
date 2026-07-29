"""
    src/data/dataset.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest code assistant. "
    "Always answer as helpfully as possible."
)

def load_and_split_dataset(
    dataset_name: str = "nickrosh/Evol-Instruct-Code-80k-v1",
    split: str = "train",
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[Any, Any]:
    """
    Tải dataset Evol-Instruct-Code-80k-v1 từ HuggingFace và chia train/test.

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset ID.
        Mặc định: "nickrosh/Evol-Instruct-Code-80k-v1"
    split : str
        Split cần tải. Mặc định "train" (dataset này chỉ có split train).
    test_size : float
        Tỉ lệ dữ liệu dành cho test set. Mặc định 0.1 (10%).
    seed : int
        Random seed để đảm bảo kết quả tái lặp.

    Returns
    -------
    (dataset_train, dataset_test) : tuple
        dataset_train – HuggingFace Dataset cho training (~72k mẫu).
        dataset_test  – HuggingFace Dataset cho testing (~8k mẫu).

    Raises
    ------
    ImportError
        Nếu thư viện `datasets` chưa được cài đặt.
    """

    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Thư viện 'datasets' chưa được cài đặt.\n"
            "Chạy: pip install datasets"
        ) from exc

    logger.info("Đang tải dataset: %s [split=%s]", dataset_name, split)

    dataset = load_dataset(dataset_name, split=split)
    split_result = dataset.train_test_split(test_size=test_size, seed=seed)

    dataset_train = split_result["train"]
    dataset_test  = split_result["test"]

    logger.info(
        "Dataset đã load: Train=%d | Test=%d",
        len(dataset_train),
        len(dataset_test),
    )
    print(
        f"[Dataset] Train: {len(dataset_train):,} mẫu | "
        f"Test: {len(dataset_test):,} mẫu"
    )

    return dataset_train, dataset_test


def build_grpo_dataset(
    dataset_train: Any,
    tokenizer: Any,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Any:
    """
    Chuyển đổi dataset thô thành format phù hợp cho GRPOTrainer.

    GRPO chỉ cần cột "prompt" — model sẽ tự sinh nhiều responses (num_generations)
    để tính reward tương đối giữa các responses trong cùng một nhóm.

    Parameters
    ----------
    dataset_train : datasets.Dataset
        Dataset thô từ `load_and_split_dataset()`.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer tương ứng với model. Dùng để áp dụng chat template.
    system_prompt : str
        System prompt định nghĩa vai trò của assistant.

    Returns
    -------
    grpo_dataset : datasets.Dataset
        Dataset chỉ chứa cột "prompt" đã được format theo chat template,
        bao gồm generation prompt token để model tiếp tục sinh.
    """

    logger.info("Đang tạo GRPO dataset với system_prompt=%r...", system_prompt[:60])

    def _format_prompt_only(examples: Dict[str, Any]) -> Dict[str, list]:
        """
        Format từng instruction thành chat template prompt.

        GRPO chỉ cần phần prompt — không cần response.
        `add_generation_prompt=True` thêm token "<|assistant|>" để model
        biết phải bắt đầu sinh response.
        """
        prompts = []
        for instruction in examples["instruction"]:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": instruction},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # Cần thiết cho GRPO — model tự sinh tiếp
            )
            prompts.append(prompt)
        return {"prompt": prompts}

    grpo_dataset = dataset_train.map(
        _format_prompt_only,
        batched=True,
        remove_columns=dataset_train.column_names,  # Xóa tất cả columns gốc
        desc="Formatting prompts",
    )

    logger.info("GRPO dataset đã tạo: %d mẫu", len(grpo_dataset))
    print(f"[Dataset] GRPO dataset: {len(grpo_dataset):,} mẫu")
    print(f"[Dataset] Ví dụ prompt:\n{grpo_dataset[0]['prompt'][:300]}...")

    return grpo_dataset
