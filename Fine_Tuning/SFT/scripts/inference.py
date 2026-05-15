"""
scripts/inference.py
--------------------
Load LoRA model đã train và chạy inference trên test samples.

Các FIX so với notebook gốc:
1. Lấy câu hỏi từ raw_test (chưa map) → tránh KeyError "instruction"/"output"
2. Thêm system_prompt vào messages khi inference → nhất quán với lúc train
3. Cấu hình generation params lấy từ config, không hardcode

Cách dùng:
    python scripts/inference.py --config configs/training_config.yaml
    python scripts/inference.py --config configs/training_config.yaml --num_samples 5
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import prepare_datasets
from src.models.model import load_model_for_inference
from src.utils.config import AppConfig
from src.utils.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Override đường dẫn model (mặc định lấy từ config paths.final_model_dir)",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    return parser.parse_args()


def generate_response(
    model,
    tokenizer,
    question: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    # FIX: thêm system_prompt vào messages — nhất quán với lúc train
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
        )

    # Chỉ decode phần response, bỏ phần prompt
    generated = outputs[0][inputs.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    args = parse_args()
    cfg = AppConfig.from_yaml(args.config)
    logger = get_logger("inference")

    model_dir = args.model_dir or cfg.paths.final_model_dir
    num_samples = args.num_samples or cfg.inference.num_test_samples

    # Load model
    model, tokenizer = load_model_for_inference(model_dir, cfg.model)

    # FIX: Load raw_test (chưa format) để lấy được cột "instruction" và "output"
    #      Trong notebook gốc, dataset_test đã bị remove_columns → KeyError
    _, _, raw_test = prepare_datasets(cfg.data, tokenizer)

    sample_indices = random.sample(range(len(raw_test)), k=num_samples)

    print("\n" + "=" * 65)
    print("  BẮT ĐẦU KIỂM THỬ CHẤT LƯỢNG SINH CODE")
    print("=" * 65 + "\n")

    for rank, idx in enumerate(sample_indices, start=1):
        # FIX: raw_test vẫn còn cột "instruction" và "output"
        question    = raw_test[idx]["instruction"]
        gold_answer = raw_test[idx]["output"]

        generated = generate_response(
            model=model,
            tokenizer=tokenizer,
            question=question,
            system_prompt=cfg.data.system_prompt,
            max_new_tokens=cfg.inference.max_new_tokens,
            temperature=cfg.inference.temperature,
            top_p=cfg.inference.top_p,
        )

        print(f"─── [Mẫu {rank}/{num_samples} | index={idx}] " + "─" * 30)
        print(f"\n📌 YÊU CẦU:\n{question}\n")
        print(f"🤖 CODE MÔ HÌNH SINH:\n{generated}\n")
        print(f"✅ CODE MẪU ĐỐI CHIẾU:\n{gold_answer}\n")
        print("─" * 65 + "\n")


if __name__ == "__main__":
    main()
