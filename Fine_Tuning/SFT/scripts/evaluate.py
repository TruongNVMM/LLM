"""
scripts/evaluate.py
-------------------
Đánh giá model sau khi train với các metric phù hợp hơn token_accuracy:
  - eval_loss   : metric chính, reflect trực tiếp objective
  - CodeBLEU    : đánh giá chất lượng code sinh ra (có cả syntax + dataflow)
  - Pass@k      : nếu có unit test (HumanEval / MBPP)

Cách dùng:
    # Đánh giá bằng CodeBLEU trên test set:
    python scripts/evaluate.py --config configs/training_config.yaml

    # Đánh giá Pass@1 trên HumanEval (cần cài human_eval):
    python scripts/evaluate.py --config configs/training_config.yaml --benchmark humaneval
"""
from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import prepare_datasets
from src.models.model import load_model_for_inference
from src.utils.config import AppConfig
from src.utils.logging_utils import get_logger
from scripts.inference import generate_response


logger = get_logger("evaluate")


# ─────────────────────────────────────────────
# CodeBLEU
# ─────────────────────────────────────────────

def evaluate_codebleu(
    model,
    tokenizer,
    raw_test,
    cfg: AppConfig,
    num_samples: int = 100,
) -> float:
    """
    Tính CodeBLEU trên num_samples mẫu từ raw_test.
    pip install codebleu
    """
    try:
        from codebleu import calc_codebleu
    except ImportError:
        logger.error("Chưa cài codebleu: pip install codebleu")
        return -1.0

    import random
    indices = random.sample(range(len(raw_test)), k=min(num_samples, len(raw_test)))

    references, predictions = [], []
    for i, idx in enumerate(indices):
        question = raw_test[idx]["instruction"]
        gold     = raw_test[idx]["output"]
        pred     = generate_response(
            model, tokenizer,
            question=question,
            system_prompt=cfg.data.system_prompt,
            max_new_tokens=cfg.inference.max_new_tokens,
            temperature=cfg.inference.temperature,
            top_p=cfg.inference.top_p,
        )
        references.append(gold)
        predictions.append(pred)
        if (i + 1) % 10 == 0:
            logger.info(f"CodeBLEU progress: {i+1}/{len(indices)}")

    result = calc_codebleu(
        references=references,
        predictions=predictions,
        lang="python",
        weights=(0.25, 0.25, 0.25, 0.25),
    )
    score = result["codebleu"]
    logger.info(f"CodeBLEU: {score:.4f}")
    logger.info(f"  ngram_match    : {result['ngram_match_score']:.4f}")
    logger.info(f"  weighted_ngram : {result['weighted_ngram_match_score']:.4f}")
    logger.info(f"  syntax_match   : {result['syntax_match_score']:.4f}")
    logger.info(f"  dataflow_match : {result['dataflow_match_score']:.4f}")
    return score


# ─────────────────────────────────────────────
# Pass@k
# ─────────────────────────────────────────────

def _pass_at_k(n: int, c: int, k: int) -> float:
    """Công thức Pass@k (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def evaluate_pass_at_k(
    model,
    tokenizer,
    problems: List[dict],
    cfg: AppConfig,
    k: int = 1,
    n_samples: int = 10,
) -> float:
    """
    Tính Pass@k bằng cách chạy code sinh ra trong subprocess.
    problems: list of {"prompt": str, "test": str}  (định dạng HumanEval / MBPP)
    """
    import subprocess
    import tempfile
    import os

    results = []
    for i, problem in enumerate(problems):
        passed = 0
        for _ in range(n_samples):
            code = generate_response(
                model, tokenizer,
                question=problem["prompt"],
                system_prompt=cfg.data.system_prompt,
                max_new_tokens=cfg.inference.max_new_tokens,
                temperature=0.8,   # temperature cao hơn khi sample nhiều lần
                top_p=cfg.inference.top_p,
            )
            full_code = code + "\n" + problem["test"]
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False, encoding="utf-8"
                ) as f:
                    f.write(full_code)
                    fname = f.name
                result = subprocess.run(
                    ["python", fname],
                    timeout=10,
                    capture_output=True,
                )
                if result.returncode == 0:
                    passed += 1
            except Exception:
                pass
            finally:
                try:
                    os.unlink(fname)
                except Exception:
                    pass
        results.append(_pass_at_k(n_samples, passed, k))
        logger.info(f"Pass@{k} problem {i+1}/{len(problems)}: passed={passed}/{n_samples}")

    score = sum(results) / len(results)
    logger.info(f"Pass@{k} = {score:.4f}")
    return score


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--config",  type=str, default="configs/training_config.yaml")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["codebleu", "humaneval"],
        default="codebleu",
        help="Metric để đánh giá",
    )
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Số mẫu dùng để đánh giá (CodeBLEU)")
    parser.add_argument("--pass_k", type=int, default=1,
                        help="k trong Pass@k")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = AppConfig.from_yaml(args.config)
    model_dir = args.model_dir or cfg.paths.final_model_dir

    model, tokenizer = load_model_for_inference(model_dir, cfg.model)
    _, _, raw_test = prepare_datasets(cfg.data, tokenizer)

    if args.benchmark == "codebleu":
        logger.info(f"Đánh giá CodeBLEU trên {args.num_samples} mẫu...")
        evaluate_codebleu(model, tokenizer, raw_test, cfg, num_samples=args.num_samples)

    elif args.benchmark == "humaneval":
        # Cần cài: pip install human-eval
        try:
            from human_eval.data import read_problems
            problems = list(read_problems().values())
            logger.info(f"HumanEval: {len(problems)} bài | Pass@{args.pass_k}")
            evaluate_pass_at_k(model, tokenizer, problems, cfg, k=args.pass_k)
        except ImportError:
            logger.error("Chưa cài human-eval: pip install human-eval")


if __name__ == "__main__":
    main()
