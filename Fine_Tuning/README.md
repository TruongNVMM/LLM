# Fine-Tuning for Python Code Generation

This repository presents a professional two-stage fine-tuning pipeline for adapting large language models to Python code generation tasks. The workflow combines:

- Supervised Fine-Tuning (SFT) for domain adaptation and instruction following
- Group Relative Policy Optimization (GRPO) for further alignment using execution-based feedback

The project is designed to improve model quality in terms of correctness, syntax validity, executability, and formatting, while remaining practical for resource-constrained environments such as Google Colab and single-GPU setups.

---

## Overview

Large language models pretrained on general text can perform reasonably well on code-related prompts, but they often require task-specific adaptation to become reliable for software engineering scenarios. This project addresses that need through a structured training strategy:

1. SFT is used to teach the model the expected structure, style, and reasoning patterns for Python code generation.
2. GRPO is then applied to refine the model further by rewarding high-quality outputs based on execution and validation signals.

This two-step approach provides a strong balance between imitation learning and reinforcement-style optimization.

---

## Why This Approach?

### 1. SFT: foundational adaptation
SFT establishes a strong baseline by training the model on curated instruction-following examples. It helps the model learn:

- proper code formatting
- task decomposition
- Python-specific syntax and structure
- better response quality for programming prompts

### 2. GRPO: targeted refinement
After SFT, GRPO is used to optimize the model with reward signals derived from code quality. This helps improve:

- executability
- syntax validity
- code completeness
- formatting consistency
- reduction of placeholder or incomplete outputs

In practice, this makes the model more reliable for real-world coding tasks than SFT alone.

---

## Project Structure

```text
Fine_Tuning/
├── GRPO/
│   ├── configs/
│   ├── logs/
│   ├── model/
│   ├── notebook/
│   ├── scripts/
│   └── src/
├── SFT/
│   ├── configs/
│   ├── notebook/
│   ├── scripts/
│   └── src/
└── readme.md
```

---

## Training Pipeline

### Stage 1 — Supervised Fine-Tuning (SFT)
The SFT stage uses a base instruction-tuned model and fine-tunes it with LoRA/QLoRA techniques for Python code generation tasks. The objective is to teach the model how to produce high-quality code responses in a structured and consistent manner.

Key characteristics:

- efficient parameter-efficient fine-tuning
- support for Unsloth-based acceleration
- lightweight adaptation suitable for limited GPU resources
- strong initialization for downstream optimization

### Stage 2 — GRPO Alignment
The GRPO stage starts from the SFT checkpoint and further improves generation quality through group-based policy optimization. For each prompt, multiple candidate completions are generated and scored using reward functions that evaluate:

- code execution success
- syntax correctness
- avoidance of placeholders or incomplete patterns
- formatting quality
- response length appropriateness

This process encourages the model to prefer outputs that are more useful and robust.

---

## Key Components

### SFT Project
The SFT pipeline focuses on building a strong base model for code generation and instruction following. More details are available in the SFT documentation.

### GRPO Project
The GRPO pipeline focuses on reinforcement-style refinement using reward-based learning and execution feedback. More details are available in the GRPO documentation.

---

## Evaluation Results

The following results reflect the outcomes reported by the two training pipelines in their respective documentation.

| Stage | Benchmark | Pass@1 | Pass@5 | Pass@10 |
| :--- | :--- | ---: | ---: | ---: |
| SFT | HumanEval | 0.73 | 0.85 | 0.91 |
| SFT | HumanEval+ | 0.65 | 0.78 | 0.85 |
| GRPO (after SFT) | HumanEval | 0.82 | 0.85 | 0.91 |
| GRPO (after SFT) | HumanEval+ | 0.70 | 0.75 | 0.80 |

These results indicate that the GRPO refinement stage contributed to stronger code generation performance, especially on execution-oriented and reasoning-heavy tasks.

---

## Quick Start

### 1. Run SFT training
```bash
cd SFT
python scripts/train.py --config configs/training_config.yaml
```

### 2. Run GRPO training
```bash
cd ../GRPO
python scripts/train.py
```

### 3. Evaluate outputs
Use the provided evaluation and inference scripts in each subproject to benchmark model quality and inspect generated results.

---

## Notes

- The training workflow was designed to be practical and modular.
- The SFT stage provides a strong supervised baseline.
- The GRPO stage further improves generation quality through reward-driven optimization.
- The implementation is suitable for experimentation in Colab and other GPU environments with limited resources.

---

## Summary

This repository demonstrates a complete and professional approach to fine-tuning language models for Python code generation. By combining SFT and GRPO, the project moves from supervised adaptation to reward-guided optimization, resulting in a more capable and reliable code generation model.
