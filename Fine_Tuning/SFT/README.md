# SFT Fine-tuning: Qwen2.5-Coder-3B with Evol-Instruct-Code-80k

Fine-tuning `unsloth/Qwen2.5-Coder-3B-Instruct` using SFT with LoRA via Unsloth.

---

## Directory Structure

```
SFT/
├── .gitignore
├── configs/
│   └── training_config.yaml    # All hyperparameters
├── src/
│   ├── data/
│   │   └── dataset.py          # Load, split, format dataset
│   ├── models/
│   │   └── model.py            # Load model, apply LoRA, save
│   ├── training/
│   │   └── trainer.py          # Build SFTTrainer
│   └── utils/
│       ├── config.py           # Typed dataclasses for YAML config
│       └── logging_utils.py    # Shared logger
├── scripts/
│   ├── train.py                # Training entry point
│   ├── inference.py            # Run inference + compare outputs
│   └── evaluate.py             # CodeBLEU / Pass@k evaluation
├── notebook/
|   ├── Fine-Tuning_SFT_Evol-Instruct-Code.ipynb
├── README.md
└── requirements.txt
```

---

## Installation

```bash
# Unsloth (follow official documentation)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Remaining libraries
pip install -r requirements.txt
```

---

## Usage

### 1. Configuration

Edit `configs/training_config.yaml`. Most important parameters:

```yaml
training:
  num_train_epochs: 1   # Sufficient for model to learn, instead of max_steps=200 (only 2.2% of data)

lora:
  lora_alpha: 64        # = 2 * r=32, better scaling factor

training:
  metric_for_best_model: "eval_loss"   # More accurate metric than token_accuracy
  prediction_loss_only: true
```

### 2. Training

```bash
python scripts/train.py --config configs/training_config.yaml
```

### 3. Inference

```bash
python scripts/inference.py --config configs/training_config.yaml
python scripts/inference.py --config configs/training_config.yaml --num_samples 5
```

### 4. Evaluation

```bash
# CodeBLEU on 100 test samples
python scripts/evaluate.py --benchmark codebleu --num_samples 100

# Pass@1 on HumanEval (requires pip install human-eval)
python scripts/evaluate.py --benchmark humaneval --pass_k 1
```

---

## Benchmark Results

| Benchmark | Pass@1 |
|-----------|--------|
| HumanEval | 0.73 |
| HumanEval+ | 0.65 |

---
Due to limited resources and the lack of a powerful GPU such as an RTX 3090, this project had to be trained on Google Colab