# SFT Fine-tuning: Qwen2.5-Coder-3B với Evol-Instruct-Code-80k

Fine-tuning `unsloth/Qwen2.5-Coder-3B-Instruct` bằng SFT sử dụng LoRA qua Unsloth.

---

## Cấu trúc thư mục

```
sft_project/
├── configs/
│   └── training_config.yaml    # Toàn bộ hyperparameter
├── src/
│   ├── data/
│   │   └── dataset.py          # Load, split, format dataset
│   ├── models/
│   │   └── model.py            # Load model, apply LoRA, save
│   ├── training/
│   │   └── trainer.py          # Build SFTTrainer
│   └── utils/
│       ├── config.py           # Typed dataclasses cho YAML config
│       └── logging_utils.py    # Logger dùng chung
├── scripts/
│   ├── train.py                # Entry point training
│   ├── inference.py            # Chạy inference + so sánh output
│   └── evaluate.py             # CodeBLEU / Pass@k evaluation
└── requirements.txt
```

---

## Cài đặt

```bash
# Unsloth (theo hướng dẫn chính thức)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Các thư viện còn lại
pip install -r requirements.txt
```

---

## Cách dùng

### 1. Cấu hình

Chỉnh sửa `configs/training_config.yaml`. Các thông số quan trọng nhất:

```yaml
training:
  num_train_epochs: 1   # Đủ để model học, thay vì max_steps=200 (chỉ 2.2% data)

lora:
  lora_alpha: 64        # = 2 * r=32, scaling factor tốt hơn

training:
  metric_for_best_model: "eval_loss"   # Metric chính xác hơn token_accuracy
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
# CodeBLEU trên 100 mẫu test
python scripts/evaluate.py --benchmark codebleu --num_samples 100

# Pass@1 trên HumanEval (cần pip install human-eval)
python scripts/evaluate.py --benchmark humaneval --pass_k 1
```

---

## Các fix quan trọng so với notebook gốc

| # | Vấn đề | Fix |
|---|--------|-----|
| 1 | `max_steps=200` → chỉ train 2.2% data | `num_train_epochs=1` |
| 2 | `dataset_test` bị `remove_columns` → `KeyError` ở inference | Giữ `raw_test` chưa format |
| 3 | `train_on_responses_only` thiếu `\n` → mask bị lệch | Thêm `\n` vào `instruction_part` và `response_part` |
| 4 | `lora_alpha=32` (= r) → scaling thấp | `lora_alpha=64` (= 2×r) |
| 5 | System prompt thiếu ở inference | Thêm vào `messages` trước khi generate |
| 6 | `token_accuracy` không phù hợp cho code | `eval_loss` + `CodeBLEU` / `Pass@k` |
| 7 | `peft_config` truyền vào `SFTTrainer` dư thừa | Bỏ hoàn toàn |

---

## Lộ trình đánh giá model

| Giai đoạn | Metric | Dùng ở đâu |
|-----------|--------|------------|
| Trong training | `eval_loss` | `SFTTrainer` tự tính |
| Sau training (nhanh) | `CodeBLEU` | `scripts/evaluate.py --benchmark codebleu` |
| Sau training (chuẩn) | `Pass@1` trên HumanEval | `scripts/evaluate.py --benchmark humaneval` |
