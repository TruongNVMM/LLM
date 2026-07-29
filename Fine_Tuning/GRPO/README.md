# GRPO Fine-Tuning — Evol-Instruct-Code-80k-v1

Fine-tune một LLM để sinh code Python chất lượng cao bằng thuật toán **GRPO (Group Relative Policy Optimization)** kết hợp với **Unsloth** và **TRL**.

## Tổng quan

Dự án này thực hiện Reinforcement Learning from Human Feedback (RLHF) cho tác vụ code generation, sử dụng hệ thống reward tự động đánh giá chất lượng code qua nhiều tiêu chí.

### Đặc điểm nổi bật
- **GRPO thay vì PPO**: Không cần Critic model riêng, đơn giản và ổn định hơn
- **5-dimensional Reward**: Đánh giá format, syntax, executability, placeholder và length
- **Sandbox Execution**: Chạy code thực tế trong subprocess để kiểm tra
- **Early Stopping thông minh**: Dừng training khi chất lượng không còn cải thiện
- **Tối ưu cho Colab T4**: Unsloth + 4-bit QLoRA giúp chạy trên GPU miễn phí

---

## Cấu trúc dự án

```
GRPO/
├── src/
│   ├── model/
│   │   └── loader.py              # Load SFT model + áp dụng LoRA
│   ├── data/
│   │   └── dataset.py             # Load & format dataset Evol-Instruct-Code-80k
│   ├── rewards/
│   │   ├── code_rewards.py        # 5 reward functions thành phần
│   │   └── combined.py            # compute_reward() tổng hợp
│   ├── callbacks/
│   │   └── early_stopping.py      # CodeQualityCallback + Early Stopping
│   └── training/
│       └── trainer.py             # GRPOConfig + GRPOTrainer
├── configs/
│   └── grpo_config.yaml           # Toàn bộ hyperparameters tập trung
├── scripts/
│   ├── train.py                   # Entry point: chạy pipeline
│   └── export_model.py            # Merge LoRA → full float16 model
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
# 1. Cài dependencies
pip install --no-deps unsloth
pip install -r requirements.txt

# 2. Cài unsloth_zoo (cần thiết cho Unsloth)
pip install unsloth_zoo
```

> **Lưu ý**: Dự án thiết kế để chạy trên Google Colab với GPU T4. Để chạy local, cần GPU NVIDIA với ít nhất 16GB VRAM.

---

## Cách sử dụng

### Bước 1: Cấu hình

Mở file `configs/grpo_config.yaml` và chỉnh sửa đường dẫn model SFT của bạn:

```yaml
model:
  # Đường dẫn tới SFT model đã fine-tune trước đó
  sft_model_dir: "/content/drive/MyDrive/.../model"
```

### Bước 2: Training

```bash
# Chạy với config mặc định
python scripts/train.py

# Chỉ định config và override model path
python scripts/train.py \
  --config configs/grpo_config.yaml \
  --sft_model_dir /path/to/your/sft_model
```

### Bước 3: Export model

```bash
# Merge LoRA và lưu full model
python scripts/export_model.py

# Chỉ định thư mục lưu
python scripts/export_model.py \
  --best_model_dir best_grpo_checkpoint \
  --output_dir exported_model \
  --save_method merged_16bit
```

---

## Hệ thống Reward

| Reward Function | Trọng số | Mô tả |
|---|---|---|
| `reward_code_executable` | 40% | Code chạy thành công trong subprocess |
| `reward_syntax_valid` | 25% | Python AST parse không lỗi |
| `reward_no_placeholder` | 20% | Không có `pass`, `TODO`, `NotImplementedError` |
| `reward_code_format` | 10% | Có markdown ` ```python ... ``` ` |
| `reward_length_quality` | 5% | Độ dài code hợp lý (3–50 dòng) |

---

## Hyperparameters quan trọng

| Parameter | Giá trị | Giải thích |
|---|---|---|
| `num_generations` | 4 | Số responses/prompt để GRPO so sánh |
| `learning_rate` | 5e-6 | Nhỏ hơn SFT để tránh catastrophic forgetting |
| `lora_dropout` | 0.0 | GRPO thường không cần dropout |
| `eval_steps` | 100 | Tần suất validation |
| `patience` | 3 | Số lần không cải thiện trước khi dừng |

---

## Metrics theo dõi

Trong quá trình training, callback sẽ log các metrics sau mỗi `eval_steps` bước:

- `val/pass_rate`: Tỉ lệ code chạy thành công
- `val/syntax_rate`: Tỉ lệ code đúng syntax
- `val/format_rate`: Tỉ lệ code đúng định dạng
- `val/mean_reward`: Điểm reward trung bình (metric chính để early stopping)
