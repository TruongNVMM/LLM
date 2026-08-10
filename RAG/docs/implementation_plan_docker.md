# Kế hoạch đóng gói Docker (Dockerization)

Việc đóng gói hệ thống RAG này bằng Docker yêu cầu thiết lập môi trường Python chuẩn, quản lý thư viện nặng như `torch`, và đặc biệt là **bảo toàn dữ liệu** (model weights, vector db) thông qua Docker Volumes.

## 1. Cấu trúc file cần thêm

### [NEW] `Dockerfile`
Sử dụng base image `python:3.11-slim` để tối ưu kích thước.
- Cài đặt các gói build cơ bản (nếu cần cho psycopg2).
- Cài đặt `requirements.txt`.
- Copy mã nguồn (`src/`) vào thư mục `/app`.
- Thiết lập biến môi trường `HF_HOME=/app/data/hf_cache` để HuggingFace lưu model vào thư mục dùng chung (mount volume), tránh tải lại model 2.2GB mỗi lần khởi động.

### [NEW] `docker-compose.yml`
Quản lý container của ứng dụng (và tùy chọn thêm database).
- **Service `rag-app`**:
  - Build từ `Dockerfile`.
  - Mount thư mục `./data:/app/data`: Để lưu chung Qdrant DB và HuggingFace model cache.
  - Mount file `.env`: Đọc cấu hình kết nối PostgreSQL.
  - Hỗ trợ GPU (NVIDIA) thông qua `deploy: resources: reservations: devices` (đối với Windows WSL2 hoặc Linux có cài NVIDIA Container Toolkit).
- *(Tùy chọn)* **Service `postgres`**: Nếu bạn muốn chạy Postgres cục bộ thay vì dùng IP `100.103.102.46`.

### [NEW] `.dockerignore`
Ngăn Docker copy các file không cần thiết vào image, giúp tăng tốc quá trình build:
- Thư mục `data/` (sẽ được mount qua volume)
- Các thư mục ảo (`.venv`, `__pycache__`)
- File nhạy cảm ngoài `.env`

---

## 2. Chiến lược thực thi (Entrypoint)
Do hiện tại dự án đang là tập hợp các script độc lập (pipeline crawl, pipeline embed, test retriever) chứ chưa có Web API (như FastAPI/Flask), Docker container sẽ được thiết lập để chạy dưới dạng **CLI Tool** hoặc **Job**.

Có 2 cách dùng sau khi có Docker:
1. **Chạy Crawl**:
   `docker compose run --rm rag-app python -m src.data_processing.pipeline`
2. **Chạy Embedding**:
   `docker compose run --rm rag-app python -m src.embedding.embedding_pipeline`
3. **Thử Retriever**:
   `docker compose run --rm rag-app python src/embedding/retriever.py "đăng ký học lại"`

---

## Open Questions

> [!IMPORTANT]
> - Bạn có muốn tôi thêm một service **PostgreSQL** luôn vào trong `docker-compose.yml` để tạo thành một cụm hoàn chỉnh (tự host DB trên máy này), hay vẫn giữ nguyên việc kết nối ra DB bên ngoài (`100.103.102.46`)?
> - Trong tương lai bạn có dự định làm giao diện chat hoặc API (bằng FastAPI/Gradio/Streamlit) không? Nếu có, tôi có thể chuẩn bị sẵn cấu hình expose port trong file docker-compose.
