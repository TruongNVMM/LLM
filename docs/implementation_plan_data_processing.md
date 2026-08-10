# Kế hoạch chuẩn hóa attachment trước khi chunk

Mục tiêu: Tạo một danh sách `attachments` chuẩn xác trong mô hình dữ liệu `article` thay vì phụ thuộc hoàn toàn vào việc parse text từ `rag_text` để lấy file đính kèm. Điều này giúp quản lý dữ liệu đính kèm minh bạch hơn và hỗ trợ cho quá trình chunking (cắt đoạn) ổn định, chính xác.

## User Review Required

Vui lòng xem lại danh sách các trường trong object `attachment`. Nếu bạn cần thêm bất kỳ trường metadata nào khác (ví dụ: `file_size`, `mime_type`...), hãy cho tôi biết trước khi bắt tay vào code.

## Proposed Changes

Chúng ta sẽ chỉnh sửa quy trình xử lý dữ liệu để sinh ra trường `attachments` ngay sau khi tải nội dung file đính kèm (hoặc từ internet hoặc từ thư mục local `data_fetch`).

### `src/data_processing/db/schema.sql`
- **[MODIFY]**: Thêm cột `attachments JSONB DEFAULT '[]'` vào bảng `articles` để có thể lưu trữ danh sách các file đính kèm chuẩn hóa xuống database PostgreSQL.

### `src/data_processing/db/connection.py`
- **[MODIFY]**: Cập nhật hàm `upsert_article` để bổ sung trường `attachments` vào câu lệnh INSERT và `ON CONFLICT DO UPDATE SET`.

### `src/data_processing/pipeline.py`
- **[MODIFY]**: Trong hàm `run_pipeline`, sau khi bài viết được `normalize_article` và `enrich_article_with_local_files` (nếu có), chúng ta sẽ thêm một bước **xây dựng danh sách `attachments`** cho `article`.
  - Duyệt qua `article["links"]` (có sử dụng `enumerate` để lấy `link_index`).
  - Lọc ra các link có `fetch_status == "ok"` và có `content` (nội dung text của tài liệu đính kèm).
  - Khởi tạo object attachment với cấu trúc:
    ```json
    {
      "attachment_id": "md5 hoặc sha256 của URL để định danh",
      "name": "Tên file hoặc anchor text",
      "url": "URL gốc",
      "local_filename": "Tên file local (nếu có)",
      "link_index": 6,
      "fetch_status": "ok",
      "content": "Nội dung text đã trích xuất",
      "content_hash": "sha256 của content để deduplicate"
    }
    ```
  - Gán danh sách này vào `article["attachments"]`.
- **[MODIFY]**: (Tuỳ chọn) Chỉnh sửa quá trình gọi `build_chunks` để truyền `attachments` qua RAG Document hoặc trực tiếp, làm tiền đề cho việc chunk attachment chuẩn xác ở bước sau.

### `src/data_processing/crawl_hust_sotay.py`
- **[MODIFY]**: Trong `to_rag_document`, bổ sung việc đưa `article["attachments"]` vào phần `metadata` của `rag_doc` để module chunking có thể truy xuất dễ dàng.

## Verification Plan

### Automated/Manual Tests
1. Chạy thử `pipeline.py` trên một bài Sổ tay có đính kèm file (ví dụ bằng cờ `--enrich-local`).
2. Kiểm tra log và dump dữ liệu của `article` xem trường `attachments` có cấu trúc đúng như yêu cầu hay không.
3. Kiểm tra trong PostgreSQL (bảng `articles`) xem cột `attachments` đã lưu mảng JSON chính xác chưa.
4. Xác nhận `rag_text` vẫn giữ lại link mô tả file đính kèm bình thường mà không bị ảnh hưởng.
