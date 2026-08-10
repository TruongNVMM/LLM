# Kế hoạch sửa đổi hàm `build_chunks()`

Mục tiêu: Độc lập hóa việc phân rã (chunking) phần nội dung bài viết và phần file đính kèm bằng cách sử dụng trực tiếp cấu trúc `attachments` đã chuẩn hóa, loại bỏ hoàn toàn việc dùng Regex để parse `[Nội dung tài liệu đính kèm...]` từ `rag_text`.

## Proposed Changes

### `src/data_processing/chunking_pipeline.py`
- **[MODIFY]**: Trong hàm `build_chunks(doc)`:
  - Thay vì gọi `extract_attachments(text)` bằng Regex, chúng ta sẽ gán trực tiếp: `body = doc['text']`.
  - Lặp qua mảng `doc.get("attachments", [])` để lấy nội dung các file đính kèm.
  - Sử dụng `_split_attachment(att["content"])` để chia nhỏ từng file đính kèm thành các chunk.
  - Xây dựng metadata cho attachment chunk đúng như định dạng yêu cầu:
    ```python
    {
        **base_meta,
        "chunk_type": "attachment",
        "is_attachment": True,
        "attachment_id": att.get("attachment_id"),
        "attachment_name": att.get("name"),
        "attachment_url": att.get("url"),
        "local_filename": att.get("local_filename"),
        "source_link_index": att.get("link_index"),
        "fetch_status": att.get("fetch_status"),
        "content_hash": att.get("content_hash"),
        "attachment_chunk_index": ... # Thứ tự chunk của riêng attachment này
    }
    ```
- **[DELETE]**: Xóa bỏ các hàm và biến liên quan đến Regex như `ATTACH_PATTERN`, `extract_attachments()`.

### `src/data_processing/crawl_hust_sotay.py` & `src/data_processing/pipeline.py`
- **[MODIFY/DELETE]**: Vì `build_chunks` không còn lọc `[Nội dung tài liệu đính kèm...]` ra khỏi body nữa, nếu ta cứ để khối lượng text khổng lồ này trong `rag_text`, nó sẽ bị chia thành các chunk `body` (gây trùng lặp với chunk `attachment`). Do đó:
  - Trong `crawl_hust_sotay.py`, xóa bỏ `build_attachment_section()` và không append kết quả này vào `rag_text` nữa.
  - Trong `pipeline.py` (hàm `enrich_article_with_local_files`), không append text của file đính kèm vào cuối `rag_text` nữa. (Các thẻ link markdown mô tả vẫn sẽ được giữ lại bình thường).

## User Review Required
Việc xóa bỏ khối text `[Nội dung tài liệu đính kèm...]` khỏi `rag_text` là bắt buộc để tránh tạo ra chunk trùng lặp (vì attachment giờ đây đã có metadata riêng và được xử lý độc lập). Các đường link (như `[Biểu mẫu đính kèm: tên file](url)`) vẫn sẽ tồn tại trong text bài viết. Bạn có đồng ý với việc loại bỏ khối nội dung này khỏi `rag_text` không?

## Verification Plan
1. Chạy `pipeline.py` và sau đó chạy lại script test đối với `chunking_pipeline.py`.
2. Kiểm tra file output (hoặc log) xem `metadata` của các chunk đính kèm có chứa đầy đủ `attachment_id`, `source_link_index`... hay không.
3. Đảm bảo body chunk không bị dính nội dung dài dằng dặc của file đính kèm.
