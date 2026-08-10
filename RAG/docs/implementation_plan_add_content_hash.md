# Kế hoạch sửa cơ chế re-embed (Cập nhật Chunk)

Mục tiêu (theo ảnh 5): Thêm trường `content_hash` để theo dõi sự thay đổi của nội dung chunk. Cập nhật câu lệnh `upsert_chunks` để tự động reset cờ `embedded_at = NULL` khi chunk bị thay đổi nội dung.

## Tại sao phải làm điều này? (Giải thích nguyên nhân)

Hiện tại, bảng `rag_chunks` lưu dữ liệu dạng text, còn vector embedding được tạo ra và lưu bên Qdrant. Để biết chunk nào cần được đem đi tạo vector, pipeline embedding sẽ quét những dòng có `embedded_at IS NULL`.
Trong code `upsert_chunks` hiện hành:
```sql
ON CONFLICT (id) DO UPDATE SET
    text = EXCLUDED.text,
    metadata = EXCLUDED.metadata,
    created_at = NOW();
```
Khi nội dung bài viết thay đổi (sửa lỗi chính tả, đổi cách chia đoạn), chunk sẽ được cập nhật (update) trong PostgreSQL. **Tác hại nghiêm trọng** là trường `embedded_at` không bị đụng tới và vẫn giữ nguyên giá trị thời gian cũ. 
Hậu quả: Embedding pipeline sẽ tưởng lầm chunk này "đã được embed rồi" (vì `embedded_at` khác NULL) và bỏ qua nó. Điều này dẫn đến tình trạng **Out of Sync (lệch pha)**: PostgreSQL chứa nội dung mới, nhưng Qdrant vẫn chứa vector của nội dung cũ, làm hỏng kết quả tìm kiếm RAG.

Việc thêm `content_hash` và lệnh reset `embedded_at = NULL` sẽ hoạt động như một trigger hoàn hảo: hễ nội dung text hoặc metadata thay đổi (làm hash thay đổi), cờ embed lập tức bị reset, buộc pipeline embedding phải chạy lại cho chunk đó.

## Proposed Changes

### 1. `src/data_processing/db/schema.sql`
- **[MODIFY]**: Thêm cột `content_hash TEXT` vào định nghĩa bảng `rag_chunks`.

### 2. `src/data_processing/chunking_pipeline.py`
- **[MODIFY]**: Trong hàm `build_chunks()`, trước khi append chunk vào `chunks_out`, chúng ta tính toán mã hash cho chunk đó:
  ```python
  import hashlib
  
  # Tạo chuỗi đại diện (kết hợp text và những metadata quan trọng)
  hash_input = chunk_text + str(chunk_metadata.get('attachment_name', ''))
  content_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
  ```
  Thêm trường `content_hash` vào dict trả về của từng chunk. (Lưu ý: Đối với chunk đính kèm, ta có thể dùng luôn mã hash đã có từ bước tải file, nhưng tính lại hash chung cho toàn chunk vẫn an toàn hơn).

### 3. `src/data_processing/db/connection.py`
- **[MODIFY]**: Sửa đổi hàm `upsert_chunks()`:
  - Thêm `content_hash` vào lệnh `INSERT INTO`.
  - Trong mệnh đề `ON CONFLICT DO UPDATE SET`, thêm logic reset cờ:
    ```sql
    content_hash = EXCLUDED.content_hash,
    embedded_at = CASE 
        WHEN rag_chunks.content_hash IS DISTINCT FROM EXCLUDED.content_hash THEN NULL 
        ELSE rag_chunks.embedded_at 
    END,
    ```

## User Review Required
Cơ chế hash này cực kỳ nhẹ và hiệu quả. Bạn có đồng ý với thiết kế tạo `content_hash` bằng MD5 trong Python (giúp giảm tải tính toán cho Database) và reset `embedded_at` theo logic `CASE WHEN` trong SQL như trên không? Nếu đồng ý, hãy nhấn **Proceed**!
