# Kế hoạch sửa lỗi Deduplication (Theo hình 7)

Mục tiêu: Loại bỏ việc sử dụng `hash()` mặc định của Python (vì nó thay đổi ngẫu nhiên giữa các lần chạy script do seed randomization). Đồng thời giải quyết vấn đề "drop" mất chunk đính kèm gây mất ngữ cảnh.

## Phân tích và Lựa chọn giải pháp

Hình 7 đưa ra 2 yêu cầu:
1. **Dùng `sha256` thay cho `hash()`**: Điều này hoàn toàn chính xác. Ở bước trước chúng ta đã dùng `md5`, mình sẽ nâng cấp lên `sha256` để chống đụng độ (collision) tuyệt đối cho `content_hash`.
2. **Cân nhắc không drop attachment duplicate hoàn toàn**: 
   - Hiện tại, code trong `pipeline.py` (dòng 505-515) và `chunking_pipeline.py` đang thẳng tay DROP (xóa) những chunk đính kèm nếu nội dung bị trùng. 
   - **Tác hại**: Giả sử "Đơn xin thôi học" được đính kèm ở Bài viết A và Bài viết B. Nếu ta drop chunk ở bài B, khi người dùng hỏi về bài B, RAG sẽ mò ra chunk của bài A (vì chỉ còn mỗi nó) và trích dẫn nhầm nguồn (cite sai bài viết).
   - **Đề xuất**: Mình đề xuất **XÓA BỎ HOÀN TOÀN** logic drop chunk đính kèm. Thay vì vất vả thiết kế cơ chế `duplicate_of` hay `source_refs` (đòi hỏi gom nhóm toàn bộ chunk và sửa cấu trúc database phức tạp), việc cứ để nguyên 2 chunk đính kèm tồn tại độc lập với 2 `parent_id` khác nhau là tốt nhất cho độ chính xác của RAG. Việc lưu dư vài text/vector trùng lặp tốn chi phí cực kỳ nhỏ, nhưng bù lại RAG sẽ trích dẫn đúng bài viết chứa ngữ cảnh.

## Proposed Changes

1. **`src/data_processing/chunking_pipeline.py`**:
   - **[MODIFY]**: Đổi hàm băm từ `hashlib.md5` sang `hashlib.sha256(full_text.encode('utf-8')).hexdigest()` trong hàm `build_chunks`.
   - **[DELETE]**: Xóa khối code "Deduplicate identical attachment content across documents" trong hàm `main()`.

2. **`src/data_processing/pipeline.py`**:
   - **[DELETE]**: Xóa bỏ hoàn toàn khối code "Deduplication cho attachment chunks" (khoảng dòng 505-515) đang dùng vòng lặp và `set()` để filter chunk. Tất cả chunks sinh ra từ `build_chunks()` đều sẽ được mang đi `upsert_chunks()`.

## User Review Required
Bạn có đồng ý với quan điểm "không drop bất kỳ chunk đính kèm nào để giữ tính toàn vẹn của Citation (trích dẫn nguồn)" không? Việc này giúp đơn giản hóa hệ thống và tăng độ chính xác của RAG mà không làm phức tạp hóa database bằng `source_refs`. 

Nếu bạn đồng ý, hãy nhấn **Proceed**!
