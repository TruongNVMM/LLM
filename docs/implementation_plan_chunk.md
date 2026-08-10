# Kế hoạch sửa đổi hàm `to_rag_document()`

Mục tiêu: Đưa danh sách `attachments` ra ngoài cùng cấp với `text`, đồng thời làm gọn `metadata` của bài viết (bỏ `links` và `attachments` ra khỏi metadata) để tránh lặp lại dữ liệu thừa trong từng chunk.

## Proposed Changes

### `src/data_processing/crawl_hust_sotay.py`
- **[MODIFY]**: Sửa hàm `to_rag_document(article)`.
  - Thêm trường `"attachments": article.get("attachments", [])` vào cùng cấp với `"text"`.
  - Trong block `"metadata"`, xóa đi dòng `"links": article["links"]` và `"attachments": ...` (do bước trước chúng ta đã vô tình bỏ vào metadata).
  - Giữ lại các trường cơ bản trong metadata: `source`, `source_url`, `doc_id`, `type_doc`, `category_code`, `category_name`, `time_create`, `emails`, `tags`.

Cấu trúc trả về sẽ có dạng:
```python
{
    "id": f"hust_sotay_{article['doc_id']}",
    "doc_type": "hust_student_handbook_article",
    "title": article["title"],
    "text": article["rag_text"],
    "attachments": article.get("attachments", []),
    "metadata": {
        "source": "HUST Sổ tay sinh viên",
        "source_url": article["source_url"],
        "doc_id": article["doc_id"],
        "type_doc": article["type_doc"],
        "category_code": article["category_code"],
        "category_name": article["category_name"],
        "time_create": article["time_create"],
        "emails": article.get("keywords", {}).get("emails", []),
        "tags": article.get("keywords", {}).get("tags", []),
    },
}
```

## User Review Required
Việc bỏ `links` ra khỏi metadata đồng nghĩa với việc các chunk sinh ra sau này sẽ không chứa mảng links khổng lồ. Tuy nhiên, nếu bạn vẫn muốn truy xuất link liên quan từ từng chunk, chúng ta sẽ cần dựa vào `article["links"]` ở database gốc. Bạn có đồng ý với thiết kế này không?

## Verification Plan
1. Kiểm tra lại logic sinh file RAG JSON/Dict bằng cách chạy một test nhỏ hoặc khởi chạy lại pipeline.
2. Kiểm chứng dump dict trả về từ `to_rag_document` đã đúng schema yêu cầu.
