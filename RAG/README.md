# HUST Sổ tay sinh viên RAG crawler

Trang `https://sv-ctt.hust.edu.vn/#/so-tay-sv` là Vue SPA, nên HTML ban đầu không chứa dữ liệu sổ tay. Frontend gọi API:

- `https://ctsv.hust.edu.vn/api-t/HWAdmin/GetWebTitleLst`
- `https://ctsv.hust.edu.vn/api-t/HWAdmin/GetWebTitleInfo`

Script `crawl_hust_sotay.py` dùng các endpoint này, lọc đúng `TypeDoc` của Sổ tay SV, parse `Description` HTML thành text, giữ link trong metadata, rồi xuất dữ liệu sạch cho RAG.

## Chạy crawler

```powershell
python .\crawl_hust_sotay.py
```

Output mặc định:

- `data/hust_sotay/raw/web_title_list.json`: response gốc từ API danh sách.
- `data/hust_sotay/processed/sotay_articles.json`: bài viết đã chuẩn hóa, gồm HTML, text, links.
- `data/hust_sotay/processed/sotay_articles.jsonl`: mỗi dòng là một bài viết.
- `data/hust_sotay/processed/rag_documents.jsonl`: format gọn để đưa vào pipeline chunking/embedding.
- `data/hust_sotay/summary.json`: thống kê crawl.

Nếu muốn chỉ dùng dữ liệu từ API danh sách, không gọi chi tiết từng bài:

```powershell
python .\crawl_hust_sotay.py --no-detail
```

## Gợi ý bước tiếp theo cho RAG

Với `rag_documents.jsonl`, nên chunk theo từng bài hoặc từng heading trong `text`. Các link như `tại đây`, file `.docx`, `.pdf`, Google Drive... đã được giữ trong `metadata.links`; nếu link là file mẫu đơn quan trọng, hãy tải và parse file đó thành document con có `parent_doc_id`.
