# HUST Sổ tay sinh viên RAG Pipeline

Hệ thống crawl, làm sạch và lưu trữ dữ liệu từ trang Sổ tay sinh viên HUST vào PostgreSQL cho RAG (Retrieval-Augmented Generation).

## Kiến trúc Pipeline

```
API HUST (GetWebTitleLst / GetWebTitleInfo)
    ↓
normalize_article()     → làm sạch HTML, extract text, links
    ↓ (tuỳ chọn)
enrich_local_files()    → đọc file .doc/.docx/.pdf từ data_fetch/
    ↓
to_rag_document()       → build RAG document format
    ↓
build_chunks()          → cắt đoạn thành chunks (CHUNK_SIZE ~1500 ký tự)
    ↓
PostgreSQL              → lưu vào bảng articles + rag_chunks
```

## Cấu trúc thư mục

```
src/data_processing/
├── pipeline.py              ← Pipeline chính (Crawl → Clean → Chunk → PostgreSQL)
├── crawl_hust_sotay.py      ← Logic crawl + làm sạch (tái sử dụng bởi pipeline.py)
├── chunking_pipeline.py     ← Logic chunking (tái sử dụng bởi pipeline.py)
├── integrate_data_fetch.py  ← Script cũ (đã được tích hợp vào pipeline.py)
└── db/
    ├── __init__.py
    ├── connection.py        ← Kết nối PostgreSQL + upsert helpers
    └── schema.sql           ← Schema: articles, rag_chunks, pipeline_runs
```

## Cài đặt

### 1. Cài dependencies

```powershell
pip install psycopg2-binary pypdf python-docx markdownify
```

### 2. Cấu hình PostgreSQL

Sao chép `.env.example` thành `.env` và điền thông tin:

```powershell
Copy-Item .env.example .env
```

Nội dung `.env`:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=hust_rag
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here
```

### 3. Tạo database

```sql
CREATE DATABASE hust_rag;
```

## Chạy Pipeline

### Crawl đầy đủ (khuyến nghị)

```powershell
cd src/data_processing
python pipeline.py
```

### Crawl nhanh (không gọi GetWebTitleInfo từng bài)

```powershell
python pipeline.py --no-detail
```

### Crawl + tải file đính kèm online (Google Doc/Drive/PDF)

```powershell
python pipeline.py --fetch-attachments
```

### Crawl + enrich với file cục bộ từ `data_fetch/`

```powershell
python pipeline.py --enrich-local
python pipeline.py --enrich-local --data-fetch-dir ./data_fetch
```

### Chỉ khởi tạo schema (không crawl)

```powershell
python pipeline.py --init-schema-only
```

### Verbose (in tiến độ chi tiết)

```powershell
python pipeline.py --verbose
```

## Schema PostgreSQL

| Bảng | Mô tả |
|------|-------|
| `articles` | Bài viết đã làm sạch (1 dòng = 1 bài, upsert theo `doc_id`) |
| `rag_chunks` | Chunks sẵn sàng cho vector embedding (upsert theo `id`) |
| `pipeline_runs` | Log lịch sử mỗi lần chạy pipeline |

### Kiểm tra dữ liệu sau khi chạy

```sql
-- Số bài viết theo chuyên mục
SELECT category_name, COUNT(*) FROM articles GROUP BY category_name ORDER BY COUNT(*) DESC;

-- Số chunks
SELECT chunk_type, COUNT(*) FROM rag_chunks GROUP BY chunk_type;

-- Pipeline runs gần nhất
SELECT id, started_at, total_articles, total_chunks, status
FROM pipeline_runs ORDER BY id DESC LIMIT 5;

-- Mẫu chunk
SELECT id, chunk_type, left(text, 200) FROM rag_chunks LIMIT 3;
```

## Script cũ (giữ lại cho tương thích)

- [`crawl_hust_sotay.py`](src/data_processing/crawl_hust_sotay.py): Crawl và lưu ra file JSONL (standalone)
- [`chunking_pipeline.py`](src/data_processing/chunking_pipeline.py): Chunk từ file JSONL (standalone)
- [`integrate_data_fetch.py`](src/data_processing/integrate_data_fetch.py): Enrich JSONL với file cục bộ (standalone)

Các script trên vẫn hoạt động độc lập. `pipeline.py` tái sử dụng logic từ chúng và thêm lưu trực tiếp vào PostgreSQL.
