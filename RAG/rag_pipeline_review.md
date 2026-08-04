# Đánh giá Pipeline Thu thập Dữ liệu HUST Sổ tay Sinh viên

## Tổng quan

Pipeline hiện tại đã hoàn thiện tốt phần xương sống: crawl API → HTML → plain text → RAG document. Các thiếu sót chủ yếu nằm ở việc **nội dung bên trong các link chưa được thu thập**, và một số khoảng trống về ngữ cảnh mà LLM cần để trả lời tốt hơn.

---

## ✅ Điểm mạnh đã làm tốt

| Hạng mục | Nhận xét |
|---|---|
| Crawl API thay vì parse HTML trang | Đúng hướng, tránh được vấn đề SPA/JS rendering |
| Lọc chỉ lấy bài `TypeDoc` thuộc Sổ tay SV | Loại bỏ được nhiễu (banner, thông báo khác) |
| HTML → plain text có xử lý block tags | Giữ được cấu trúc đoạn văn, danh sách |
| Phát hiện và phân loại link (`word_online`, `email`, `google_doc`, `pdf`...) | Rất hữu ích cho LLM tham chiếu |
| `context` 180 ký tự quanh link | Giúp LLM hiểu link dùng để làm gì |
| Header ngữ cảnh trong `rag_text` (Tiêu đề/Nhóm/Loại tài liệu) | Cải thiện embedding, giúp retrieval chính xác hơn |
| `metadata.category_code` có thể dùng làm filter | Cho phép hybrid search (semantic + metadata filter) |

---

## ❌ Vấn đề cần cải thiện

### 1. Nội dung "Tại đây" → File biểu mẫu CHƯA được thu thập

**Hiện trạng:** Có **22 link Word Online** (SharePoint), **13 Google Doc**, **12 PDF** và **2 file `.doc`** thuần túy. Tất cả đều chỉ được lưu dưới dạng URL, không có nội dung.

**Ảnh hưởng:** Khi sinh viên hỏi *"Mẫu đơn xin rút học phần gồm những gì?"* hoặc *"Điền gì vào mẫu đơn hoãn thi?"* — LLM **không có thông tin gì** về nội dung thực tế của biểu mẫu.

**Hướng xử lý theo độ khó:**
- `word_online` / `google_doc`: Cần xác thực (có thể public) → Thử fetch, parse nội dung văn bản
- `pdf`: Dùng `pdfminer` / `pypdf` để extract text
- `doc` thuần: Dùng `python-docx` nếu có thể download

> ⚠️ Cần kiểm tra từng link xem có public access không trước khi fetch hàng loạt.

---

### 2. Link đến bài viết Sổ tay khác CHƯA được nội tuyến hóa (inline)

**Hiện trạng:** Có **8 link** trỏ đến bài viết sổ tay khác (ví dụ: nhiều bài trỏ về `doc_id=69` để hướng dẫn quy trình nộp đơn). Những link này chỉ được lưu URL, không được embed nội dung.

**Ví dụ cụ thể:**
```
[hust_sotay_77] "TẠI ĐÂY" → /so-tay-sv/69/...  (Hướng dẫn gửi câu hỏi tới Phòng ĐT)
[hust_sotay_68] "Xem tại đây." → /so-tay-sv/69/...  (3 lần)
[hust_sotay_127] "tại đây" → /so-tay-sv/68/...  (Các quy định và biểu mẫu)
```

**Hướng xử lý:** Sau khi crawl xong, resolve cross-references: nếu một bài link đến `doc_id=X`, thêm một trường `referenced_docs: [X]` và append tóm tắt của doc X vào `rag_text`.

---

### 3. Link đến form đặt câu hỏi (ctsv.hust.edu.vn/#/viet-giay) thiếu ngữ nghĩa

**Hiện trạng:** Có **27 link nội bộ** đến các form của CTSV (`#/viet-giay`, `#/xin-cap-giay`, `#/danh-sach-tuyen-dung`...) nhưng LLM không biết form đó dùng để làm gì.

**Hướng xử lý:** Thêm mapping tĩnh (hard-coded) của các URL CTSV phổ biến → mô tả ngắn, hoặc bổ sung description vào trường `context` của link.

---

### 4. Không có thông tin về **ngày có hiệu lực** / **kỳ học áp dụng**

**Hiện trạng:** Chỉ có `time_create` (ngày tạo bài viết). Nhiều quy định thay đổi theo từng năm học.

**Hướng xử lý:** Thêm trường `academic_year` được trích xuất từ tiêu đề hoặc nội dung (regex tìm `2025-2026`, `K71`, `HK1`...).

---

### 5. Chunking: Bài dài như `doc_id=69` (~42 KB text) quá lớn cho embedding

**Hiện trạng:** Mỗi bài viết là 1 document duy nhất. Bài `69` (Hướng dẫn Ban Đào tạo) có bảng 42 hạng mục công việc → token rất lớn, vượt context window của nhiều embedding model.

**Hướng xử lý:** Thêm bước chunking thông minh:
- Cắt theo heading (`I.`, `II.`, `Bước 1`, `Bước 2`...)
- Cắt theo hàng của bảng (mỗi hạng mục công việc = 1 chunk)
- Mỗi chunk kế thừa đầy đủ metadata của bài gốc

---

### 6. `rag_text` chứa cả `link_text` dạng `| anchor=... | type=... | url=...` — khó đọc

**Hiện trạng:** Phần cuối `rag_text` có dạng:
```
- Liên kết trong bài '...': ... | anchor='tại đây' | type=word_online | url=https://...
```
Định dạng này khó cho LLM đọc và parse ra URL chính xác.

**Hướng xử lý:** Định dạng lại dạng Markdown tự nhiên:
```markdown
**Liên kết liên quan:**
- [Mẫu đơn rút học phần](https://...) *(biểu mẫu Word)*
- Liên hệ: dat.nguyenquoc@hust.edu.vn
```

---

### 7. Thiếu trường `keywords` / `tags` để tăng precision khi retrieval

**Hiện trạng:** Không có keywords. Retrieval hoàn toàn dựa vào semantic similarity của embedding.

**Hướng xử lý:** Thêm trường `keywords` bằng cách:
- Trích xuất email (`@hust.edu.vn`) → tên người phụ trách
- Trích xuất các chuỗi trong ngoặc vuông `[Học phí]`, `[Chuyển ngành]`... → từ khóa chức năng
- Trích xuất mã sinh viên, khoa, trường liên quan

---

## 📋 Bảng ưu tiên cải thiện

| # | Cải thiện | Tác động RAG | Độ khó |
|---|---|---|---|
| 1 | **Fetch nội dung file biểu mẫu** (PDF/Word public) | 🔴 Rất cao | Trung bình |
| 2 | **Chunking thông minh** cho bài dài | 🔴 Rất cao | Trung bình |
| 3 | **Inline nội dung link → bài sổ tay khác** | 🟠 Cao | Thấp |
| 4 | **Định dạng lại link_text** sang Markdown | 🟠 Cao | Thấp |
| 5 | **Mapping URL CTSV** → mô tả ngữ nghĩa | 🟡 Trung bình | Thấp |
| 6 | **Trích xuất keywords** (email, [hạng mục]) | 🟡 Trung bình | Thấp |
| 7 | **Thêm academic_year** từ nội dung | 🟡 Trung bình | Thấp |

---

## 📊 Thống kê link hiện tại (29 documents)

| Loại link | Số lượng | Đã có nội dung? |
|---|---|---|
| `html_or_external` | 96 | ✅ (URL là đủ) |
| `email` | 32 | ✅ (email là đủ) |
| `word_online` (SharePoint) | 22 | ❌ **Chưa** |
| `google_drive` | 14 | ❌ **Chưa** |
| `google_doc` | 13 | ❌ **Chưa** |
| `pdf` | 12 | ❌ **Chưa** |
| `form` | 3 | ✅ (URL là đủ) |
| `doc` (file .doc) | 2 | ❌ **Chưa** |
| `powerpoint_online` | 1 | ❌ **Chưa** |

> **Kết luận:** Có tổng **64/195 links** (33%) là tài liệu đính kèm quan trọng (biểu mẫu, hướng dẫn chi tiết) chưa được thu thập nội dung.
