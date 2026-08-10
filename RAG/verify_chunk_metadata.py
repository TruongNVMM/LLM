import json
from src.data_processing.chunking_pipeline import build_chunks

def main():
    dummy_doc = {
        "id": "test_doc_1",
        "doc_type": "hust_student_handbook_article",
        "title": "Hướng dẫn đăng ký học tập",
        "text": "Đây là phần nội dung chính của bài viết. Sinh viên cần đăng ký đúng hạn.",
        "attachments": [
            {
                "attachment_id": "att_001",
                "name": "Mẫu đơn xin đăng ký muộn",
                "url": "http://example.com/mau-don.docx",
                "local_filename": "mau-don.docx",
                "link_index": 2,
                "fetch_status": "ok",
                "content_hash": "abcdef123456",
                "content": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\n\nĐƠN XIN ĐĂNG KÝ MUỘN..."
            }
        ],
        "metadata": {
            "source": "HUST",
            "source_url": "http://example.com/doc1",
            "doc_id": "doc1",
            "type_doc": "Hướng dẫn",
            "category_code": "HD",
            "category_name": "Hướng dẫn chung",
            "time_create": "2023-10-01",
            "emails": ["ctsv@hust.edu.vn"],
            "tags": ["đăng ký học"]
        }
    }
    
    chunks = build_chunks(dummy_doc)
    
    print("Tổng số chunks:", len(chunks))
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {chunk['id']}")
        print(f"Loại: {chunk.get('chunk_type', 'N/A')}")
        print(f"Is Attachment: {chunk.get('is_attachment', False)}")
        print("Metadata Keys:", list(chunk['metadata'].keys()))
        if 'links' in chunk['metadata']:
            print("CẢNH BÁO: 'links' tồn tại trong metadata!")
        else:
            print("OK: Không có 'links' trong metadata.")
        print("Metadata Dump:")
        print(json.dumps(chunk['metadata'], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
