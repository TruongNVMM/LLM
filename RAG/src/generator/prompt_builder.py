from typing import List, Any

class PromptBuilder:
    """
    Xây dựng prompt cho mô hình ngôn ngữ dựa trên kết quả RAG.
    """
    
    DEFAULT_SYSTEM_PROMPT = """Bạn là một trợ lý tư vấn học thuật nhiệt tình và chuyên nghiệp của Trường Đại học Bách Khoa Hà Nội (HUST).
Nhiệm vụ của bạn là giải đáp thắc mắc của sinh viên dựa trên các tài liệu tham khảo được cung cấp.

Quy tắc bắt buộc:
1. LUÔN LUÔN trả lời bằng tiếng Việt, ngôn từ thân thiện, dễ hiểu.
2. CHỈ sử dụng thông tin có trong phần [TÀI LIỆU THAM KHẢO]. Không tự bịa đặt hoặc suy diễn ngoài tài liệu.
3. Nếu tài liệu không chứa thông tin để trả lời câu hỏi, hãy nói rõ: "Xin lỗi, hiện tại mình không tìm thấy thông tin về vấn đề này trong sổ tay sinh viên."
4. Trình bày câu trả lời rõ ràng, dùng bullet points hoặc in đậm những ý chính nếu cần.
5. Cuối câu trả lời, hãy trích dẫn nguồn ngắn gọn (Ví dụ: Nguồn: Hướng dẫn đăng ký học tập).
6. Nếu tài liệu tham khảo chứa nội dung mẫu đơn/biểu mẫu (có cụm "CỘNG HÒA", "Kính gửi", "Họ và tên"...), hãy hiển thị toàn bộ nội dung mẫu đó trong câu trả lời (dưới dạng markdown/plain text), KHÔNG chỉ đưa link.
7. Nếu câu hỏi hỏi về một loại học bổng cụ thể (ví dụ "loại A"), hãy tổng hợp TẤT CẢ thông tin có trong tài liệu: mức tiền, điều kiện, tỷ lệ, cách xét...
8. Nếu trong tài liệu có link SharePoint/Google Drive, chỉ đưa link khi không có nội dung thực trong context. Ưu tiên hiển thị nội dung thực trước link."""

    def __init__(self, system_prompt: str = None, max_context_chars: int = 12000):
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.max_context_chars = max_context_chars

    def build_system(self) -> str:
        """Trả về system prompt."""
        return self.system_prompt

    def build_prompt(self, query: str, context_docs: List[Any]) -> str:
        """
        Ghép ngữ cảnh từ tài liệu và câu hỏi của người dùng thành prompt hoàn chỉnh.
        `context_docs` là danh sách các đối tượng RetrievalResult.
        """
        context_str = self._format_context(context_docs, query=query)
        
        prompt = f"""[TÀI LIỆU THAM KHẢO]
{context_str}
---
[CÂU HỎI CỦA SINH VIÊN]
{query}
"""
        return prompt

    def _format_context(self, docs: List[Any], query: str = "") -> str:
        """Định dạng các chunk tài liệu thành chuỗi văn bản."""
        if not docs:
            return "Không có tài liệu tham khảo nào."
            
        # Sắp xếp attachment lên đầu nếu hỏi về biểu mẫu
        q_lower = query.lower()
        if any(kw in q_lower for kw in ["mẫu", "đơn", "biểu mẫu"]):
            docs = sorted(docs, key=lambda d: 0 if d.metadata.get("chunk_type") == "attachment" else 1)

        parts = []
        current_len = 0
        
        # Sắp xếp lại docs nếu cần (ví dụ: theo thứ tự chunk_index)
        # Assuming they are already somewhat sorted by relevance or chunk_index from retriever
        
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "Không rõ tiêu đề")
            url = doc.metadata.get("source_url", "Không rõ URL")
            text = doc.text.strip()
            
            doc_str = f"Tài liệu [{i}] - {title}\n{text}\nNguồn: {url}\n"
            
            # Tránh tràn context window
            if current_len + len(doc_str) > self.max_context_chars:
                # Cắt bớt phần text nếu doc đầu tiên đã quá dài
                if i == 1:
                    allowed_len = self.max_context_chars - len(f"Tài liệu [{i}] - {title}\n\nNguồn: {url}\n")
                    if allowed_len > 0:
                        doc_str = f"Tài liệu [{i}] - {title}\n{text[:allowed_len]}...\nNguồn: {url}\n"
                        parts.append(doc_str)
                break
                
            parts.append(doc_str)
            current_len += len(doc_str)
            
        return "\n".join(parts)
