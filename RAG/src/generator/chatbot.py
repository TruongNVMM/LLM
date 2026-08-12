#!/usr/ विक्रांत/env python3
"""
CLI Chatbot cho hệ thống HUST RAG.
Cho phép hỏi đáp tương tác trực tiếp trên terminal với streaming.
"""
import argparse
import sys
import os
import logging
from termcolor import colored

# Thêm root path để import
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.generator.rag_chain import RAGChain, RAGChainConfig

# Tắt log quá chi tiết
logging.getLogger("httpx").setLevel(logging.WARNING)

def print_welcome():
    print(colored("="*60, "cyan", attrs=["bold"]))
    print(colored(" HUST SỔ TAY SINH VIÊN - RAG CHATBOT ", "white", "on_blue", attrs=["bold"]).center(70))
    print(colored("="*60, "cyan", attrs=["bold"]))
    print(" Lệnh đặc biệt:")
    print(f"   {colored('/quit', 'yellow')} hoặc {colored('/exit', 'yellow')} : Thoát chương trình")
    print(f"   {colored('/clear', 'yellow')}               : Xóa màn hình")
    print(f"   {colored('/sources', 'yellow')}             : Xem các nguồn tài liệu của câu trả lời gần nhất")
    print(f"   {colored('/mode', 'yellow')}                : Xem/đổi search mode (vd: /mode rerank-expand)")
    print(colored("="*60, "cyan"))
    print()

def main():
    parser = argparse.ArgumentParser(description="HUST RAG Chatbot")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"), help="Ollama model")
    parser.add_argument("--mode", default=os.getenv("RAG_SEARCH_MODE", "rerank-expand"), help="Search mode")
    parser.add_argument("--url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), help="Ollama base URL")
    args = parser.parse_args()

    config = RAGChainConfig(
        model=args.model,
        search_mode=args.mode,
        base_url=args.url
    )

    print("Đang khởi tạo hệ thống RAG...")
    chain = RAGChain(config)

    if not chain.health_check():
        print(colored(f"\n[LỖI] Không thể kết nối đến Ollama hoặc không tìm thấy model '{args.model}'.", "red", attrs=["bold"]))
        print(f"Vui lòng đảm bảo Ollama đang chạy và bạn đã tải model bằng lệnh:\n> ollama pull {args.model}")
        sys.exit(1)

    print_welcome()
    
    last_sources = []

    try:
        while True:
            # Nhận input
            try:
                query = input(colored("\nBạn: ", "green", attrs=["bold"])).strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not query:
                continue

            # Xử lý lệnh đặc biệt
            if query.lower() in ["/quit", "/exit"]:
                break
            elif query.lower() == "/clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                print_welcome()
                continue
            elif query.lower() == "/sources":
                if not last_sources:
                    print(colored("Chưa có câu hỏi nào hoặc không tìm thấy nguồn.", "yellow"))
                else:
                    print(colored("\n[NGUỒN TÀI LIỆU THAM KHẢO]", "cyan", attrs=["bold"]))
                    seen = set()
                    for i, doc in enumerate(last_sources, 1):
                        title = doc.metadata.get("title", "N/A")
                        url = doc.metadata.get("source_url", "N/A")
                        # Tránh in trùng bài viết
                        doc_id = doc.metadata.get("doc_id")
                        if doc_id and doc_id in seen:
                            continue
                        if doc_id: seen.add(doc_id)
                        
                        print(f"  {i}. {title}")
                        print(f"     URL: {colored(url, 'blue', attrs=['underline'])}")
                continue
            elif query.lower().startswith("/mode"):
                parts = query.split()
                if len(parts) == 1:
                    print(f"Search mode hiện tại: {colored(chain.config.search_mode, 'yellow')}")
                else:
                    new_mode = parts[1]
                    if new_mode in ["hybrid", "rerank", "expand", "rerank-expand"]:
                        chain.config.search_mode = new_mode
                        print(f"Đã chuyển search mode sang: {colored(new_mode, 'green')}")
                    else:
                        print(colored(f"Chế độ không hợp lệ. Chọn: hybrid, rerank, expand, rerank-expand", "red"))
                continue

            # Trả lời câu hỏi
            print(colored("Chatbot: ", "magenta", attrs=["bold"]), end="", flush=True)
            
            try:
                last_sources = []
                for chunk in chain.answer_stream(query):
                    if chunk["type"] == "chunk":
                        print(colored(chunk["content"], "white"), end="", flush=True)
                    elif chunk["type"] == "done":
                        last_sources = chunk["sources"]
                print() # Newline sau khi stream xong
            except Exception as e:
                print(colored(f"\n[LỖI] Đã xảy ra lỗi khi tạo câu trả lời: {e}", "red"))
                
    finally:
        chain.close()
        print(colored("\nĐã đóng kết nối. Tạm biệt!", "yellow"))

if __name__ == "__main__":
    main()
