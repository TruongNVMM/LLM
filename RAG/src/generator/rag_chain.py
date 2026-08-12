import logging
from dataclasses import dataclass
from typing import Iterator, Any, List

from .ollama_client import OllamaClient
from .prompt_builder import PromptBuilder
from src.embedding.retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class RAGChainConfig:
    search_mode: str = "rerank-expand" # hybrid, rerank, expand, rerank-expand
    top_k: int = 10
    candidate_k: int = 60
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    max_context_chars: int = 6000
    temperature: float = 0.1

class RAGChain:
    """
    Orchestrator kết nối HybridRetriever và OllamaClient để tạo thành một hệ thống RAG hoàn chỉnh.
    """
    def __init__(self, config: RAGChainConfig = None):
        self.config = config or RAGChainConfig()
        
        # Init Retriever
        self.retriever = HybridRetriever(
            top_k=self.config.top_k,
            candidate_k=self.config.candidate_k,
            rerank_enabled=("rerank" in self.config.search_mode)
        )
        
        # Init Generator
        self.llm = OllamaClient(
            base_url=self.config.base_url,
            model=self.config.model
        )
        
        # Init Prompt Builder
        self.prompt_builder = PromptBuilder(
            max_context_chars=self.config.max_context_chars
        )

    def health_check(self) -> bool:
        """Kiểm tra trạng thái của các thành phần."""
        return self.llm.health_check()

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Truy xuất context từ câu hỏi."""
        mode = self.config.search_mode
        
        if mode == "hybrid":
            return self.retriever.search(query, top_k=self.config.top_k)
        elif mode == "rerank":
            return self.retriever.search_with_rerank(query, top_k=self.config.top_k)
        elif mode == "expand":
            return self.retriever.search_with_expansion(query, top_k=self.config.top_k)
        elif mode == "rerank-expand":
            return self.retriever.search_with_rerank_and_expansion(query, top_k=self.config.top_k)
        else:
            logger.warning(f"Unknown search mode '{mode}', fallback to hybrid")
            return self.retriever.search(query, top_k=self.config.top_k)

    def answer(self, query: str) -> dict:
        """
        Trả về toàn bộ câu trả lời cùng với sources.
        """
        # 1. Retrieve
        docs = self.retrieve(query)
        
        # 2. Build prompt
        system_prompt = self.prompt_builder.build_system()
        user_prompt = self.prompt_builder.build_prompt(query, docs)
        
        # 3. Generate
        answer_text = self.llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            stream=False,
            temperature=self.config.temperature
        )
        
        return {
            "answer": answer_text,
            "sources": docs,
            "prompts": {
                "system": system_prompt,
                "user": user_prompt
            }
        }

    def answer_stream(self, query: str) -> Iterator[dict]:
        """
        Yield chunk của câu trả lời dần dần (dành cho streaming UI).
        Yield item cuối cùng sẽ chứa metadata (sources).
        """
        docs = self.retrieve(query)
        system_prompt = self.prompt_builder.build_system()
        user_prompt = self.prompt_builder.build_prompt(query, docs)
        
        stream = self.llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            stream=True,
            temperature=self.config.temperature
        )
        
        full_answer = ""
        for chunk in stream:
            full_answer += chunk
            yield {"type": "chunk", "content": chunk}
            
        # Cuối cùng yield nguồn
        yield {
            "type": "done",
            "content": full_answer,
            "sources": docs
        }

    def close(self):
        self.retriever.close()
        
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
