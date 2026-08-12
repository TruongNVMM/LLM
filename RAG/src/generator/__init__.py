"""
HUST RAG - Generator module
Tích hợp local LLM thông qua Ollama để sinh câu trả lời từ context.
"""
from .ollama_client import OllamaClient
from .prompt_builder import PromptBuilder
from .rag_chain import RAGChain, RAGChainConfig

__all__ = [
    'OllamaClient',
    'PromptBuilder',
    'RAGChain',
    'RAGChainConfig',
]
