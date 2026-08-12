import json
import logging
from typing import Any, Iterator, Dict
import httpx

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client giao tiếp với Ollama REST API.
    """
    def __init__(
        self, 
        base_url: str = "http://localhost:11434", 
        model: str = "qwen2.5:7b",
        timeout: int = 120
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def health_check(self) -> bool:
        """Kiểm tra kết nối và model có sẵn không."""
        try:
            # 1. Check kết nối
            res = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            res.raise_for_status()
            
            # 2. Check model có tồn tại không
            models = [m.get("name") for m in res.json().get("models", [])]
            if not any(m.startswith(self.model) for m in models):
                logger.warning(f"Ollama connected, but model '{self.model}' not found in {models}.")
                logger.warning(f"Please run: ollama pull {self.model}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            logger.warning(f"Is Ollama running at {self.base_url}?")
            return False

    def generate(self, prompt: str, system: str = "", stream: bool = False, **kwargs) -> Any:
        """
        Gọi endpoint /api/generate.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        
        # Thêm các tham số khác như temperature, top_p...
        if kwargs:
            payload["options"] = kwargs

        if stream:
            return self._generate_stream(url, payload)
        
        res = httpx.post(url, json=payload, timeout=self.timeout)
        res.raise_for_status()
        return res.json().get("response", "")

    def _generate_stream(self, url: str, payload: Dict[str, Any]) -> Iterator[str]:
        """Xử lý streaming response từ Ollama."""
        with httpx.stream("POST", url, json=payload, timeout=self.timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON stream line: {line}")
                        continue
