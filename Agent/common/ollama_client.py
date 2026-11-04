"""A small, optional Ollama client. Does not require cloud credentials."""
import os
from typing import Any
import requests


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:4b")
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")

    def available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(m.get("name", "") in {self.model, self.model + ":latest"}
                       or (self.model.endswith(":latest") and m.get("name") == self.model[:-7])
                       for m in models)
        except (requests.RequestException, ValueError, TypeError):
            return False

    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None,
             timeout: int = 120) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        if tools:
            payload["tools"] = tools
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result.get("message"), dict):
                raise OllamaError("Phản hồi từ Ollama không đúng định dạng.")
            return result["message"]
        except requests.RequestException as exc:
            raise OllamaError(f"Không gọi được Ollama: {exc}") from exc
        except ValueError as exc:
            raise OllamaError("Ollama trả về dữ liệu JSON không hợp lệ.") from exc
