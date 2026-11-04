"""Ollama local chat adapter. A network/model failure is never replaced with a fake AI reply."""
import os
import json
import time
import requests
from dotenv import load_dotenv
from .config import ROOT

class ModelUnavailable(RuntimeError):
    pass

class LocalLLM:
    def __init__(self):
        load_dotenv(ROOT / ".env")
        self.base = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "qwen3:4b")

    def status(self):
        try:
            result = requests.get(self.base + "/api/tags", timeout=5)
            result.raise_for_status()
            names = [m["name"] for m in result.json().get("models", [])]
            return {"ready": self.model in names, "model": self.model, "models": names}
        except (requests.RequestException, ValueError):
            return {"ready": False, "model": self.model, "models": []}

    def generate(self, messages):
        try:
            schema = {"type": "object", "properties": {
                "answer": {"type": "string", "maxLength": 1600},
                "citations": {"type": "array", "items": {"type": "integer"}, "maxItems": 3},
                "follow_up": {"type": "string", "maxLength": 200}},
                "required": ["answer", "citations", "follow_up"], "additionalProperties": False}
            payload = {"model": self.model, "messages": messages, "stream": True,
                "think": False, "format": schema, "keep_alive": "10m",
                "options": {"temperature": .2, "num_ctx": 4096, "num_predict": 700}}
            endpoint = "/api/chat"
            if self.model.startswith("qwen3:"):
                # The installed Qwen template always opens <think>, even for think=false.
                # Explicitly close it using Qwen's non-thinking chat prefix, without
                # changing the user's installed model or exposing reasoning in chat.
                prompt = ""
                for message in messages:
                    if message["role"] not in ("system", "user", "assistant"):
                        raise ValueError("Invalid chat role")
                    content = message["content"].replace("<|", "< |")
                    prompt += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
                prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                payload.pop("messages")
                payload.update({"raw": True, "prompt": prompt})
                endpoint = "/api/generate"
            response = requests.post(self.base + endpoint, json=payload,
                timeout=(5, 180), stream=True)
            response.raise_for_status()
            parts, result = [], {}
            started = time.monotonic()
            with response:
                for line in response.iter_lines():
                    if time.monotonic() - started > 300:
                        raise ModelUnavailable("Mô hình chạy quá chậm. Thử câu hỏi ngắn hơn hoặc giảm ngữ cảnh.")
                    if not line:
                        continue
                    result = json.loads(line)
                    if result.get("error"):
                        raise ModelUnavailable("Ollama báo lỗi khi suy luận.")
                    parts.append(result.get("response", "") if endpoint == "/api/generate" else result.get("message", {}).get("content", ""))
                    # Some local models keep producing whitespace after a complete JSON object.
                    # Finish once a complete, parseable object arrives; ReviewAgent still validates it.
                    try:
                        complete = json.loads("".join(parts))
                        if isinstance(complete, dict) and all(k in complete for k in ("answer", "citations", "follow_up")):
                            return json.dumps(complete, ensure_ascii=False)
                    except json.JSONDecodeError:
                        pass
                    if result.get("done"):
                        break
            self.last_partial = "".join(parts)
            if not result.get("done") or result.get("done_reason") == "length":
                raise ModelUnavailable("Mô hình chưa trả lời hoàn chỉnh. Hãy thử câu hỏi ngắn hơn.")
            content = "".join(parts).strip()
            if not content:
                raise ModelUnavailable("Mô hình trả lời rỗng.")
            return content
        except (requests.RequestException, ValueError) as exc:
            raise ModelUnavailable("Không nhận được câu trả lời từ Ollama. Kiểm tra Ollama và mô hình đã chọn.") from exc
