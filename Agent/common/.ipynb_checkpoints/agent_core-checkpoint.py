"""Tool-calling loop shared by the document and warehouse agents."""
import json
from dataclasses import dataclass
from typing import Callable, Any
from common.ollama_client import OllamaClient, OllamaError


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}


class AgentRunner:
    def __init__(self, client: OllamaClient, system_prompt: str, tools: list[Tool], max_steps: int = 5):
        self.client = client
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps

    def run(self, query: str, history: list[dict[str, str]] | None = None):
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": query})
        trace = []
        for _ in range(self.max_steps):
            msg = self.client.chat(messages, tools=[t.schema() for t in self.tools.values()])
            calls = msg.get("tool_calls") or []
            if not calls:
                return {"answer": msg.get("content", "").strip() or "Không có câu trả lời.", "trace": trace}
            messages.append(msg)
            for call in calls:
                function = call.get("function", {})
                name = function.get("name", "")
                args = function.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                if name not in self.tools:
                    output = {"error": f"Công cụ {name} không được cho phép."}
                else:
                    try:
                        output = self.tools[name].handler(**args)
                    except (TypeError, ValueError, KeyError) as exc:
                        output = {"error": f"Tham số hoặc dữ liệu không hợp lệ: {exc}"}
                    except Exception as exc:
                        output = {"error": f"Công cụ bị lỗi: {type(exc).__name__}"}
                trace.append({"tool": name, "arguments": args, "result": output})
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(output, ensure_ascii=False, default=str)})
        # Force a final response without further tools if the agent reaches its step limit.
        messages.append({"role": "system", "content": "Đã hết lượt gọi công cụ. Hãy trả lời chỉ dựa trên dữ liệu đã nhận, nêu rõ nếu còn thiếu thông tin."})
        try:
            result = self.client.chat(messages)
            answer = result.get("content", "").strip()
        except OllamaError:
            answer = "Đã tới giới hạn số bước; không thể tổng hợp câu trả lời. Xem nhật ký công cụ."
        return {"answer": answer, "trace": trace}
