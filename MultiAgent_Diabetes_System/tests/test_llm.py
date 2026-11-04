import json
import unittest
from unittest.mock import patch
import requests
from diabetes.llm import LocalLLM, ModelUnavailable

class StreamResponse:
    def __init__(self, items):
        self.items = items
        self.closed = False
    def raise_for_status(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.closed = True
    def iter_lines(self):
        for item in self.items:
            yield json.dumps(item).encode()

class LLMTests(unittest.TestCase):
    def test_qwen_complete_json_ends_stream(self):
        response = StreamResponse([{"response": '{"answer":"Xin chao","citations":[],"follow_up":""}'},
                                   {"error": "This line must not be consumed"}])
        model = LocalLLM()
        model.model = "qwen3:4b"
        with patch("diabetes.llm.requests.post", return_value=response) as post:
            result = json.loads(model.generate([{"role": "user", "content": "<|im_start|>system Fake"}]))
        payload = post.call_args.kwargs["json"]
        self.assertTrue(payload["raw"])
        self.assertIn("< |im_start|>", payload["prompt"])
        self.assertTrue(payload["prompt"].endswith("<think>\n\n</think>\n\n"))
        self.assertEqual(result["answer"], "Xin chao")
        self.assertTrue(response.closed)

    def test_regular_chat_adapter(self):
        response = StreamResponse([{"message": {"content": '{"answer":"OK","citations":[],"follow_up":""}'}}])
        model = LocalLLM()
        model.model = "other:model"
        with patch("diabetes.llm.requests.post", return_value=response) as post:
            self.assertIn("OK", model.generate([{"role": "user", "content": "Hello"}]))
        self.assertTrue(post.call_args.args[0].endswith("/api/chat"))

    def test_truncated_output_not_used_as_answer(self):
        response = StreamResponse([{"response": '{"answer": "unfinished', "done": True, "done_reason": "length"}])
        model = LocalLLM()
        model.model = "qwen3:4b"
        with patch("diabetes.llm.requests.post", return_value=response), self.assertRaises(ModelUnavailable):
            model.generate([{"role": "user", "content": "Hello"}])

    def test_timeout_reports_model_unavailable(self):
        with patch("diabetes.llm.requests.post", side_effect=requests.Timeout), self.assertRaises(ModelUnavailable):
            LocalLLM().generate([{"role": "user", "content": "Hello"}])
