import json
import unittest
from unittest.mock import patch
from diabetes.storage import Database
from diabetes.chat import ChatCoordinator, RetrievalAgent, ReviewAgent, IntakeAgent
from diabetes.knowledge import ingest_text, chunk_text, seed_knowledge, fetch_web
from diabetes.llm import ModelUnavailable

class FakeLLM:
    model = "fixture-model"
    def __init__(self, replies=None):
        self.messages = []
        self.replies = replies or [{"answer": "Cần trao đổi với bác sĩ để xác nhận kết quả [1].", "citations": [1], "follow_up": ""}]
    def generate(self, messages):
        self.messages.append(messages)
        return json.dumps(self.replies[min(len(self.messages) - 1, len(self.replies) - 1)])

class ChatTests(unittest.TestCase):
    def setUp(self):
        self.db = Database("sqlite:///:memory:")
        self.cid = self.db.create_conversation("Unit test")
    def tearDown(self):
        self.db.engine.dispose()

    def test_document_deduplication_and_overlap(self):
        text = "Kiến thức xét nghiệm và chăm sóc người bệnh. " * 50
        first = ingest_text(self.db, text, "Title", "source")
        second = ingest_text(self.db, text, "Title", "source")
        self.assertEqual(first, second)
        self.assertEqual(len(self.db.document_list()), 1)
        chunks = chunk_text(text)
        self.assertEqual(chunks[0][-180:], chunks[1][:180])

    def test_pdf_text_ingestion(self):
        import io
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        from diabetes.knowledge import read_upload
        writer = PdfWriter()
        page = writer.add_blank_page(400, 400)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 20 350 Td (A diabetes education document with enough text for ingestion.) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        out = io.BytesIO()
        writer.write(out)
        content = read_upload("book.pdf", out.getvalue())
        self.assertIn("diabetes education", content)
        self.assertTrue(ingest_text(self.db, content, "Book", "Uploaded test PDF"))

    def test_retrieve_nutrition(self):
        seed_knowledge(self.db)
        docs = RetrievalAgent(self.db).run("dinh dưỡng khẩu phần carbohydrate chế độ ăn")
        self.assertTrue(docs)
        self.assertIn("khẩu phần", docs[0]["title"].lower())

    def test_reviewer_requests_second_round(self):
        llm = FakeLLM([{"answer": "Tham khảo [99]", "citations": [99]},
                       {"answer": "Hãy xác nhận kết quả với bác sĩ [1].", "citations": [1]}])
        result = ChatCoordinator(self.db, llm).run(self.cid, "HbA1c có nghĩa là gì?")
        self.assertEqual(result["mode"], "llm_rag")
        self.assertEqual(len(llm.messages), 2)
        self.assertEqual(len(self.db.chat_history(self.cid)), 1)

    def test_followup_history_and_context(self):
        llm = FakeLLM()
        coordinator = ChatCoordinator(self.db, llm)
        report = {"patient": {"fasting_glucose": 145, "hba1c": 7.1, "age": 45, "bmi": 27,
                              "fasting_confirmed": True, "pregnant": False}, "rule": {"level": 0}}
        result = coordinator.run(self.cid, "Giải thích HbA1c", report)
        self.assertEqual(result["context"]["rule"]["level"], 2)
        coordinator.run(self.cid, "Vậy cần làm gì tiếp?", report)
        self.assertIn("Giải thích HbA1c", str(llm.messages[-1]))
        history = self.db.chat_history(self.cid)
        self.assertIsNotNone(history[0]["next_state"])

    def test_no_fake_llm_on_outage(self):
        llm = FakeLLM()
        with patch.object(llm, "generate", side_effect=ModelUnavailable("offline")):
            with self.assertRaises(ModelUnavailable):
                ChatCoordinator(self.db, llm).run(self.cid, "HbA1c là gì?")
        self.assertEqual(self.db.chat_history(self.cid), [])

    def test_urgent_bypasses_llm(self):
        llm = FakeLLM()
        with patch.object(llm, "generate", side_effect=AssertionError("LLM must not run")):
            result = ChatCoordinator(self.db, llm).run(self.cid, "Người bệnh đang bất tỉnh")
        self.assertEqual(result["mode"], "safety_rule")

    def test_dose_and_invalid_citations_blocked(self):
        review = ReviewAgent()
        docs = [{"text": "source"}]
        self.assertTrue(review.run({"answer": "Uống 500 mg thuốc [1]", "citations": [1]}, docs))
        self.assertIsNone(review.run({"answer": "Glucose 126 mg/dL cần bác sĩ đánh giá [1]", "citations": [1]}, docs))
        self.assertTrue(review.run({"answer": "Không có nguồn", "citations": []}, docs))

    def test_rl_transition_and_idempotence(self):
        first = self.db.save_turn(self.cid, "q1", {}, "s1", "concise")
        second = self.db.save_turn(self.cid, "q2", {}, "s2", "concise")
        self.db.reward_turn(second, 1)
        self.db.reward_turn(first, 1)
        self.assertFalse(self.db.reward_turn(first, -1))
        values = {row["state"]: float(row["value"]) for row in self.db.qtable()}
        self.assertAlmostEqual(values["s2"], .2)
        self.assertAlmostEqual(values["s1"], .232)
        self.assertEqual(self.db.chat_history(self.cid)[0]["reward"], 1)

    def test_web_rejects_local_and_unapproved(self):
        for url in ["http://127.0.0.1/", "https://example.com/", "https://www.cdc.gov:444/", "https://user@www.cdc.gov/"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                fetch_web(url)

    def test_intake_medication_beats_general_words(self):
        self.assertEqual(IntakeAgent().run("Thuốc cho bệnh tiểu đường?", [])["intent"], "medication")

    def test_review_failure_is_visible(self):
        llm = FakeLLM([{"answer": "Sai nguồn [99]", "citations": [99]}])
        result = ChatCoordinator(self.db, llm).run(self.cid, "HbA1c là gì?")
        self.assertEqual(result["mode"], "review_blocked")
        self.assertEqual(result["citations"], [])
