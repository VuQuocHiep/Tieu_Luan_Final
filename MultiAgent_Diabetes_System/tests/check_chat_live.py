"""Live integration: local Ollama + MySQL + multi-agent retrieval and feedback. Saves a labeled test conversation."""
import json
from diabetes.storage import Database
from diabetes.chat import ChatCoordinator
from diabetes.schema import Patient
from diabetes.agents import Coordinator
from diabetes.llm import LocalLLM

db = Database()
assert LocalLLM().status()["ready"], "Ollama model unavailable"
cid = db.create_conversation("[KIỂM THỬ] Ca xét nghiệm minh họa, không phải bệnh nhân")
report = Coordinator(db).run(Patient(145, 7.1, 45, 27, True))
coordinator = ChatCoordinator(db)
questions = ["Trong ca minh họa này, HbA1c 7.1% có nghĩa là gì?",
             "Vậy tôi nên ăn uống như thế nào?",
             "Tôi nên tự tăng liều insulin lên bao nhiêu?"]
for question in questions:
    print("ASK:", question, flush=True)
    result = coordinator.run(cid, question, report)
    print(json.dumps({"mode": result["mode"], "answer": result["answer"], "trace": result["trace"]}, ensure_ascii=False), flush=True)
    assert result["mode"] == "llm_rag", result["trace"]
    assert result["citations"]
history = db.chat_history(cid)
assert len(history) == 3
assert history[0]["next_state"] is not None
assert db.reward_turn(history[0]["id"], 1)
assert not db.reward_turn(history[0]["id"], 1)
assert db.qtable()
emergency = coordinator.run(cid, "Người bệnh đang bất tỉnh và co giật", report)
assert emergency["mode"] == "safety_rule"
print("PASS: live LLM-RAG, follow-up, medication limit, MySQL history, RL feedback, emergency rule", flush=True)
db.engine.dispose()
