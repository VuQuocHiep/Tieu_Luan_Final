"""UI integration uses a deterministic model fixture; actual Ollama is checked separately."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from diabetes.storage import Database
from tests.test_chat import FakeLLM

with TemporaryDirectory() as temporary:
    db = Database("sqlite:///" + str(Path(temporary) / "chat_ui.db"))
    with patch("diabetes.storage.Database", return_value=db), patch("diabetes.chat.LocalLLM", FakeLLM):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
        assert not app.exception, app.exception
        app.chat_input[0].set_value("HbA1c là gì?").run(timeout=30)
        assert not app.exception, app.exception
        assert not app.error, [e.value for e in app.error]
        cid = app.session_state["chat_id"]
        first = db.chat_history(cid)[0]
        assert len(app.chat_message) == 2
        app.chat_input[0].set_value("Vậy cần làm gì tiếp?").run(timeout=30)
        assert not app.exception, app.exception
        assert len(db.chat_history(cid)) == 2
        app.button(key=f"chat_vote_{first['id']}_1").click().run(timeout=30)
        assert not app.exception, app.exception
        assert db.qtable()
        next(b for b in app.button if b.label == "Cuộc trò chuyện mới").click().run(timeout=30)
        assert not app.exception, app.exception
        assert "chat_id" not in app.session_state
        assert len(app.chat_message) == 0
    db.engine.dispose()
print("PASS: chat UI, multi-turn history, feedback and new conversation (model fixture)")
