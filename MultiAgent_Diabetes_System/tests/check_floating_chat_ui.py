"""Verify the corner assistant shares conversation state with the full consultation page."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from diabetes.storage import Database
from tests.test_chat import FakeLLM

with TemporaryDirectory() as temporary:
    db = Database("sqlite:///" + str(Path(temporary) / "floating.db"))
    with patch("diabetes.storage.Database", return_value=db), patch("diabetes.chat.LocalLLM", FakeLLM):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
        assert not app.exception, app.exception
        assert not app.session_state["assistant_popup"]
        app.session_state["assistant_popup"] = True
        app.run(timeout=30)
        assert not app.exception, app.exception
        assert len(app.chat_input) == 2
        app.chat_input(key="floating_consultation_prompt").set_value("HbA1c là gì?").run(timeout=30)
        assert not app.exception, app.exception
        assert not app.error, [e.value for e in app.error]
        cid = app.session_state["chat_id"]
        assert len(db.chat_history(cid)) == 1
        app.session_state["assistant_popup"] = False
        app.run(timeout=30)
        assert len(app.chat_input) == 1
        app.chat_input(key="consultation_prompt").set_value("Vậy cần làm gì tiếp?").run(timeout=30)
        assert not app.exception, app.exception
        assert len(db.chat_history(cid)) == 2
    db.engine.dispose()
print("PASS: floating assistant opens, sends a message, closes and shares history with full chat")
