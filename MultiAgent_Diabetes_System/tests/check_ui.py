from streamlit.testing.v1 import AppTest
from pathlib import Path
from unittest.mock import patch
from diabetes.storage import Database
from tempfile import TemporaryDirectory

temporary = TemporaryDirectory()
db = Database("sqlite:///" + str(Path(temporary.name) / "ui.db"))
with patch("diabetes.storage.Database", return_value=db):
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    assert not app.exception, app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception, app.exception
    assert "generated_image" in app.session_state
    app.checkbox[0].check().run(timeout=30)
    next(b for b in app.button if b.label == "Đọc chỉ số bằng OCR").click().run(timeout=60)
    assert not app.exception, app.exception
    assert app.session_state["ocr"]["values"]["fasting_glucose"] == 145
    app.checkbox(key="fasting").check()
    app.checkbox(key="confirmed").check()
    next(b for b in app.button if b.label == "Phân tích với các Agent").click().run(timeout=30)
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]
    result = app.session_state["result"]
    assert result["rule"]["level"] == 2
    assert "run_id" in result
    next(b for b in app.button if b.label == "Hữu ích").click().run(timeout=30)
    assert db.stats()[0][2] == 1
db.engine.dispose()
temporary.cleanup()
print("PASS: UI generates image, OCRs, confirms, analyzes, saves, and records feedback (isolated test DB)")
