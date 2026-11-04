"""Real pixel OCR smoke test; first run may download public OCR weights."""
import json
from diabetes.imaging import OCRAgent, render_table
from diabetes.schema import Patient
from diabetes.rules import screen

engine = OCRAgent()
for g, a, expected in [(90, 5.2, 0), (112, 6.0, 1), (145, 7.1, 2)]:
    result = engine.read(render_table(Patient(g, a, 45, 27)))
    print(json.dumps(result, ensure_ascii=True))
    assert abs(result["values"]["fasting_glucose"] - g) < .01
    assert abs(result["values"]["hba1c"] - a) < .01
    patient = Patient(**result["values"], fasting_confirmed=True)
    assert screen(patient)["level"] == expected
print("PASS: 3 image -> OCR -> rule cases")
