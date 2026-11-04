import io
import unittest
import tempfile
from pathlib import Path
from diabetes.schema import Patient
from diabetes.rules import screen
from diabetes.imaging import parse_text, render_table
from diabetes.storage import Database
from diabetes.agents import Coordinator, PresentationPolicy

class RulesTest(unittest.TestCase):
    def test_glucose_boundaries(self):
        for value, expected in [(99.9, 0), (100, 1), (125.9, 1), (126, 2)]:
            with self.subTest(value=value):
                self.assertEqual(screen(Patient(value, age=40, fasting_confirmed=True))["level"], expected)

    def test_a1c_boundaries(self):
        for value, expected in [(5.69, 0), (5.7, 1), (6.49, 1), (6.5, 2)]:
            self.assertEqual(screen(Patient(hba1c=value, age=40))["level"], expected)

    def test_unconfirmed_fasting(self):
        self.assertIsNone(screen(Patient(160, age=40))["level"])

    def test_pregnancy_and_child(self):
        self.assertIsNone(screen(Patient(160, 7, 40, pregnant=True))["level"])
        self.assertIsNone(screen(Patient(160, 7, 15))["level"])

    def test_missing_invalid(self):
        for patient in [Patient(), Patient(hba1c=float("nan")), Patient(hba1c=40)]:
            with self.assertRaises(ValueError):
                patient.validate()

class OCRParserTest(unittest.TestCase):
    def test_units_and_decimal(self):
        values, _ = parse_text("Fasting glucose | 7,0 | mmol/L\nHbA1c | 6,5 | %\nBMI | 27 | kg/m2")
        self.assertAlmostEqual(values["fasting_glucose"], 126.1274, places=3)
        self.assertEqual(values["hba1c"], 6.5)
        self.assertEqual(values["bmi"], 27)

    def test_missing_unit_not_guessed(self):
        values, warnings = parse_text("Fasting glucose 7.0\nHbA1c 48 mmol/mol")
        self.assertEqual(values, {})
        self.assertEqual(len(warnings), 2)

    def test_duplicate_ambiguous(self):
        values, _ = parse_text("HbA1c 6.5 %\nHbA1c 5.2 %\nHbA1c 7 %")
        self.assertNotIn("hba1c", values)

    def test_missing_value_not_reference_range(self):
        values, _ = parse_text("Fasting glucose -- mg/dL 70-99\nBMI -- kg/m2")
        self.assertEqual(values, {})

    def test_generated_pixel_image(self):
        from PIL import Image
        content = render_table(Patient(145, 7.1, 45, 27))
        image = Image.open(io.BytesIO(content))
        self.assertEqual(image.size, (1200, 650))
        self.assertFalse(image.info)

class IntegrationTest(unittest.TestCase):
    def test_mysql_settings_override_old_url(self):
        import os
        from unittest.mock import patch
        from diabetes.storage import connection_url
        with patch.dict(os.environ, {"MYSQL_USER": "test_user", "MYSQL_PASSWORD": "test@#%pass",
                "MYSQL_DATABASE": "mutilagent", "DATABASE_URL": "mysql+pymysql://old:old@localhost/old"}, clear=True):
            url = connection_url()
            self.assertEqual(url.username, "test_user")
            self.assertEqual(url.password, "test@#%pass")
            self.assertEqual(url.database, "mutilagent")

    def setUp(self):
        self.db = Database("sqlite:///:memory:")

    def tearDown(self):
        self.db.engine.dispose()

    def test_coordinator_save_feedback(self):
        result = Coordinator(self.db).run(Patient(145, 7.1, 45, 27, True), persist=True)
        self.assertEqual(result["rule"]["level"], 2)
        self.assertTrue(result["rag"]["sources"])
        run_id = result["run_id"]
        self.db.feedback(run_id, 0)
        self.db.feedback(run_id, 1)
        self.assertEqual(self.db.stats(), [("concise", 1, 1)])
        self.assertEqual(len(self.db.recent()), 1)

    def test_import_replace_and_load(self):
        import pandas as pd
        frame = pd.DataFrame([{"fasting_glucose": 145, "outcome": 1}])
        self.db.import_samples(frame, "test")
        self.db.import_samples(frame, "test")
        self.assertEqual(len(self.db.load_samples("test")), 1)

    def test_bandit_explores(self):
        policy = PresentationPolicy()
        self.assertEqual(policy.choose([]), "concise")
        self.assertEqual(policy.choose([("concise", 1, 1)]), "detailed")
        from decimal import Decimal
        self.assertIn(policy.choose([("concise", 2, Decimal("1")),
                                     ("detailed", 1, Decimal("1"))]), policy.actions)

    def test_reject_bad_training(self):
        import pandas as pd
        from diabetes.learning import train
        with self.assertRaises(ValueError):
            train(frame=pd.DataFrame({"wrong": [1]}))

    def test_external_split_rejects_same_participant(self):
        import pandas as pd
        from diabetes.learning import train
        row = {"participant_id": 10, "fasting_glucose": 145, "hba1c": 7.1, "age": 45, "bmi": 27, "outcome": 1}
        frame = pd.DataFrame([row])
        with self.assertRaisesRegex(ValueError, "participant_id"):
            train(frame=frame, external_test=frame)

    def test_train_predict_held_out(self):
        import numpy as np
        import pandas as pd
        from unittest.mock import patch
        from diabetes.learning import train, predict
        rng = np.random.default_rng(10)
        n = 200
        frame = pd.DataFrame({"fasting_glucose": rng.uniform(80, 200, n),
            "hba1c": rng.uniform(4.5, 9, n), "age": rng.integers(20, 80, n),
            "bmi": rng.uniform(20, 35, n)})
        frame["outcome"] = ((frame.fasting_glucose >= 126) | (frame.hba1c >= 6.5)).astype(int)
        with tempfile.TemporaryDirectory() as temporary:
            with patch("diabetes.learning.ARTIFACTS", Path(temporary)):
                metrics = train(frame=frame, synthetic=True)
                self.assertEqual(metrics["test_size"], 50)
                result = predict(Patient(145, 7.1, 45, 27, True))
                self.assertTrue(result["available"])
                self.assertTrue(result["synthetic"])
                self.assertTrue(0 <= result["score"] <= 1)

if __name__ == "__main__":
    unittest.main()
