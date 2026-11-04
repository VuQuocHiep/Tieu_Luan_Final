from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"
KB = ROOT / "knowledge"
SOURCE = "https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis"
FEATURES = ["fasting_glucose", "hba1c", "age", "bmi"]

def prepare():
    for path in (DATA, ARTIFACTS, OUTPUTS, KB):
        path.mkdir(parents=True, exist_ok=True)
