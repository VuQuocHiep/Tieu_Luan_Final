"""App sign_language | CNN using pytorch. Shared CNN architecture, independent entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.train_runner import fit_one
from shared.inference import get_result, load_model, predict_image

DATASET = "sign_language"
METHOD = "pytorch"

def train(**settings):
    return fit_one(DATASET, METHOD, settings)

def predict(image):
    model = load_model(DATASET, METHOD)
    return predict_image(model, METHOD, image)

if __name__ == '__main__':
    print(train())
