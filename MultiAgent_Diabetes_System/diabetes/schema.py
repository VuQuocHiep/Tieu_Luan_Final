from dataclasses import asdict, dataclass
import math

@dataclass
class Patient:
    fasting_glucose: float | None = None
    hba1c: float | None = None
    age: float | None = None
    bmi: float | None = None
    fasting_confirmed: bool = False
    pregnant: bool = False

    def validate(self):
        limits = {"fasting_glucose": (20, 1000), "hba1c": (2, 25),
                  "age": (1, 120), "bmi": (8, 100)}
        for key, (low, high) in limits.items():
            value = getattr(self, key)
            if value is not None and (not math.isfinite(value) or not low <= value <= high):
                raise ValueError(f"{key}: giá trị phải trong khoảng {low}–{high}.")
        if self.fasting_glucose is None and self.hba1c is None:
            raise ValueError("Cần ít nhất HbA1c hoặc glucose. Không suy đoán chỉ số bị thiếu.")
        return self

    def to_dict(self):
        return asdict(self)
