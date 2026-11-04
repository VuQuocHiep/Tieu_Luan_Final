"""Render actual pixel tables and recover values using OCR, never PNG metadata."""
import io
import re
import unicodedata
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

def font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)

def render_table(patient):
    patient.validate()
    image = Image.new("RGB", (1200, 650), "#f4f8ff")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1200, 110), fill="#082344")
    draw.text((45, 27), "BẢNG CHỈ SỐ XÉT NGHIỆM — MÔ PHỎNG", font=font(34), fill="white")
    rows = [("Chỉ số / Test", "Kết quả / Result", "Đơn vị / Unit"),
            ("Fasting glucose", patient.fasting_glucose, "mg/dL"),
            ("HbA1c", patient.hba1c, "%"), ("Age", patient.age, "years"),
            ("BMI", patient.bmi, "kg/m2")]
    for i, row in enumerate(rows):
        y = 135 + i * 78
        draw.rectangle((35, y, 1165, y + 77), fill="#d7e7fa" if i == 0 else "white", outline="#b7c8dc")
        for x, value in zip((55, 600, 900), row):
            value = "--" if value is None else f"{value:g}" if isinstance(value, (float, int)) else value
            draw.text((x, y + 20), value, font=font(28), fill="#082344")
    draw.text((45, 555), "Ảnh minh họa phần mềm — không phải phiếu xét nghiệm thật.", font=font(24), fill="#334155")
    draw.text((45, 600), "Cần xác nhận đơn vị và tình trạng nhịn ăn trước khi phân tích.", font=font(23), fill="#334155")
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()

def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
                   if unicodedata.category(c) != "Mn")

def parse_text(text):
    """Only accept named rows and explicit units. Ambiguity stays unresolved."""
    values, warnings = {}, []
    for line in text.splitlines():
        s = normalize(line)
        key = None
        if re.search(r"hba\s*1\s*c|a1c", s):
            key, unit = "hba1c", "%"
            tail = re.split(r"hba\s*1\s*c|a1c", s, maxsplit=1)[-1]
        elif re.search(r"fasting\s+glucose|glucose\s+doi|duong\s+huyet\s+doi|fpg", s):
            key, unit = "fasting_glucose", "glucose"
            tail = re.split(r"fasting\s+glucose|glucose\s+doi|duong\s+huyet\s+doi|fpg", s, maxsplit=1)[-1]
        elif re.search(r"\bage\b|\btuoi\b|\bbmi\b", s):
            match = re.search(r"\bage\b|\btuoi\b|\bbmi\b", s)
            key = "bmi" if match.group() == "bmi" else "age"
            unit, tail = "other", s[match.end():]
        if not key:
            continue
        # Never take a number from the unit (kg/m2) or a reference range.
        match = re.match(r"\s*[:|=]?\s*(\d+(?:[.,]\d+)?)\s*(.*)", tail)
        if not match:
            continue
        number, suffix = float(match[1].replace(",", ".")), match[2]
        if unit == "%" and "%" not in suffix:
            warnings.append("HbA1c thiếu đơn vị %; cần nhập lại.")
            continue
        if unit == "glucose":
            if re.search(r"mmol\s*/?\s*l", suffix):
                number *= 18.0182
            elif not re.search(r"mg\s*/?\s*dl", suffix):
                warnings.append("Glucose thiếu đơn vị mg/dL hoặc mmol/L; cần nhập lại.")
                continue
        if key in values:
            values[key] = None
            warnings.append(f"Có nhiều dòng {key}; cần kiểm tra thủ công.")
        else:
            values[key] = round(number, 4)
    return {k: v for k, v in values.items() if v is not None}, warnings

class OCRAgent:
    def __init__(self):
        from rapidocr import RapidOCR
        self.engine = RapidOCR()

    def read(self, content):
        import numpy as np
        with Image.open(io.BytesIO(content)) as source:
            if source.width * source.height > 20_000_000:
                raise ValueError("Ảnh quá lớn (tối đa 20 triệu pixel).")
            image = ImageOps.exif_transpose(source).convert("RGB")
        result = self.engine(np.array(image)[:, :, ::-1].copy())
        rows = []
        items = zip(result.boxes, result.txts, result.scores) if result.boxes is not None else []
        for box, text, score in items:
            if float(score) < .65:
                continue
            x, y = min(p[0] for p in box), sum(p[1] for p in box) / 4
            height = max(p[1] for p in box) - min(p[1] for p in box)
            rows.append((y, x, height, text, float(score)))
        groups = []
        for item in sorted(rows):
            if groups and abs(item[0] - groups[-1][0][0]) < max(12, item[2] * .6):
                groups[-1].append(item)
            else:
                groups.append([item])
        text = "\n".join(" | ".join(i[3] for i in sorted(g, key=lambda i: i[1])) for g in groups)
        values, warnings = parse_text(text)
        return {"text": text, "values": values, "warnings": warnings,
                "mean_confidence": sum(i[4] for i in rows) / len(rows) if rows else 0}
