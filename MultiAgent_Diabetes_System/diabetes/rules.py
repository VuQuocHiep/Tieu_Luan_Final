from .config import SOURCE
from .schema import Patient

LABELS = {0: "Trong khoảng tham chiếu của chỉ số đã nhập",
          1: "Trong khoảng tiền đái tháo đường",
          2: "Đạt ngưỡng nghi ngờ đái tháo đường"}

def screen(patient: Patient):
    patient.validate()
    notes = ["Kết quả sàng lọc, cần bác sĩ đánh giá và xét nghiệm xác nhận; không phân biệt type 1/type 2."]
    if patient.pregnant or (patient.age is not None and patient.age < 18):
        return {"level": None, "label": "Cần đánh giá chuyên khoa", "evidence": [],
                "notes": ["Bộ quy tắc này chỉ áp dụng cho người trưởng thành không mang thai."], "source": SOURCE}
    evidence = []
    if patient.hba1c is not None:
        x = patient.hba1c
        evidence.append({"test": "HbA1c", "value": x, "unit": "%",
                         "level": 2 if x >= 6.5 else 1 if x >= 5.7 else 0})
        notes.append("HbA1c có thể bị ảnh hưởng bởi thiếu máu, biến thể hemoglobin và một số bệnh lý.")
    if patient.fasting_glucose is not None:
        if patient.fasting_confirmed:
            x = patient.fasting_glucose
            evidence.append({"test": "Glucose đói", "value": x, "unit": "mg/dL",
                             "level": 2 if x >= 126 else 1 if x >= 100 else 0})
            if x < 70:
                notes.append("Glucose thấp: liên hệ nhân viên y tế; nếu lú lẫn, ngất hoặc không thể nuốt, cần hỗ trợ cấp cứu.")
        else:
            notes.append("Chưa xác nhận nhịn ăn ít nhất 8 giờ: không áp ngưỡng glucose đói.")
    if not evidence:
        return {"level": None, "label": "Chưa đủ điều kiện diễn giải", "evidence": [], "notes": notes, "source": SOURCE}
    level = max(e["level"] for e in evidence)
    if len({e["level"] for e in evidence}) > 1:
        notes.append("Các xét nghiệm không cùng khoảng phân loại; cần kiểm tra lại với bác sĩ.")
    return {"level": level, "label": LABELS[level], "evidence": evidence, "notes": notes, "source": SOURCE}
