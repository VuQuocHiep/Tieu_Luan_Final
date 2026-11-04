"""Reproducible preparation of public CDC NHANES data; no generated disease labels."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import requests
from .config import DATA, FEATURES, prepare

TARGET = ("Nhãn DIQ010: người tham gia tự báo cáo từng được nhân viên y tế thông báo mắc đái tháo đường; "
          "1=yes, 2=no. Không phải nhãn chẩn đoán mới từ xét nghiệm; người đang điều trị có thể có chỉ số bình thường.")

def prepare_nhanes(year=2017):
    if year not in (2015, 2017):
        raise ValueError("Hỗ trợ NHANES 2015–2016 và 2017–2018.")
    prepare()
    suffix = "I" if year == 2015 else "J"
    raw = DATA / "nhanes" / str(year)
    raw.mkdir(parents=True, exist_ok=True)
    columns = {"DEMO": ["SEQN", "RIDAGEYR", "RIDEXPRG"], "BMX": ["SEQN", "BMXBMI"],
               "GLU": ["SEQN", "LBXGLU", "WTSAF2YR"], "GHB": ["SEQN", "LBXGH"],
               "DIQ": ["SEQN", "DIQ010"]}
    frame = None
    provenance = []
    for component, names in columns.items():
        filename = f"{component}_{suffix}"
        url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{filename}.xpt"
        path = raw / (filename + ".xpt")
        if not path.exists():
            response = requests.get(url, timeout=(10, 60))
            response.raise_for_status()
            if not response.content.startswith(b"HEADER RECORD"):
                raise ValueError(f"CDC không trả tệp XPT hợp lệ: {filename}")
            path.write_bytes(response.content)
        part = pd.read_sas(path, format="xport")[names]
        frame = part if frame is None else frame.merge(part, on="SEQN", how="inner", validate="one_to_one")
        provenance.append({"url": url, "codebook": url[:-4] + ".htm",
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    total = len(frame)
    frame = frame[(frame.RIDAGEYR >= 18) & (frame.RIDEXPRG != 1) &
                  (frame.WTSAF2YR > 0) & frame.DIQ010.isin([1, 2])].copy()
    frame = frame.rename(columns={"SEQN": "participant_id", "RIDAGEYR": "age", "BMXBMI": "bmi",
                                  "LBXGLU": "fasting_glucose", "LBXGH": "hba1c"})
    frame["outcome"] = (frame.DIQ010 == 1).astype(int)
    frame = frame.dropna(subset=FEATURES)
    # Keep clinical plausibility checks aligned with application schema, without silently clipping.
    from .schema import Patient
    valid = []
    for row in frame[FEATURES].to_dict("records"):
        try:
            Patient(**row).validate()
            valid.append(True)
        except ValueError:
            valid.append(False)
    frame = frame.loc[valid, ["participant_id"] + FEATURES + ["outcome"]]
    path = DATA / f"nhanes_{year}.csv"
    frame.to_csv(path, index=False)
    metadata = {"dataset": f"nhanes_{year}", "synthetic": False, "year": year,
        "target_description": TARGET, "merged_rows": total, "complete_adult_rows": len(frame),
        "positive": int(frame.outcome.sum()), "sources": provenance,
        "limitations": "Complete-case sample, excluding known pregnancy; no survey-weighted population estimates. US sample, not validated for Vietnam. Pregnancy status not available for everyone. Adults aged 80+ top-coded."}
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return path, frame, metadata
