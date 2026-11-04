"""A two-hidden-layer neural network; training is explicit and evaluated on a held-out set."""
import json
import pickle
import hashlib
from .config import ARTIFACTS, FEATURES, prepare

def train(csv_path=None, synthetic=False, frame=None, external_test=None, dataset_metadata=None):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
        balanced_accuracy_score, precision_score, recall_score, f1_score, brier_score_loss)
    from sklearn.utils.class_weight import compute_sample_weight
    frame = pd.read_csv(csv_path) if frame is None else frame.copy()
    groups = frame["participant_id"].copy() if "participant_id" in frame else None
    if groups is not None and groups.isna().any():
        raise ValueError("participant_id không được thiếu khi dùng để chia theo bệnh nhân.")
    manifest = dataset_metadata or {}
    synthetic = synthetic or bool(manifest.get("synthetic", False))
    if external_test is not None and "participant_id" in frame and "participant_id" in external_test:
        if set(frame.participant_id) & set(external_test.participant_id):
            raise ValueError("Train và test chứa cùng participant_id.")
    dataset_hash = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    needed = FEATURES + ["outcome"]
    if not set(needed).issubset(frame.columns):
        raise ValueError(f"CSV cần các cột: {', '.join(needed)}")
    frame = frame[needed].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("CSV chứa giá trị thiếu hoặc vô hạn.")
    # Distinct people may legitimately share four measurements but have different
    # labels (e.g. treatment history). Do not force a deterministic clinical label.
    if groups is None:
        frame = frame.drop_duplicates()
    from .schema import Patient
    for row in frame[FEATURES].to_dict("records"):
        Patient(**row).validate()
    if set(frame.outcome.unique()) != {0, 1} or frame.outcome.value_counts().min() < 10:
        raise ValueError("Cần hai nhãn outcome=0/1, ít nhất 10 mẫu khác nhau mỗi nhãn.")
    x_train, x_test, y_train, y_test = train_test_split(frame[FEATURES], frame.outcome,
        test_size=.25, stratify=frame.outcome, random_state=42)
    if groups is not None and external_test is None:
        train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
                                  .split(frame[FEATURES], frame.outcome, groups))
        x_train, x_test = frame[FEATURES].iloc[train_idx], frame[FEATURES].iloc[test_idx]
        y_train, y_test = frame.outcome.iloc[train_idx], frame.outcome.iloc[test_idx]
        if y_train.nunique() != 2 or y_test.nunique() != 2:
            raise ValueError("Chia theo participant_id không đủ hai nhãn; cần thêm bệnh nhân mỗi lớp hoặc tập test riêng.")
    if external_test is not None:
        external_test = external_test[needed].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(external_test.to_numpy()).all() or set(external_test.outcome.unique()) != {0, 1}:
            raise ValueError("Tập kiểm tra ngoài cần dữ liệu hữu hạn và đủ nhãn 0/1.")
        for row in external_test[FEATURES].to_dict("records"):
            Patient(**row).validate()
        x_train, y_train = frame[FEATURES], frame.outcome
        x_test, y_test = external_test[FEATURES], external_test.outcome
    model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16),
        max_iter=600, random_state=42, early_stopping=True))
    model.fit(x_train, y_train, mlpclassifier__sample_weight=compute_sample_weight("balanced", y_train))
    predicted = model.predict_proba(x_test)[:, 1]
    metadata = {"synthetic": synthetic, "features": FEATURES, "architecture": [4, 32, 16, 1],
        "train_size": len(x_train), "test_size": len(x_test),
        "accuracy": float(accuracy_score(y_test, predicted >= .5)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted >= .5)),
        "precision": float(precision_score(y_test, predicted >= .5, zero_division=0)),
        "sensitivity": float(recall_score(y_test, predicted >= .5, zero_division=0)),
        "f1": float(f1_score(y_test, predicted >= .5, zero_division=0)),
        "brier_score": float(brier_score_loss(y_test, predicted)),
        "roc_auc": float(roc_auc_score(y_test, predicted)),
        "confusion_matrix": confusion_matrix(y_test, predicted >= .5).tolist(),
        "dataset_sha256": dataset_hash,
        "evaluation": "external_holdout" if external_test is not None else "grouped_patient_holdout" if groups is not None else "random_stratified_holdout",
        "target_description": manifest.get("target_description", "Nhãn outcome của bộ dữ liệu do người dùng cung cấp."),
        "dataset_manifest": manifest,
        "training_ranges": {key: [float(x_train[key].min()), float(x_train[key].max())] for key in FEATURES},
        "limitation": "Chưa kiểm định lâm sàng ở Việt Nam. Điểm đầu ra chưa hiệu chuẩn. Không dùng để ước lượng tỷ lệ mắc bệnh dân số. Dữ liệu nhiều lần đo cần chia theo bệnh nhân."}
    tn, fp, fn, tp = metadata["confusion_matrix"][0] + metadata["confusion_matrix"][1]
    metadata["specificity"] = tn / (tn + fp) if tn + fp else 0
    metadata["majority_baseline_accuracy"] = float(y_test.value_counts(normalize=True).max())
    prepare()
    # Only load this local, self-produced artifact. Never accept uploaded pickle files.
    with (ARTIFACTS / "model.tmp").open("wb") as f:
        pickle.dump({"model": model, "metadata": metadata}, f)
    (ARTIFACTS / "model.tmp").replace(ARTIFACTS / "model.pkl")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

def predict(patient):
    import pandas as pd
    path = ARTIFACTS / "model.pkl"
    if not path.exists():
        return {"available": False, "reason": "Chưa huấn luyện mô hình. Chạy lệnh train theo README."}
    if any(getattr(patient, key) is None for key in FEATURES):
        return {"available": False, "reason": "Mô hình cần đủ glucose đói, HbA1c, tuổi và BMI."}
    if not patient.fasting_confirmed or patient.pregnant or (patient.age is not None and patient.age < 18):
        return {"available": False, "reason": "Ngoài phạm vi mô hình hoặc chưa xác nhận nhịn ăn."}
    with path.open("rb") as f:
        artifact = pickle.load(f)
    score = float(artifact["model"].predict_proba(pd.DataFrame(
        [[getattr(patient, k) for k in FEATURES]], columns=FEATURES))[0, 1])
    return {"available": True, "score": score, "class": int(score >= .5),
            "synthetic": artifact["metadata"]["synthetic"],
            "target_description": artifact["metadata"].get("target_description", "Nhãn outcome từ dataset."),
            "evaluation": artifact["metadata"].get("evaluation", "random_holdout"),
            "outside_training_range": [key for key, (low, high) in artifact["metadata"].get("training_ranges", {}).items()
                                       if not low <= getattr(patient, key) <= high],
            "note": "Điểm đầu ra mạng nơ-ron chưa hiệu chuẩn; không phải xác suất mắc bệnh đã kiểm chứng."}
