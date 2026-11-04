import argparse
import json
import sys
from pathlib import Path
from .config import OUTPUTS, prepare
from .schema import Patient

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Multi-agent diabetes research system")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate", help="Tạo ảnh bảng mô phỏng")
    sub.add_parser("demo-data", help="Tạo CSV giả lập")
    importer = sub.add_parser("import-db", help="Nạp CSV vào MySQL")
    importer.add_argument("csv")
    importer.add_argument("--dataset", default="clinical")
    training = sub.add_parser("train")
    training.add_argument("--csv")
    training.add_argument("--dataset")
    training.add_argument("--synthetic", action="store_true")
    ocr = sub.add_parser("ocr")
    ocr.add_argument("image")
    sub.add_parser("init-db")
    sub.add_parser("prepare-nhanes", help="Tải dữ liệu CDC NHANES vào MySQL và huấn luyện theo chu kỳ độc lập")
    args = parser.parse_args()
    prepare()
    if args.command == "generate":
        from .imaging import render_table
        for name, glucose, a1c in [("reference", 90, 5.2), ("prediabetes", 112, 6.0), ("suspected", 145, 7.1)]:
            path = OUTPUTS / f"table_{name}.png"
            path.write_bytes(render_table(Patient(glucose, a1c, 45, 27, True)))
            print(path)
    elif args.command == "demo-data":
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(42)
        n = 1200
        frame = pd.DataFrame({"fasting_glucose": rng.uniform(70, 230, n).round(1),
            "hba1c": rng.uniform(4, 10, n).round(1), "age": rng.integers(18, 85, n),
            "bmi": rng.uniform(18, 42, n).round(1)})
        frame["outcome"] = ((frame.fasting_glucose >= 126) | (frame.hba1c >= 6.5)).astype(int)
        path = OUTPUTS / "synthetic_training.csv"
        frame.to_csv(path, index=False)
        print(f"SYNTHETIC ONLY: {path}; nhãn sinh từ quy tắc, không đo năng lực lâm sàng.")
    elif args.command == "train":
        from .learning import train
        if bool(args.csv) == bool(args.dataset):
            parser.error("Chọn đúng một trong --csv hoặc --dataset")
        frame = None
        metadata = None
        if args.dataset:
            from .storage import Database
            db = Database()
            frame = db.load_samples(args.dataset)
            metadata = db.dataset_metadata(args.dataset)
        metrics = train(args.csv, args.synthetic, frame, dataset_metadata=metadata)
        if args.dataset:
            db.save_model_metrics(metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    elif args.command == "import-db":
        import pandas as pd
        from .storage import Database
        from .config import FEATURES
        frame = pd.read_csv(args.csv)
        if not set(FEATURES + ["outcome"]).issubset(frame.columns):
            parser.error("CSV thiếu các cột đặc trưng hoặc outcome")
        if len(args.dataset) > 64:
            parser.error("Tên dataset tối đa 64 ký tự")
        Database().import_samples(frame[FEATURES + ["outcome"]], args.dataset)
        print(f"Đã lưu {len(frame)} mẫu vào MySQL.")
    elif args.command == "ocr":
        from .imaging import OCRAgent
        print(json.dumps(OCRAgent().read(Path(args.image).read_bytes()), indent=2, ensure_ascii=False))
    elif args.command == "init-db":
        from .storage import Database
        Database()
        print("Đã kết nối và tạo bảng MySQL.")
    elif args.command == "prepare-nhanes":
        from .datasets import prepare_nhanes
        from .storage import Database
        from .learning import train
        db = Database()
        for year in (2015, 2017):
            _, frame, metadata = prepare_nhanes(year)
            db.import_samples(frame, f"nhanes_{year}", metadata)
            print(f"MySQL: NHANES {year}, {len(frame)} complete adult rows")
        training = db.load_samples("nhanes_2015")
        testing = db.load_samples("nhanes_2017")
        manifest = db.dataset_metadata("nhanes_2015")
        manifest["test_dataset"] = db.dataset_metadata("nhanes_2017")
        metrics = train(frame=training, external_test=testing, dataset_metadata=manifest)
        db.save_model_metrics(metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
