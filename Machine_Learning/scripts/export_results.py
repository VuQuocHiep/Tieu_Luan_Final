"""Sau khi huấn luyện CSV Kaggle, tạo bảng thực nghiệm để bổ sung vào báo cáo."""
from pathlib import Path
import pandas as pd
from shared.datasets import SOURCES, dataset
ROOT=Path(__file__).resolve().parents[1]
metrics=ROOT/'artifacts'/'metrics.csv'
if not metrics.exists():
    raise SystemExit('Chưa có kết quả. Chạy: python download_kaggle.py && python train_all.py')
df=pd.read_csv(metrics)
rows=[]
for key,spec in SOURCES.items():
    raw=dataset(key)
    counts=pd.Series(raw.target).value_counts().sort_index()
    rows.append(f'### {spec["vi_name"]}\n')
    rows.append(f'- Kaggle: {spec["source"]}\n- Kích cỡ: {len(raw.data)} mẫu × {raw.data.shape[1]} feature\n')
    rows.append('- Phân bố nhãn: '+', '.join(f'{raw.target_names[i]}: {int(counts.get(i,0))}' for i in range(len(raw.target_names)))+'\n')
    subset=df.loc[df.dataset==key, ['backend','train_size','val_size','test_size','accuracy','precision_macro','recall_macro','f1_macro','log_loss','train_seconds']].copy()
    rows.append(subset.to_markdown(index=False,floatfmt='.4f')+'\n')
    rows.append('Nhận xét: ba cách lập trình cài cùng thuật toán; đối chiếu sự chênh lệch Accuracy, F1, thời gian chạy và xem ma trận nhầm lẫn trong app.\n')
out=ROOT/'reports'/'ket_qua_thuc_nghiem_Kaggle.md'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text('# KẾT QUẢ THỰC NGHIỆM TRÊN CSV KAGGLE\n\n'+'\n'.join(rows),encoding='utf-8')
print('Đã tạo:',out)
