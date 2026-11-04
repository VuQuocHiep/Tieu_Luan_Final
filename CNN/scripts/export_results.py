"""Generate a reproducible markdown table and sample/distribution outputs from local Kaggle CSV.
Run after python train_all.py (this script is not a model training step).
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
from shared.config import CONFIG,ART,ROOT
from shared.datasets import class_distribution,preview_data


def main():
    rows=[]
    for key,spec in CONFIG.items():
        for method in ('scratch','keras','pytorch'):
            path=ART/f'{key}_{method}.json'
            if path.exists():
                d=json.loads(path.read_text(encoding='utf-8'))
                rows.append({'Dataset':spec['short'],'Framework':method,'Accuracy':round(d['accuracy'],4),
                             'Precision':round(d['precision_macro'],4),'Recall':round(d['recall_macro'],4),
                             'Macro-F1':round(d['f1_macro'],4),'Test samples':d['n_test']})
    if not rows:
        print('Chưa có metrics. Chạy python train_all.py trước.');return
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'CNN_3_apps_metrics.csv',index=False,encoding='utf-8-sig')
    lines=['# Kết quả thực nghiệm CNN từ CSV Kaggle đã chạy thực tế','','> Kết quả được lấy từ JSON tạo bởi train_all.py; không phải dữ liệu thử.','',df.to_markdown(index=False),'']
    for key,spec in CONFIG.items():
        try:
            _,labels,samples=preview_data(key,5)
            dist=class_distribution(key)
        except (FileNotFoundError,ValueError):continue
        lines.extend([f'## {spec["short"]}',f'Nguồn: {spec["url"]}','',
                      '### 5 mẫu CSV đầu tiên (một số cột)',samples.to_markdown(index=False),
                      '', '### Phân bố nhãn của tập train gốc',
                      pd.DataFrame({'Lớp':list(dist),'Số mẫu':list(dist.values())}).to_markdown(index=False),''])
    (out/'KET_QUA_CAN_DAN_VAO_CHUONG_3.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Đã xuất reports/KET_QUA_CAN_DAN_VAO_CHUONG_3.md và reports/CNN_3_apps_metrics.csv')

if __name__=='__main__':main()
