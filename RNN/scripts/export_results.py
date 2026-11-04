import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import pandas as pd
from shared.config import CONFIG, ARTIFACTS, REPORTS
from shared.train_runner import artifact_paths

rows=[]
for key in CONFIG:
    for method in ('scratch','keras','pytorch'):
        p=artifact_paths(key,method)['metrics']
        if p.exists():
            m=json.loads(p.read_text(encoding='utf-8'))
            rows.append({'Dataset':CONFIG[key]['short'],'Method':method,'Accuracy':m['accuracy'],'Macro-F1':m['macro_f1'],
                         'Precision':m['macro_precision'],'Recall':m['macro_recall'],'N test':m['n_test']})
if not rows:
    raise SystemExit('Chưa có metrics. Chạy python train_all.py trước.')
REPORTS.mkdir(parents=True,exist_ok=True)
df=pd.DataFrame(rows)
df.to_csv(REPORTS/'rnn_results.csv',index=False,encoding='utf-8-sig')
md='# Kết quả RNN trên dữ liệu Kaggle\n\n'+df.to_markdown(index=False,floatfmt='.4f')+'\n'
(REPORTS/'rnn_results.md').write_text(md,encoding='utf-8')
print(df.to_string(index=False))
print('\n[OK]',REPORTS/'rnn_results.csv')
print('[OK]',REPORTS/'rnn_results.md')
