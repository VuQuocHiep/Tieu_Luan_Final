import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import CONFIG, REPORTS
from shared.datasets import dataset_summary
import json

REPORTS.mkdir(parents=True, exist_ok=True)
lines=['# Khảo sát 3 dataset Kaggle cho Chương 4 RNN','']
for key,spec in CONFIG.items():
    s=dataset_summary(key)
    lines += [f"## {spec['title']}",f"- Link: {spec['url']}",f"- Kích thước: {s['counts']}",f"- Phân bố: {s['distribution']}",'','Mẫu:','']
    for row in s['samples']:
        lines.append(f"- **{row['label']}**: {row['text'][:240]}")
    lines.append('')
(REPORTS/'dataset_summary.md').write_text('\n'.join(lines),encoding='utf-8')
print('[OK]',REPORTS/'dataset_summary.md')
