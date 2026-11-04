"""python -m scripts.run_evaluation --k 3: saves comparable JSON + CSV reports."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from app.engine import ROOT,get_engine
from app.evaluation import compare

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--k',type=int,default=3)
    args=parser.parse_args()
    if not 1 <= args.k <= 8:
        parser.error('--k phải từ 1 đến 8')
    results=compare(get_engine(),args.k)
    target=ROOT/'eval'/'reports';target.mkdir(parents=True,exist_ok=True)
    payload={'run_utc':datetime.now(timezone.utc).isoformat(),'results':results}
    (target/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    with (target/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Mode','K','Questions','Hit@K','Recall@K','Precision@K','MRR@K'])
        for r in results:
            writer.writerow([r['mode'],r['k'],r['n_questions'],r['hit_at_k'],r['recall_at_k'],r['precision_at_k'],r['mrr_at_k']])
    with (target/'details.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Mode','ID','Question','Gold docs','Retrieved docs','Hit','Recall','Precision','RR'])
        for r in results:
            for row in r['rows']:
                writer.writerow([r['mode'],row['id'],row['question'],';'.join(row['expected']),';'.join(row['retrieved']),row['hit'],row['recall'],row['precision'],row['rr']])
    for r in results:
        print(f"{r['mode']:7s} Hit@{args.k}={r['hit_at_k']:.1%} Recall={r['recall_at_k']:.1%} MRR={r['mrr_at_k']:.3f}")
    print('Wrote',target)
if __name__=='__main__':main()
