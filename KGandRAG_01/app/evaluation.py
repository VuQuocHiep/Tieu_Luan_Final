"""Objective offline retrieval evaluation. NOT a medical accuracy certification."""
from __future__ import annotations
import json
from .engine import ROOT, KnowledgeEngine


def evaluate(engine: KnowledgeEngine, mode: str='hybrid', k: int=3) -> dict:
    data=json.loads((ROOT/'eval'/'questions.json').read_text(encoding='utf-8'))
    rows=[]
    for item in data:
        retrieved,_=engine.retrieve(item['question'],mode=mode,top_k=k)
        pred=[r.doc['id'] for r in retrieved]
        gold=set(item['gold_docs'])
        intersect=gold.intersection(pred)
        first_rank=next((idx for idx,d in enumerate(pred,1) if d in gold),None)
        rows.append({'id':item['id'],'question':item['question'],
                     'expected':sorted(gold),'retrieved':pred,
                     'hit':int(bool(intersect)),
                     'recall':len(intersect)/len(gold),
                     'precision':len(intersect)/k,
                     'rr':1/first_rank if first_rank else 0.0})
    n=len(rows)
    return {'mode':mode,'k':k,'n_questions':n,'hit_at_k':round(sum(r['hit'] for r in rows)/n,4),
            'recall_at_k':round(sum(r['recall'] for r in rows)/n,4),
            'precision_at_k':round(sum(r['precision'] for r in rows)/n,4),
            'mrr_at_k':round(sum(r['rr'] for r in rows)/n,4),'rows':rows,
            'note':'Chỉ đo chất lượng truy xuất so với tập nhãn mẫu; KHÔNG chứng minh độ chính xác y khoa hoặc chất lượng sinh câu trả lời.'}


def compare(engine: KnowledgeEngine,k: int=3) -> list[dict]:
    return [evaluate(engine,m,k) for m in ('vector','graph','hybrid')]
