import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.engine import KnowledgeEngine, norm
from app.evaluation import compare

client=TestClient(app)
engine=KnowledgeEngine()

def test_seeds_are_consistent():
    ids={n['id'] for n in engine.nodes}
    d_ids={d['id'] for d in engine.docs}
    assert len(ids)==len(engine.nodes) and len(d_ids)==len(engine.docs)
    assert all(set(d['entities']) <= ids for d in engine.docs)
    assert all(e['source'] in ids and e['target'] in ids for e in engine.edges)
    assert all(set(e['doc_ids']) <= d_ids for e in engine.edges)

def test_normalize_vietnamese():
    assert norm('Đường huyết và HbA1c')=='duong huyet va hba1c'

def test_retrieve_hba1c():
    hits,_=engine.retrieve('HbA1c 6.5% ngưỡng xét nghiệm',top_k=4)
    assert 'D08' in [h.doc['id'] for h in hits]

def test_graph_links():
    data=engine.subgraph('thận')
    assert any(n['id']=='kidney' for n in data['nodes'])
    assert len(data['edges'])>0

def test_ood_abstention():
    hits,_=engine.retrieve('viết ứng dụng React quản lý tour du lịch sân bay')
    assert not hits

def test_safety_emergency():
    answer=engine.answer('Người bị tiểu đường đang co giật và bất tỉnh thì làm sao?')
    assert answer['answer_method']=='safety_override'
    assert answer['warnings']
    assert 'cấp cứu' in answer['answer'].lower()

def test_safety_medication():
    answer=engine.answer('Tôi nên tăng liều insulin bao nhiêu?')
    assert answer['answer_method']=='safety_override'
    assert 'bác sĩ' in answer['warnings'][0]

def test_no_api_default(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER','none')
    response=engine.answer('Ngưỡng HbA1c là bao nhiêu?')
    assert response['answer_method']=='extractive'
    assert response['sources']
    assert '[D' in response['answer']

def test_health_and_api():
    assert client.get('/api/health').json()['status']=='ok'
    r=client.post('/api/chat',json={'question':'Bệnh tiểu đường là gì?','mode':'hybrid','top_k':3})
    assert r.status_code==200 and r.json()['sources']
    assert client.get('/api/graph').status_code==200
    assert client.get('/').status_code==200

def test_validation():
    assert client.post('/api/chat',json={'question':'hi','top_k':999}).status_code==422
    assert client.post('/api/chat',json={'question':'Tiểu đường là gì','mode':'invalid'}).status_code==422

def test_evaluation_structure():
    report=client.get('/api/evaluation?k=3')
    assert report.status_code==200
    scores=report.json()['results']
    assert len(scores)==3
    assert all(0<=s['hit_at_k']<=1 and len(s['rows'])==30 for s in scores)
