"""Kiểm thử chế độ học thuật chứng khoán & vàng, không dùng live market API."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.engine import KnowledgeEngine, norm
from app.evaluation import compare

client = TestClient(app)
engine = KnowledgeEngine()


def test_seeds_are_consistent():
    ids = {n['id'] for n in engine.nodes}
    d_ids = {d['id'] for d in engine.docs}
    assert len(ids) == len(engine.nodes) == 52
    assert len(d_ids) == len(engine.docs) == 40
    assert len(engine.edges) == 74
    assert all(set(d['entities']) <= ids for d in engine.docs)
    assert all(e['source'] in ids and e['target'] in ids for e in engine.edges)
    assert all(set(e['doc_ids']) <= d_ids for e in engine.edges)
    assert all(d['source_url'].startswith('https://') for d in engine.docs)


def test_normalize_vietnamese():
    assert norm('Đầu tư chứng khoán và vàng!') == 'dau tu chung khoan va vang'


def test_stock_pe_earnings_retrieval():
    hits, _ = engine.retrieve('Hệ số P/E và EPS có ý nghĩa gì?', top_k=3)
    assert 'D13' in [hit.doc['id'] for hit in hits]


def test_gold_real_yields_retrieval():
    hits, _ = engine.retrieve('Lãi suất thực ảnh hưởng giá vàng như thế nào?', top_k=3)
    assert 'D28' in [hit.doc['id'] for hit in hits]


def test_stocks_vs_gold_retrieval():
    hits, _ = engine.retrieve('Điểm khác nhau khi đầu tư vàng và cổ phiếu?', top_k=3)
    assert 'D33' in [hit.doc['id'] for hit in hits]


def test_graph_links():
    data = engine.subgraph('lãi suất thực')
    assert any(n['id'] == 'realyield' for n in data['nodes'])
    assert data['edges']
    assert any(e['source'] == 'realyield' or e['target'] == 'realyield' for e in data['edges'])


def test_out_of_domain_abstention():
    hits, _ = engine.retrieve('viết ứng dụng React quản lý tour du lịch sân bay')
    assert not hits
    hits, _ = engine.retrieve('Dữ liệu thời tiết ngoài hành tinh')
    assert not hits


def test_live_quote_safety():
    answer = engine.answer('Giá vàng hôm nay bao nhiêu?')
    assert answer['answer_method'] == 'safety_override'
    assert answer['warnings']
    assert 'không lấy được giá' in answer['answer'].lower()


def test_stock_live_quote_safety():
    answer = engine.answer('Giá cổ phiếu hôm nay thế nào?')
    assert answer['answer_method'] == 'safety_override'


def test_no_personal_investment_advice():
    answer = engine.answer('Tôi nên mua cổ phiếu nào ngay bây giờ?')
    assert answer['answer_method'] == 'safety_override'
    assert 'không thể xác định' in answer['answer'].lower()


def test_no_api_default(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'none')
    answer = engine.answer('P/E và EPS là gì?')
    assert answer['answer_method'] == 'extractive'
    assert answer['sources']
    assert '[D' in answer['answer']


def test_health_and_api():
    assert client.get('/api/health').json()['status'] == 'ok'
    response = client.post('/api/chat', json={'question': 'ETF vàng là gì?', 'mode': 'hybrid', 'top_k': 3})
    assert response.status_code == 200
    assert response.json()['sources']
    assert client.get('/api/graph').status_code == 200
    assert client.get('/api/stats').json()['documents'] == 40
    assert client.get('/').status_code == 200


def test_invalid_payload():
    assert client.post('/api/chat', json={'question': 'abc', 'top_k': 999}).status_code == 422
    assert client.post('/api/chat', json={'question': 'Cổ phiếu là gì?', 'mode': 'invalid'}).status_code == 422


def test_evaluation_structure():
    res = client.get('/api/evaluation?k=3')
    assert res.status_code == 200
    scores = res.json()['results']
    assert len(scores) == 3
    assert all(0 <= r['hit_at_k'] <= 1 and len(r['rows']) == 40 for r in scores)


def test_modes_and_top_k():
    for mode in ('vector', 'graph', 'hybrid'):
        hits, _ = engine.retrieve('Vàng ETF có chi phí gì?', mode=mode, top_k=2)
        assert len(hits) <= 2
        assert all(0 <= h.score <= 1 for h in hits)
