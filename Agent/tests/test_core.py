import csv
import io
import tempfile
import unittest
from pathlib import Path

from app1_hoi_dap.rag_core import BM25Index, Chunk, build_chunks, sample_file
from app2_quan_ly_kho import warehouse_core as db
from app3_multi_agent.agents import DataAgent, AnalysisAgent, ReportAgent, MultiAgentCoordinator, SAMPLE_CSV
from common.agent_core import AgentRunner, Tool


class TestDocumentAgent(unittest.TestCase):
    def test_txt_chunks_and_search(self):
        chunks = build_chunks([sample_file()])
        self.assertTrue(chunks)
        hits = BM25Index(chunks).search('Quy định xuất kho và tồn tối thiểu')
        self.assertTrue(hits)
        self.assertIn('huong_dan_kho.txt', hits[0]['source'])

    def test_search_empty_and_not_found(self):
        index = BM25Index([Chunk('test.txt', 1, 'Nhiệt độ hàng hóa', 1)])
        self.assertEqual(index.search(''), [])
        self.assertEqual(index.search('xylophone'), [])


class TestWarehouseAgent(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = Path(self.dir.name) / 'warehouse_test.db'
        db.initialize(self.db)

    def tearDown(self):
        self.dir.cleanup()

    def test_warehouse_tools(self):
        items = db.low_stock(self.db)
        self.assertGreater(len(items), 0)
        self.assertTrue(all(x['quantity'] <= x['min_stock'] for x in items))
        self.assertEqual(db.find_product('SP001', self.db)[0]['name'], 'Bàn phím cơ')
        self.assertIn('gas_adc', db.sensor_status(self.db))
        self.assertGreater(db.sales_summary(db_path=self.db)['products'][0]['units'], 0)
        self.assertTrue(db.reorder_suggestions(db_path=self.db)['suggestions'])

    def test_drafts_explicit(self):
        self.assertEqual(db.get_drafts(self.db), [])
        draft_id = db.add_draft('SP001', 20, self.db)
        self.assertEqual(draft_id, 1)
        self.assertEqual(db.get_drafts(self.db)[0]['status'], 'PENDING_REVIEW')
        with self.assertRaises(ValueError):
            db.add_draft('NOT_VALID', 5, self.db)


class TestMultiAgents(unittest.TestCase):
    def test_three_agents_pipeline(self):
        output = MultiAgentCoordinator().run(SAMPLE_CSV.read_bytes())
        self.assertEqual(len(output['trace']), 3)
        self.assertEqual(output['data']['included_rows'], 42)
        metrics = output['analysis']['metrics']
        self.assertGreater(metrics['revenue_vnd'], 0)
        self.assertGreater(metrics['estimated_gross_profit_vnd'], 0)
        self.assertIn('BÁO CÁO', output['report'])

    def test_bad_csv(self):
        with self.assertRaises(ValueError):
            DataAgent().run(b'col1,col2\na,b')

    def test_invalid_row_skipped(self):
        csv_data = ('date,sku,product,category,quantity,unit_price,unit_cost\n'
                    '2026-09-01,X,San pham,Nhom,2,100,50\n'
                    '2026-09-02,X,San pham,Nhom,INVALID,100,50\n')
        data = DataAgent().run(csv_data.encode())
        self.assertEqual(data['included_rows'], 1)
        self.assertEqual(len(data['skipped_rows']), 1)


class MockClient:
    def __init__(self):
        self.n = 0

    def chat(self, messages, tools=None):
        self.n += 1
        if self.n == 1:
            return {'role': 'assistant', 'content': '', 'tool_calls': [{
                'function': {'name': 'add', 'arguments': {'a': 2, 'b': 3}}}]}
        return {'role': 'assistant', 'content': 'Kết quả là 5.'}


class TestToolAgent(unittest.TestCase):
    def test_tool_loop(self):
        agent = AgentRunner(MockClient(), 'Use tools', [Tool('add', 'Cộng hai số',
            {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}},
            lambda a, b: {'sum': a + b})])
        result = agent.run('2 + 3 = ?')
        self.assertEqual(result['trace'][0]['result']['sum'], 5)
        self.assertEqual(result['answer'], 'Kết quả là 5.')


if __name__ == '__main__':
    unittest.main()
