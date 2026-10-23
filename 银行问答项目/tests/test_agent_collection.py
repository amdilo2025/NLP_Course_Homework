"""Use fake client/engine and synthetic responses to test paid-call limits offline."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.agent_development_collection import validate_plan, run_approved_plan
from icbc_baseline.bm25 import canonical_hash
from icbc_rag.common import load_config, file_hash


class AgentCollectionTests(unittest.TestCase):
    def fixture(self, root):
        config = load_config(ROOT / 'config/rag.json')
        session = root / 'outputs/session'
        (session / 'answers').mkdir(parents=True)
        test = root / 'heldout.jsonl'
        test.write_text('仅哈希，不作为运行输入', encoding='utf-8')
        info = {'session_fingerprint': 'fixture', 'test_sha256_before': file_hash(test), 'pending_attempts': []}
        queries = [{'query_id': 'EV003', 'query': '合成问题一'}, {'query_id': 'EV004', 'query': '合成问题二'}]
        plan = {'kind': 'bounded_development_collection', 'config_sha256': canonical_hash(config),
                'session_fingerprint': 'fixture', 'test_sha256': file_hash(test),
                'query_ids': ['EV003', 'EV004'], 'max_new_api_calls': 2,
                'retry_count': 0, 'stop_on_first_failed_result': True,
                'max_output_tokens_per_call': 1200, 'execution_started': False,
                'attempted_query_ids': [], 'completed_query_ids': [], 'external_model_calls': 0,
                'existing_answer_hashes': {}, 'approval': {'actor': 'human_user', 'approved': True,
                'source_message_summary': '合成授权夹具，不是真实授权', 'approved_at': 'fixture'}}
        return config, session, info, queries, test, plan

    def test_missing_approval_and_test_ids_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config, session, info, queries, test, plan = self.fixture(Path(folder))
            missing = copy.deepcopy(plan); missing['approval'] = None
            with self.assertRaises(ValueError): validate_plan(missing, config, queries, info, session)
            wrong = copy.deepcopy(plan); wrong['query_ids'][0] = 'EV036'
            with self.assertRaises(ValueError): validate_plan(wrong, config, queries, info, session)

    def test_execution_limit_and_restart_are_guarded(self):
        with tempfile.TemporaryDirectory() as folder:
            config, session, info, queries, test, plan = self.fixture(Path(folder))
            for key, value in [('max_new_api_calls', 3), ('retry_count', 1), ('execution_started', True)]:
                changed = copy.deepcopy(plan); changed[key] = value
                with self.assertRaises(ValueError): validate_plan(changed, config, queries, info, session)

    def run_fixture(self, folder, status, relative=False, fail_after_result=False):
        root = Path(folder)
        config, session, info, queries, test, plan = self.fixture(root)
        path = root / 'plan.json'
        path.write_text(json.dumps(plan), encoding='utf-8')
        executor = Mock(side_effect=lambda row, *args: {'query_id': row['query_id'], 'query': row['query'],
                        'generation_status': status, 'external_model_calls': 1, 'answer': None,
                        'synthetic_fixture': True})
        client = Mock(return_value=object())
        from tools.agent_development_collection import write_json as real_write
        def persist(dest, value):
            if fail_after_result and dest.parent.name == 'answers':
                raise OSError('合成落盘故障')
            return real_write(dest, value)
        with patch('tools.agent_development_collection.ROOT', root), \
             patch('tools.agent_development_collection.load_config', return_value=config), \
             patch('tools.agent_development_collection.prepare', return_value=(queries, {'test_file': test}, session, info, {})), \
             patch('tools.agent_development_collection.save_progress'), \
             patch('tools.agent_development_collection.write_json', side_effect=persist), \
             patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('禁止网络')):
            selected_path = Path('plan.json') if relative else path
            if fail_after_result:
                with self.assertRaises(OSError):
                    run_approved_plan(selected_path, client_factory=client,
                       engine_factory=lambda cfg: (object(), object()), execute_fn=executor)
                result = json.loads(path.read_text(encoding='utf-8'))
            else:
                result = run_approved_plan(selected_path, client_factory=client,
                           engine_factory=lambda cfg: (object(), object()), execute_fn=executor)
        return result, executor

    def test_first_failed_response_stops_batch_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            result, executor = self.run_fixture(folder, 'provider_error')
            self.assertEqual(executor.call_count, 1)
            self.assertEqual(result['external_model_calls'], 1)
            self.assertEqual(result['stopped_on_failed_query'], 'EV003')
            self.assertTrue(result['test_file_unchanged'])

    def test_success_uses_only_selected_queries_once(self):
        with tempfile.TemporaryDirectory() as folder:
            result, executor = self.run_fixture(folder, 'generated_citations_checked_semantics_unverified')
            self.assertEqual(executor.call_count, 2)
            self.assertEqual(result['external_model_calls'], 2)
            self.assertEqual(result['completed_query_ids'], ['EV003', 'EV004'])
            self.assertEqual(result['attempted_query_ids'], ['EV003', 'EV004'])

    def test_relative_plan_path_does_not_fail_after_paid_result(self):
        with tempfile.TemporaryDirectory() as folder:
            result, executor = self.run_fixture(folder, 'generated_citations_checked_semantics_unverified', relative=True)
            self.assertEqual(executor.call_count, 2)
            self.assertEqual(result['completed_query_ids'], ['EV003', 'EV004'])
            raw = Path(folder) / 'outputs/session/request_results/plan_EV003.json'
            self.assertTrue(raw.is_file())
            self.assertTrue(json.loads(raw.read_text(encoding='utf-8'))['synthetic_fixture'])

    def test_full_result_survives_later_metadata_write_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            result, executor = self.run_fixture(folder, 'generated_citations_checked_semantics_unverified', relative=True, fail_after_result=True)
            self.assertEqual(executor.call_count, 1)
            self.assertEqual(result['pending_request_ids'], ['EV003'])
            self.assertEqual(result['stopped_on_exception_type'], 'OSError')
            self.assertTrue((Path(folder) / 'outputs/session/request_results/plan_EV003.json').is_file())


if __name__ == '__main__':
    unittest.main()
