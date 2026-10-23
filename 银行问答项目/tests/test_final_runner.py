"""Synthetic-only tests: never retrieve or generate held-out questions."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_final_test as runner
from icbc_rag.prompts import build_messages
from icbc_rag.generation import ProviderError


class FakeProvider:
    def __init__(self, fail=False):
        self.attempts = 0
        self.fail = fail
    def complete(self, messages):
        self.attempts += 1
        if self.fail:
            raise ProviderError('synthetic failure')
        return {'content':'synthetic raw output', 'usage':{'total_tokens':1}}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve()
        self.doc = {'chunk_id':'SYNTHETIC', 'text':'这是完全合成的证据原文。',
                    'source_title':'合成来源','source_version':'v1','scope':'合成',
                    'citation_locator':'合成段落'}
        self.rows = [{'query_id':'SYN01','query':'合成问题甲','reference_answer':'GOLD_SENTINEL'},
                     {'query_id':'SYN02','query':'合成问题乙','reference_answer':'GOLD_SENTINEL'}]
        self.state = {'reserved_ids':[], 'completed_ids':[], 'pending_query_id':None,
                      'external_model_calls':0}

    def client(self, provider=None, persist=lambda:None, limit=16):
        return runner.JournaledClient(provider or FakeProvider(), self.folder,
                                       self.state, persist, [self.doc], limit)

    def unused_plan(self, changes=None):
        """Do not make tests depend on whether the real formal batch has run."""
        original_load=runner.load
        original_path=runner.project_path
        plan=copy.deepcopy(original_load(ROOT/runner.PLAN))
        plan.update(approval=None,execution_started=False,external_model_calls=0,final_test_queries_run=0)
        plan.update(changes or {})
        context=ExitStack()
        context.enter_context(patch.object(runner,'load',side_effect=lambda path:plan if Path(path)==ROOT/runner.PLAN else original_load(path)))
        context.enter_context(patch.object(runner,'project_path',side_effect=lambda path:self.folder/'unused_run' if str(path)==runner.RUN else original_path(path)))
        return context

    def test_preflight_hashes_only_never_reads_test_queries(self):
        with self.unused_plan(),patch.object(runner, 'read_jsonl', side_effect=AssertionError('held-out read')):
            _, plan, _, _, _ = runner.preflight(runner.PLAN)
        self.assertEqual(plan['query_count'], 16)
        self.assertIsNone(plan['approval'])

    def test_no_approval_prevents_question_loading(self):
        with self.unused_plan(),patch.object(runner, 'read_jsonl', side_effect=AssertionError('held-out read')):
            with self.assertRaisesRegex(ValueError, '明确授权'):
                runner.execute_plan(runner.PLAN)

    def test_approval_is_bound_to_harness_and_freeze(self):
        plan = {'freeze_manifest_sha256':'frozen','approval':{
            'actor':'human_user','approved':True,'max_new_model_calls':16,
            'source_message':'synthetic explicit authorization','approved_at':'synthetic',
            'runner_sha256':runner.file_hash(Path(runner.__file__)),
            'freeze_manifest_sha256':'frozen'}}
        runner.require_approval(plan)
        plan['approval']['runner_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            runner.require_approval(plan)

    def test_changed_limits_or_ids_rejected(self):
        for key, value in [('max_new_model_calls',17), ('retries',1), ('query_ids',['EV003'])]:
            with self.unused_plan({key:value}):
                with self.assertRaises(ValueError):
                    runner.preflight(runner.PLAN)

    def test_duplicate_and_call_cap_prevent_provider_requests(self):
        client = self.client(limit=1)
        client.query = self.rows[0]
        client.complete(build_messages(client.query['query'], [self.doc]))
        with self.assertRaises(ValueError):
            client.complete(build_messages(client.query['query'], [self.doc]))
        client.query = self.rows[1]
        with self.assertRaises(ValueError):
            client.complete(build_messages(client.query['query'], [self.doc]))
        self.assertEqual(client.attempts, 1)

    def test_prompt_gold_or_changed_evidence_rejected_before_send(self):
        client = self.client()
        client.query = self.rows[0]
        messages = build_messages(client.query['query'], [self.doc])
        payload = json.loads(messages[1]['content'])
        payload['reference_answer'] = 'GOLD_SENTINEL'
        messages[1]['content'] = json.dumps(payload, ensure_ascii=False)
        with self.assertRaises(ValueError):
            client.complete(messages)
        self.assertEqual(client.attempts, 0)

    def test_failed_durable_reservation_does_not_send(self):
        def fail():
            raise OSError('synthetic disk failure')
        client = self.client(persist=fail)
        client.query = self.rows[0]
        with self.assertRaises(OSError):
            client.complete(build_messages(client.query['query'], [self.doc]))
        self.assertEqual(client.attempts, 0)

    def test_raw_provider_output_survives_metadata_failure(self):
        def persist():
            if client.attempts:
                raise OSError('synthetic metadata failure')
        client = self.client(persist=persist)
        client.query = self.rows[0]
        with self.assertRaises(OSError):
            client.complete(build_messages(client.query['query'], [self.doc]))
        self.assertEqual(runner.load(self.folder / 'SYN01_provider.json')['content'], 'synthetic raw output')

    def batch(self, provider, reject=False):
        client = self.client(provider)
        doc = self.doc
        class Baseline:
            def search(self, query, top_k):
                return {'query':query,'results':[]}
        def engine(row, index, encoder, mode, proxy):
            self.assertEqual(set(row), {'query_id','query'})
            before = proxy.attempts
            try:
                output = proxy.complete(build_messages(row['query'], [doc]))
                status = 'rejected_by_citation_or_schema_check' if reject else runner.ACCEPTED
            except ProviderError:
                output, status = None, 'provider_error'
            return {'query_id':row['query_id'], 'provider_response':output,
                    'generation_status':status,'external_model_calls':proxy.attempts-before}
        return runner.run_batch(self.rows, self.folder, engine, Baseline(), None, None,
                                client, self.state, lambda:None)

    def test_provider_failure_stops_without_second_request(self):
        provider = FakeProvider(fail=True)
        self.assertFalse(self.batch(provider))
        self.assertEqual(provider.attempts, 1)
        self.assertEqual(self.state['completed_ids'], ['SYN01'])

    def test_invalid_response_stops_and_preserves_raw(self):
        provider = FakeProvider()
        self.assertFalse(self.batch(provider, reject=True))
        self.assertEqual(provider.attempts, 1)
        self.assertTrue((self.folder / 'SYN01_provider.json').is_file())

    def test_success_strips_labels_and_records_each_once(self):
        provider = FakeProvider()
        self.assertTrue(self.batch(provider))
        self.assertEqual(provider.attempts, 2)
        self.assertEqual(self.state['reserved_ids'], ['SYN01','SYN02'])
        self.assertNotIn('GOLD_SENTINEL', (self.folder / 'SYN01_request.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
