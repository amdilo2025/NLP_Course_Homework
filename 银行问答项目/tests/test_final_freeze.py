"""Offline checks for reference-only corrections and frozen artifact integrity."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.freeze_final_configuration import apply_reference_errata, text_hash, verify_environment, verify_freeze
from icbc_rag.common import file_hash


class FreezeTests(unittest.TestCase):
    def fixture_errata(self):
        record = {'query_id':'EV036','query':'合成题目，不是真实测试输入','split':'test','answer_type':'可回答',
                  'reference_answer':'旧参考合成文字','reference_ids':['R19']}
        errata = {'corrections':{'EV036':{'old_reference_answer_sha256':text_hash(record['reference_answer']),
                                        'new_reference_answer':'新参考合成文字','answer_type_unchanged':'可回答'}}}
        return record, errata

    def test_reference_correction_preserves_input_question_split_and_type(self):
        record, errata = self.fixture_errata(); before = copy.deepcopy(record)
        updated = apply_reference_errata(record, errata)
        self.assertEqual(record, before)
        for key in ['query','query_id','split','answer_type','reference_ids']:
            self.assertEqual(updated[key], before[key])
        self.assertEqual(updated['reference_answer'], '新参考合成文字')
        self.assertEqual(updated['reference_answer_original'], before['reference_answer'])

    def test_unaffected_question_returns_unchanged_copy(self):
        record, errata = self.fixture_errata(); record['query_id'] = 'EV035'
        self.assertEqual(apply_reference_errata(record, errata), record)
        self.assertIsNot(apply_reference_errata(record, errata), record)

    def test_wrong_original_reference_or_split_cannot_receive_erratum(self):
        record, errata = self.fixture_errata()
        for key, value in [('reference_answer','different'),('split','development'),('answer_type','资料不足')]:
            changed = copy.deepcopy(record); changed[key] = value
            with self.assertRaises(ValueError): apply_reference_errata(changed, errata)

    def config(self):
        return {'api':{'base_url_env':'TEST_BASE','model_env':'TEST_MODEL',
                       'default_base_url':'https://api.deepseek.com','default_model':'synthetic_model'}}

    def test_environment_cannot_silently_change_endpoint_or_model(self):
        cfg = self.config()
        verify_environment(cfg, {})
        verify_environment(cfg, {'TEST_BASE':'https://api.deepseek.com/'})
        for env in [{'TEST_MODEL':'other'}, {'TEST_BASE':'https://other.invalid'}]:
            with self.assertRaises(ValueError): verify_environment(cfg, env)

    def frozen_fixture(self, root):
        folder = root/'frozen';folder.mkdir()
        (root/'engine.py').write_text('original',encoding='utf-8')
        (folder/'rag_config.json').write_text(json.dumps(self.config()),encoding='utf-8')
        manifest = {'state':'frozen','files':{'rag_config.json':file_hash(folder/'rag_config.json')},
                    'protected_files':{'engine.py':file_hash(root/'engine.py')}}
        (folder/'freeze_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        return folder

    def test_post_freeze_code_or_config_drift_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder = self.frozen_fixture(root)
            with patch('tools.freeze_final_configuration.ROOT', root), patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('禁止网络')):
                self.assertEqual(verify_freeze(folder, {})['state'], 'frozen')
                (root/'engine.py').write_text('changed',encoding='utf-8')
                with self.assertRaises(ValueError): verify_freeze(folder,{})
                (root/'engine.py').write_text('original',encoding='utf-8')
                (folder/'rag_config.json').write_text('{}',encoding='utf-8')
                with self.assertRaises(ValueError): verify_freeze(folder,{})

    def test_protected_paths_must_remain_inside_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder = self.frozen_fixture(root)
            p = folder/'freeze_manifest.json'; m = json.loads(p.read_text(encoding='utf-8'))
            m['protected_files'] = {'../outside.py':'0'*64};p.write_text(json.dumps(m),encoding='utf-8')
            with patch('tools.freeze_final_configuration.ROOT', root):
                with self.assertRaises(ValueError): verify_freeze(folder,{})


if __name__ == '__main__':
    unittest.main()
