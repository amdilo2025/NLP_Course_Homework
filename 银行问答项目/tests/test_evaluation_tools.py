"""Offline evaluation-tool checks; provider constructors and requests are forbidden."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
if (ROOT/'.rag_packages').is_dir():sys.path.insert(0,str(ROOT/'.rag_packages'))
from tools.compare_retrievers import validate_ks
from tools.manual_rag_review import generation_allowed,matching_cache,main as manual_main
from icbc_rag.common import load_config,read_jsonl

class EvaluationToolTests(unittest.TestCase):
    def test_parameter_values_are_explicit_and_valid(self):
        self.assertEqual(validate_ks([1,3,5],24),[1,3,5])
        for values in [[],[3,3],[0],[25],[True]]:
            with self.assertRaises(ValueError):validate_ks(values,24)

    def test_only_exact_manual_generate_command_can_send(self):
        self.assertTrue(generation_allowed('生成'))
        for text in ['','继续','g','下一题','GEN']:
            self.assertFalse(generation_allowed(text))
        self.assertFalse(generation_allowed('生成',offline=True))
        self.assertFalse(generation_allowed('生成',already_saved=True))
        self.assertFalse(generation_allowed('生成',pending=True))

    def test_real_cached_answer_matches_only_original_configuration(self):
        folder=ROOT/'outputs/rag_runs/20261020T200036_117370'
        config=load_config(ROOT/'config/rag.json')
        meta=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
        row=read_jsonl(folder/'results.jsonl')[0]
        docs=read_jsonl(ROOT/'data/processed/corpus.jsonl')
        self.assertTrue(matching_cache(row,meta,row['query'],config,docs))
        changed=copy.deepcopy(config);changed['top_k']=1
        self.assertFalse(matching_cache(row,meta,row['query'],changed,docs))
        self.assertFalse(matching_cache(row,meta,'不同问题',config,docs))
        service=load_config(ROOT/'config/service_rag.json')
        self.assertFalse(matching_cache(row,meta,row['query'],service,docs))

    def test_cached_answer_and_prompt_tampering_are_rejected(self):
        folder=ROOT/'outputs/rag_runs/20261020T200036_117370'
        config=load_config(ROOT/'config/rag.json')
        meta=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
        row=read_jsonl(folder/'results.jsonl')[0];docs=read_jsonl(ROOT/'data/processed/corpus.jsonl')
        changed=copy.deepcopy(row);changed['answer']['claims'][0]['text']='错误改动'
        self.assertFalse(matching_cache(changed,meta,row['query'],config,docs))
        changed=copy.deepcopy(row);changed['messages'][0]['content']='篡改提示'
        self.assertFalse(matching_cache(changed,meta,row['query'],config,docs))

    def test_preparation_never_initializes_client_or_network(self):
        with patch('tools.manual_rag_review.ChatClient',side_effect=AssertionError('禁止真实客户端')), \
             patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('禁止网络')), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(manual_main(['--prepare-only']),0)

    def test_offline_mode_refuses_manual_generation_without_client(self):
        # EV003 has not been collected; even an explicit command remains offline.
        answers=iter(['EV003','生成','exit'])
        with patch('tools.manual_rag_review.ChatClient',side_effect=AssertionError('禁止真实客户端')), \
             patch('builtins.input',side_effect=lambda *args:next(answers)),redirect_stdout(io.StringIO()):
            self.assertEqual(manual_main(['--offline']),0)

if __name__=='__main__':unittest.main()
