"""Offline-only service corpus and presentation checks. Never instantiate an API client."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.rag_packages').is_dir():sys.path.insert(0,str(ROOT/'.rag_packages'))
sys.path.insert(0,str(ROOT/'src'))
from icbc_rag.cli import display,main
from icbc_rag.common import read_jsonl,file_hash,load_config
from icbc_rag.prompts import build_messages,SYSTEM,system_for_style
from icbc_rag.retrieval import article_context

class ServiceExperienceTests(unittest.TestCase):
    def test_expansion_supplies_governing_paragraph_and_keeps_sources_separate(self):
        docs=[{'chunk_id':'A8P1','source_id':'A','article_number':8,'paragraph_number':1},
          {'chunk_id':'A8P2','source_id':'A','article_number':8,'paragraph_number':2},
          {'chunk_id':'B8P1','source_id':'B','article_number':8,'paragraph_number':1}]
        output=article_context(docs,[docs[1],docs[0]])
        self.assertEqual([x['chunk_id'] for x in output],['A8P1','A8P2'])

    def test_service_corpus_preserves_frozen_records_and_conditions(self):
        original=read_jsonl(ROOT/'data/processed/corpus.jsonl')
        service=read_jsonl(ROOT/'data/processed/corpus_service_20261020.jsonl')
        self.assertEqual(service[:len(original)],original)
        self.assertEqual(len(service),29)
        added={x['chunk_id']:x for x in service[len(original):]}
        self.assertIn('经发卡银行审核通过后',added['S001-A08-P01']['text'])
        self.assertIn('指定的线上渠道',added['S001-A08-P03']['text'])
        self.assertIn('监管机构允许的特殊卡种除外',added['S001-A08-P03']['text'])
        self.assertTrue(all(x['review_status'].startswith('AI') for x in added.values()))

    def test_service_prompt_is_optional_and_does_not_change_original_template(self):
        self.assertEqual(system_for_style(),SYSTEM)
        self.assertIn('不因缺少App逐级菜单',system_for_style('service_steps_v2'))
        docs=read_jsonl(ROOT/'data/processed/corpus_service_20261020.jsonl')
        messages=build_messages('怎么办理借记卡',[docs[-4]],style='service_steps_v2')
        self.assertNotIn('reference_answer',json.dumps(messages))

    def fixture(self):
        hit={'chunk_id':'SYNTH','text':'合成原文句子，仅用于离线展示检查。','source_title':'合成来源',
          'source_url':'https://example.org/fixture','citation_locator':'合成第一条','rank':1,'score':.8}
        return {'query':'合成问题','mode':'api','retrieval':{'results':[hit]},
          'answer':{'status':'可回答','message':'','validation_note':'合成核验说明',
          'claims':[{'text':'合成办理指引。','citations':[{**hit,'quote':hit['text']}]}]}}

    def test_default_api_display_prioritizes_answer_and_hides_search_dump(self):
        buffer=io.StringIO()
        with redirect_stdout(buffer):display(self.fixture())
        text=buffer.getvalue()
        self.assertLess(text.index('合成办理指引'),text.index('依据：'))
        self.assertNotIn('相似度',text)
        self.assertNotIn('合成原文句子',text)
        self.assertIn('/详情',text)

    def test_details_are_pure_local_display_and_show_quotes(self):
        buffer=io.StringIO()
        with patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('不得访问网络')):
            with redirect_stdout(buffer):display(self.fixture(),details=True)
        self.assertIn('合成原文句子',buffer.getvalue())
        self.assertIn('合成核验说明',buffer.getvalue())

    def test_old_reviewed_files_are_unchanged(self):
        folder=ROOT/'data/evaluation/reviewed/20261017_word_v1'
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        for name,digest in manifest['files'].items():self.assertEqual(file_hash(folder/name),digest)

    def test_service_profile_cannot_run_old_evaluation_batch(self):
        with patch('icbc_rag.cli.ChatClient',side_effect=AssertionError('不得初始化API客户端')):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config',str(ROOT/'config/service_rag.json'),'--development','--mode','api']),1)

if __name__=='__main__':unittest.main()
