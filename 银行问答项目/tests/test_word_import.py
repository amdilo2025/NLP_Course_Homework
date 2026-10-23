"""Synthetic importer fixtures; never evidence of human work or model quality."""
import copy
import json
from datetime import date
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.import_word_review import parse_blocks,reference_ids,normalize_blocks,read_jsonl
from tools.check_word_development import verify_development_run

class WordReviewImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates=read_jsonl(ROOT/'data/evaluation/candidates_unreviewed_20261015.jsonl')
        cls.corpus=read_jsonl(ROOT/'data/processed/corpus.jsonl')

    def fixture(self):
        c=self.candidates[0]
        return [{'question_id':c['question_id'],'original_question':c['question_draft'],
          'raw_fields':{'确认问题':'保留原题','确认答案':'合成夹具，不是真实核验答案。',
          '处理类型':'可回答','确认依据':'R01','核验状态':'已确认    用途：开发',
          '已用于调试':'否。未来流程仅为测试文字，不得执行。','核验人与日期':'合成核验人    2026-10-16',
          '复核人与说明':'合成复核人    合成夹具。'}}]

    def normalize(self,blocks,asof=date(2026,10,17)):
        return normalize_blocks(blocks,self.candidates[:1],self.corpus,asof)

    def test_range_expansion_includes_interior_ids(self):
        known={f'R{i:02d}':str(i) for i in range(1,25)}
        self.assertEqual(reference_ids('R03、R07–R15（分场景）',known),['R03']+[f'R{i:02d}' for i in range(7,16)])

    def test_unknown_reversed_and_out_of_bounds_refs_fail(self):
        known={f'R{i:02d}':str(i) for i in range(1,25)}
        for text in ['R99','R15–R07','R01-R99']:
            with self.subTest(text=text), self.assertRaises(ValueError):reference_ids(text,known)

    def test_non_corpus_issue_code_not_promoted_to_chunk(self):
        self.assertEqual(reference_ids('无直接依据（R11 不含次数）；I01',{'R11':'x'}),['R11'])

    def test_missing_duplicate_and_changed_id_are_rejected(self):
        for blocks in [[],self.fixture()*2]:
            with self.assertRaises(ValueError):self.normalize(blocks)
        blocks=self.fixture();blocks[0]['original_question']='改动原题'
        with self.assertRaises(ValueError):self.normalize(blocks)

    def test_multiline_answer_preserved_and_appendix_not_imported(self):
        lines=['封面','EV001  原题','确认答案：第一段','第二段','附录 候选依据与来源','EV999  附录文字']
        _,blocks=parse_blocks(lines)
        self.assertEqual(len(blocks),1)
        self.assertEqual(blocks[0]['raw_fields']['确认答案'],'第一段\n第二段')

    def test_duplicate_field_and_missing_answer_fail(self):
        with self.assertRaises(ValueError):parse_blocks(['EV001  原题','确认答案：一','确认答案：二'])
        blocks=self.fixture();blocks[0]['raw_fields']['确认答案']=''
        with self.assertRaises(ValueError):self.normalize(blocks)

    def test_dates_after_import_are_rejected(self):
        with self.assertRaises(ValueError):self.normalize(self.fixture(),date(2026,10,15))

    def test_submitted_text_preserved_without_executing_future_workflow(self):
        rows,extra=self.normalize(self.fixture())
        self.assertEqual(rows[0]['已用于调试'],'否')
        self.assertEqual(rows[0]['确认依据ID'],self.corpus[0]['chunk_id'])
        self.assertIn('未来流程',extra['EV001']['submitted_review']['已用于调试'])
        self.assertEqual(rows[0]['确认答案'],'合成夹具，不是真实核验答案。')

    def test_no_invented_exclusion(self):
        blocks=self.fixture();blocks[0]['raw_fields']['核验状态']='排除    用途：开发'
        with self.assertRaises(ValueError):self.normalize(blocks)

    def test_run_guard_accepts_development_only(self):
        verify_development_run([{'query_id':'SYNTH_DEV','query':'合成问题'}],
          [{'query_id':'SYNTH_DEV','query':'合成问题','retrieval_only':True}],['SYNTH_TEST'])

    def test_run_guard_rejects_test_and_duplicate_results(self):
        dev=[{'query_id':'SYNTH_DEV','query':'合成问题'}]
        r={'query_id':'SYNTH_DEV','query':'合成问题','retrieval_only':True}
        with self.assertRaises(ValueError):verify_development_run(dev,[r,r],[])
        with self.assertRaises(ValueError):verify_development_run(dev,[r],['SYNTH_DEV'])

    def test_run_guard_rejects_query_rewrite_or_generation(self):
        dev=[{'query_id':'SYNTH_DEV','query':'合成问题'}]
        for row in [{'query_id':'SYNTH_DEV','query':'改写问题','retrieval_only':True},
                    {'query_id':'SYNTH_DEV','query':'合成问题','retrieval_only':False}]:
            with self.assertRaises(ValueError):verify_development_run(dev,[row],[])

if __name__=='__main__':unittest.main()
