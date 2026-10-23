"""Synthetic engineering fixtures, never actual human-review records."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.export_reviewed_eval import HEADERS, read_review_rows, validate_rows


class ReviewGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates = [json.loads(s) for s in (ROOT / 'data/evaluation/candidates_unreviewed_20261015.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        cls.corpus = [json.loads(s) for s in (ROOT / 'data/processed/corpus.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        cls.book = ROOT / 'data/evaluation/评估题目待核验_20261015.xlsx'

    def fixture(self, index=0, split='开发'):
        c = self.candidates[index]
        r = dict.fromkeys(HEADERS, '')
        r.update({'题目ID':c['question_id'],'题组':c['group_id'],'问题草稿':c['question_draft'],
                  '核验状态':'已确认','确认问题':'保留原题','确认答案':'合成测试夹具答案，不代表真实人工核验。',
                  '确认类型':'可回答','确认依据ID':'S001-A02-P01','用途':split,'已用于调试':'否',
                  '核验人':'合成测试夹具','核验日期':'2026-10-15'})
        return r

    def validate(self, rows, smoke=None):
        return validate_rows(rows, self.candidates, self.corpus, smoke or set())

    def test_initial_workbook_has_50_pending_rows_and_no_gold(self):
        rows = read_review_rows(self.book)
        self.assertEqual(len(rows), 50)
        self.assertTrue(all(r['核验状态'] == '待核验' for r in rows))
        self.assertTrue(all(not r['确认答案'] and not r['核验人'] for r in rows))
        self.assertEqual(self.validate(rows), [])

    def test_explicit_fixture_exports_only_confirmed_fields(self):
        row = self.fixture()
        out = self.validate([row])[0]
        self.assertEqual(out['reference_answer'], row['确认答案'])
        self.assertEqual(out['query'], self.candidates[0]['question_draft'])
        self.assertNotIn('draft_keypoints', out)

    def test_no_automatic_fallback_to_ai_answer_or_label(self):
        row = self.fixture()
        row['确认答案'] = ''
        with self.assertRaisesRegex(ValueError, '缺少独立确认'):
            self.validate([row])

    def test_used_for_tuning_cannot_be_test(self):
        row = self.fixture(split='测试')
        row['已用于调试'] = '是'
        with self.assertRaisesRegex(ValueError, '不进入测试集'):
            self.validate([row])

    def test_same_group_cannot_span_development_and_test(self):
        with self.assertRaisesRegex(ValueError, '同题组跨开发/测试'):
            self.validate([self.fixture(0), self.fixture(1, '测试')])

    def test_unknown_evidence_is_rejected(self):
        row = self.fixture()
        row['确认依据ID'] = 'S999-UNKNOWN'
        with self.assertRaisesRegex(ValueError, '不在固定语料'):
            self.validate([row])

    def test_existing_smoke_question_cannot_be_final_test(self):
        row = self.fixture(split='测试')
        row['确认问题'] = '已经用于功能调试的问题'
        with self.assertRaisesRegex(ValueError, '不进入测试集'):
            self.validate([row], {'已经用于功能调试的问题'})

    def test_reviewer_and_date_are_required(self):
        row = self.fixture()
        row['核验人'], row['核验日期'] = '', ''
        with self.assertRaisesRegex(ValueError, '缺少核验人'):
            self.validate([row])


if __name__ == '__main__':
    unittest.main()
