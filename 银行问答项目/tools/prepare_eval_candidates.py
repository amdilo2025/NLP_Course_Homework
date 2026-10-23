"""Prepare AI-drafted candidates; never mark them reviewed or split them."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icbc_baseline.bm25 import canonical_hash


def normalized_question(text):
    return re.sub(r'[\W_]+', '', text).lower()


def main():
    output = ROOT / 'data/evaluation'
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'candidates_unreviewed_20261015.jsonl'
    if target.exists():
        raise ValueError('Candidate snapshot already exists; preserve it and use a new version name.')
    corpus = [json.loads(t) for t in (ROOT / 'data/processed/corpus.jsonl').read_text(encoding='utf-8').splitlines() if t.strip()]
    chunks = {d['chunk_id']: d for d in corpus}
    families = json.loads((ROOT / 'config/eval_candidate_topics.json').read_text(encoding='utf-8'))
    smoke = [json.loads(t) for t in (ROOT / 'examples/smoke_queries.jsonl').read_text(encoding='utf-8').splitlines() if t.strip()]
    smoke_questions = {normalized_question(t['query']) for t in smoke}
    prior_results = {}
    for run in sorted((ROOT / 'outputs/runs').glob('*/retrieval.jsonl')):
        for line in run.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('query_id', '').startswith('SMOKE') and row.get('results'):
                prior_results[row['query_id']] = row['results'][0]['chunk_id']
    candidates, seen = [], set()
    for family in families:
        refs = family['candidate_chunk_ids']
        if any(r not in chunks for r in refs):
            raise ValueError(f'Unknown evidence in {family["group_id"]}')
        for question in family['questions']:
            normalized = normalized_question(question)
            if normalized in seen or normalized in smoke_questions:
                raise ValueError('Duplicate candidate or an existing smoke question')
            seen.add(normalized)
            candidates.append({
                'question_id': f'EV{len(candidates) + 1:03d}',
                'group_id': family['group_id'], 'category': family['category'],
                'question_draft': question, 'draft_type': family['draft_type'],
                'draft_keypoints': family['draft_keypoints'],
                'candidate_chunk_ids': refs, 'review_note': family['review_note'],
                'evidence_locators': [chunks[r]['citation_locator'] for r in refs],
                'related_smoke_ids': sorted(k for k,v in prior_results.items() if v in refs),
                'exposure_note': '共享候选依据不等于本题已运行；人工确认是否用于调试',
                'author': 'AI起草，非人工金标准', 'review_status': '待核验',
                'confirmed_question': '', 'confirmed_answer': '', 'confirmed_type': '',
                'confirmed_chunk_ids': [], 'split': '待分配', 'used_for_tuning': '待核查',
                'reviewer': '', 'review_date': '', 'edit_note': ''
            })
    target.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in candidates), encoding='utf-8')
    manifest = {
        'created_on': '2026-10-15', 'status': 'AI候选草稿，待人工核验',
        'candidate_count': len(candidates), 'group_count': len(families),
        'reviewed_count': 0, 'assigned_development_count': 0, 'assigned_test_count': 0,
        'corpus_sha256': canonical_hash(corpus), 'candidate_sha256': canonical_hash(candidates),
        'candidate_file': target.name, 'gold_created': False,
        'note': '不预设调试/测试比例；未运行候选题检索或评分。'
    }
    (output / 'preparation_manifest_20261015.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
