"""Mechanical audit and factual usage ledger. No LLM calls or human/semantic scores."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icbc_rag.common import read_jsonl, file_hash
from icbc_rag.generation import validate_answer
from icbc_rag.prompts import system_for_style
from icbc_baseline.bm25 import canonical_hash


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    def percentile(p):
        index = (len(ordered)-1)*p
        lower = int(index)
        upper = min(lower+1, len(ordered)-1)
        return ordered[lower]+(ordered[upper]-ordered[lower])*(index-lower)
    return {'n': len(values), 'mean_ms': statistics.mean(values),
            'p50_ms': percentile(0.5), 'p95_ms': percentile(0.95),
            'quantile_method': 'linear interpolation at (n-1)*p'}


def main():
    snapshot = ROOT / 'data/evaluation/reviewed/20261021_settings_v2'
    dev = {r['query_id']: r for r in read_jsonl(snapshot / 'development.jsonl')}
    manifest = json.loads((snapshot / 'manifest.json').read_text(encoding='utf-8'))
    if any(file_hash(snapshot/name) != digest for name, digest in manifest['files'].items()):
        raise ValueError('核验快照变化')
    docs = {r['chunk_id']: r for r in read_jsonl(ROOT / 'data/processed/corpus.jsonl')}
    session = ROOT / 'outputs/manual_reviews/development_a7f94cb8e0ab'
    session_meta = json.loads((session/'manifest.json').read_text(encoding='utf-8'))
    records = []
    for path in sorted((session / 'answers').glob('EV*.json')):
        wrapper = json.loads(path.read_text(encoding='utf-8'))
        qid = wrapper['query_id']
        result = wrapper['result']
        if qid not in dev or wrapper['query'] != dev[qid]['query'] or result['query'] != dev[qid]['query']:
            raise ValueError('非固定开发题')
        valid = result.get('answer') is not None
        if valid:
            evidence = result['retrieval']['results']
            for hit in evidence:
                if hit['chunk_id'] not in docs or hit['text'] != docs[hit['chunk_id']]['text']:
                    raise ValueError('模型证据不属于固定原文')
            if validate_answer(result['provider_response']['content'], evidence) != result['answer']:
                raise ValueError('保存答案与原始响应不一致')
            messages = result['messages']
            prompt = json.loads(messages[1]['content'])
            if messages[0]['content'] != system_for_style() or prompt['question'] != dev[qid]['query'] or result['prompt_sha256'] != canonical_hash(messages):
                raise ValueError('提示或问题发生变化')
            if [r['chunk_id'] for r in prompt['evidence']] != [r['chunk_id'] for r in evidence]:
                raise ValueError('模型证据次序不一致')
            for item in prompt['evidence']:
                if any(value != docs[item['chunk_id']][key] for key, value in item.items()):
                    raise ValueError('模型证据字段与原文不一致')
        provider = result.get('provider_response') or {}
        records.append({'query_id': qid, 'origin': wrapper['origin'], 'source_sha256': file_hash(path),
                        'generation_status': result['generation_status'], 'mechanical_citation_check_passed': valid,
                        'expected_type': dev[qid]['answer_type'],
                        'model_type': (result.get('answer') or {}).get('status'),
                        'type_exact_match': (result.get('answer') or {}).get('status') == dev[qid]['answer_type'],
                        'claim_count': len((result.get('answer') or {}).get('claims', [])),
                        'usage': provider.get('usage'), 'provider_latency_ms': provider.get('latency_ms'),
                        'retrieval_ms': result.get('retrieval_ms'), 'requested_model': provider.get('requested_model'),
                        'returned_model': provider.get('returned_model'), 'semantic_support': 'not_graded_by_human',
                        'manual_notes': wrapper.get('manual_notes'), 'manual_reviewer': wrapper.get('manual_reviewer')})
    plans = [json.loads(p.read_text(encoding='utf-8')) for p in
             [ROOT/'outputs/agent_collection_plans/development_20261021.json',
              ROOT/'outputs/agent_collection_plans/development_20261021_remaining27.json']]
    ledger = {'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'scope': '24片段原提示top_k=3开发回答；不含29片段服务试用及最终测试',
              'development_query_count': len(dev), 'saved_answer_count': len(records),
              'pending_query_ids': session_meta['pending_attempts'],
              'missing_query_ids': sorted(set(dev)-{r['query_id'] for r in records}),
              'new_api_attempts_this_authorized_work': sum(p['external_model_calls'] for p in plans),
              'new_saved_responses_this_authorized_work': sum(len(p['completed_query_ids']) for p in plans),
              'mechanically_valid_answers': sum(r['mechanical_citation_check_passed'] for r in records),
              'processing_type_exact_matches': sum(r['type_exact_match'] for r in records),
              'processing_type_denominator': len(records), 'reference_type_counts': dict(Counter(r['expected_type'] for r in records)),
              'claim_count': sum(r['claim_count'] for r in records),
              'known_usage_answer_count': sum(bool(r['usage']) for r in records),
              'known_token_totals': {key: sum((r['usage'] or {}).get(key, 0) for r in records)
                                     for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']},
              'new_27_response_known_tokens': {key: sum((r['usage'] or {}).get(key, 0) for r in records
                                                       if r['origin'] == 'user_approved_agent_request')
                                               for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']},
              'type_mismatch_query_ids': [r['query_id'] for r in records if not r['type_exact_match']],
              'provider_latency': distribution([r['provider_latency_ms'] for r in records if r['provider_latency_ms'] is not None]),
              'retrieval_latency': distribution([r['retrieval_ms'] for r in records if r['retrieval_ms'] is not None]),
              'missing_response_usage': 'EV003请求后的响应未保存，是否计费及token未知；未重发，不能算0费用',
              'currency_cost': None, 'cost_reason': '单价未配置；实际费用需平台账单，已知token总数不含EV003',
              'human_semantic_scoring_performed': False, 'final_test_queries_run': 0,
              'test_file_unchanged': file_hash(snapshot/'test.jsonl') == manifest['files']['test.jsonl'],
              'answer_files': records}
    folder = ROOT / 'outputs/ai_reviews/development_20261021'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'mechanical_audit.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    text = ['# 开发回答机械核验与用量记录', '',
            f'已保存{len(records)}/30份真实回答；本次授权工作发生28次请求尝试，其中27份回答成功保存。此前2份回答复用。',
            'EV003仍未完成，未重发。它的响应未保存，不能编造回答、成功状态、token或费用。', '',
            f'保存的{ledger["mechanically_valid_answers"]}份回答通过原文来源、提示、引文与结构一致性复核。',
            f'处理类型与参考值精确一致：{ledger["processing_type_exact_matches"]}/{len(records)}；本批参考类型均为可回答，不能据此证明拒答或澄清能力。',
            '这些是自动核验，不是人工语义评分、要点覆盖率或回答准确率。', '',
            f'已保存回答的已知token：{ledger["known_token_totals"]["total_tokens"]}；包括2份历史回答，排除EV003未知用量。费用以账单为准。', '',
            '| 题号 | 引用机械复核 | 类型精确一致 | total tokens |', '| --- | --- | --- | ---: |']
    for r in records:
        text.append(f'| {r["query_id"]} | {r["mechanical_citation_check_passed"]} | {r["type_exact_match"]} | {(r["usage"] or {}).get("total_tokens", "未知")} |')
    text += ['', '原始用量、模型实际返回名称及分开计时的均值/P50/P95见同目录JSON。没有端到端精确计时，未将两种耗时之和冒充实测端到端值。',
             '16道最终测试未运行。EV036措辞待复核，人工评分、抽检、参数选择和冻结仍未完成。', '']
    (folder/'机械核验与用量.md').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps({'output': str(folder), **{k: ledger[k] for k in ['saved_answer_count', 'pending_query_ids',
                     'new_api_attempts_this_authorized_work', 'mechanically_valid_answers', 'known_token_totals',
                     'processing_type_exact_matches', 'test_file_unchanged']}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
