"""Calculate declared development retrieval metrics, without LLM calls or semantic grading."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icbc_rag.common import read_jsonl, file_hash


def score_ranking(ranked_ids, groups, full_diagnostic=()):
    if not groups:
        return {'eligible': False, 'required_group_count': 0, 'hit_group_count': 0,
                'recall': None, 'reciprocal_rank': None, 'complete_coverage': None,
                'alternative_full_coverage': None}
    if len(ranked_ids) != len(set(ranked_ids)) or any(not g for g in groups):
        raise ValueError('排名不能重复，必需组不能为空')
    hits = [bool(set(g) & set(ranked_ids)) for g in groups]
    needed = set().union(*(set(g) for g in groups))
    first = next((i for i, cid in enumerate(ranked_ids, 1) if cid in needed), None)
    return {'eligible': True, 'required_group_count': len(groups), 'hit_group_count': sum(hits),
            'recall': sum(hits) / len(groups), 'reciprocal_rank': 1 / first if first else 0,
            'complete_coverage': int(all(hits)),
            'alternative_full_coverage': int(set(full_diagnostic) <= set(ranked_ids)) if full_diagnostic else None}


def summarize(records):
    eligible = [r for r in records if r['eligible']]
    result = {'query_count': len(records), 'retrieval_eligible_queries': len(eligible),
              'excluded_from_retrieval_denominator': len(records) - len(eligible)}
    for key in ['recall', 'reciprocal_rank', 'complete_coverage', 'alternative_full_coverage']:
        values = [r[key] for r in eligible if r[key] is not None]
        result[key] = sum(values) / len(values) if values else None
        result[key + '_denominator'] = len(values)
    return result


def collect_scores(snapshot, comparisons):
    snapshot = Path(snapshot)
    if not snapshot.is_absolute():
        snapshot = ROOT / snapshot
    manifest = json.loads((snapshot / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        if file_hash(snapshot / name) != digest:
            raise ValueError('快照哈希不一致：' + name)
    v1 = ROOT / 'data/evaluation/reviewed/20261017_word_v1'
    dev = read_jsonl(snapshot / 'development.jsonl')
    queries = read_jsonl(snapshot / 'development_queries.jsonl')
    if queries != read_jsonl(v1 / 'development_queries.jsonl') or file_hash(snapshot / 'test.jsonl') != file_hash(v1 / 'test.jsonl'):
        raise ValueError('开发问题或最终测试文件发生变化')
    byid = {r['query_id']: r for r in dev}
    expected = {r['query_id']: r['query'] for r in queries}
    corpus = read_jsonl(ROOT / 'data/processed/corpus.jsonl')
    rids = {f'R{i:02d}': c['chunk_id'] for i, c in enumerate(corpus, 1)}
    policy = json.loads((snapshot / 'evaluation_policy.json').read_text(encoding='utf-8'))
    grouped = {}
    origins = []
    encoder_fingerprint = None
    for folder in comparisons:
        folder = Path(folder)
        if not folder.is_absolute():
            folder = ROOT / folder
        meta = json.loads((folder / 'run_metadata.json').read_text(encoding='utf-8'))
        if meta['dataset_sha256'] != file_hash(snapshot / 'development.jsonl') or meta['corpus_sha256'] != manifest['corpus_sha256']:
            raise ValueError('检索运行的数据版本不一致')
        if meta.get('snapshot_manifest_sha256') != file_hash(snapshot / 'manifest.json'):
            raise ValueError('检索运行未绑定同一核验快照')
        if encoder_fingerprint is not None and meta['encoder_fingerprint'] != encoder_fingerprint:
            raise ValueError('检索运行的编码器版本不一致')
        encoder_fingerprint = meta['encoder_fingerprint']
        if meta['external_model_calls'] != 0 or meta['final_test_queries_run'] != 0 or not meta['test_file_unchanged']:
            raise ValueError('只接收离线开发检索记录')
        if meta['results_sha256'] != file_hash(folder / 'comparison.jsonl'):
            raise ValueError('检索记录哈希不一致')
        if meta.get('bm25_k1') != 1.2 or meta.get('bm25_b') != 0.75:
            raise ValueError('BM25参数与小组固定值不一致')
        if sorted(meta['top_k_values']) != policy['retrieval_parameters']['top_k']:
            raise ValueError('k取值与小组方案不一致')
        if meta['tokenizer'] not in policy['retrieval_parameters']['bm25_tokenizers']:
            raise ValueError('分词方式与小组方案不一致')
        rows = read_jsonl(folder / 'comparison.jsonl')
        if len(rows) != len(expected) * len(meta['top_k_values']) or len({(r['query_id'], r['top_k']) for r in rows}) != len(rows):
            raise ValueError('开发排名缺失或重复')
        origins.append({'directory': folder.relative_to(ROOT).as_posix(), 'metadata_sha256': file_hash(folder / 'run_metadata.json')})
        for row in rows:
            if row['query_id'] not in expected or row['query'] != expected[row['query_id']] or row['top_k'] not in meta['top_k_values']:
                raise ValueError('运行含非开发题或未知参数')
            roles = byid[row['query_id']]['reference_roles']
            groups = [[rids[r] for r in g] for g in roles['required_groups']]
            full = [rids[r] for r in roles['full_coverage_diagnostic']]
            for method in ['bm25', 'dense']:
                label = f'bm25_{meta["tokenizer"]}' if method == 'bm25' else 'dense'
                ranked = [x['chunk_id'] for x in row[method]]
                if len(ranked) > row['top_k'] or any(cid not in rids.values() for cid in ranked):
                    raise ValueError('排名超长或含未知片段')
                result = {'method': label, 'top_k': row['top_k'], 'query_id': row['query_id'],
                          'ranked_chunk_ids': ranked, 'required_groups': roles['required_groups'],
                          **score_ranking(ranked, groups, full)}
                key = (label, row['top_k'], row['query_id'])
                if key in grouped and grouped[key] != result:
                    raise ValueError('重复配置的排名或分数不一致，不能静默合并')
                grouped[key] = result  # Dense reused across tokenizer runs, never counted twice.
    records = sorted(grouped.values(), key=lambda r: (r['method'], r['top_k'], r['query_id']))
    summaries = []
    for method, k in sorted({(r['method'], r['top_k']) for r in records}):
        batch = [r for r in records if r['method'] == method and r['top_k'] == k]
        if {r['query_id'] for r in batch} != set(expected):
            raise ValueError('配置缺少开发题')
        summaries.append({'method': method, 'top_k': k, **summarize(batch)})
    if len(summaries) != 9:
        raise ValueError('须覆盖两种BM25分词及向量检索的1/3/5，共9组')
    return records, summaries, origins, policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=ROOT / 'data/evaluation/reviewed/20261021_settings_v2')
    parser.add_argument('--comparisons', type=Path, nargs='+', required=True)
    args = parser.parse_args(argv)
    if not args.snapshot.is_absolute():
        args.snapshot = ROOT / args.snapshot
    records, summaries, origins, policy = collect_scores(args.snapshot, args.comparisons)
    now = datetime.now(timezone(timedelta(hours=8)))
    target = ROOT / 'outputs/retrieval_metrics' / now.strftime('%Y%m%dT%H%M%S_%f')
    target.mkdir(parents=True)
    (target / 'per_query.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records), encoding='utf-8')
    report = {'created_at': now.isoformat(), 'kind': 'development_retrieval_metrics_only',
              'snapshot_manifest_sha256': file_hash(args.snapshot / 'manifest.json'), 'source_runs': origins,
              'metrics': summaries, 'definitions': policy['metric_implementation'],
              'external_model_calls': 0, 'final_test_queries_run': 0, 'automatic_parameter_selection': False,
              'human_semantic_scoring_performed': False, 'results_sha256': file_hash(target / 'per_query.jsonl')}
    (target / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 开发集检索参数对照：数值记录', '',
             '按小组返回设置计算，仅使用原24片段与30道开发题；没有调用大模型或运行最终测试。',
             '这是检索数值记录，不包含回答的人工语义评分、最优参数选择或报告结果分析。', '',
             '| 方法 | k | 计分题数 | Recall@k | MRR@k | 必需组完整覆盖率 | G19两段齐全率 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summaries:
        lines.append('| ' + ' | '.join([row['method'], str(row['top_k']), str(row['retrieval_eligible_queries'])] +
                     [f'{row[k]:.4f}' if row[k] is not None else '不适用' for k in
                      ['recall', 'reciprocal_rank', 'complete_coverage', 'alternative_full_coverage']]) + ' |')
    lines += ['', '## 数值定义与限制', '',
              '- k=3为小组指定的主报告口径；k=1、5按各自实际截断长度写为Recall@1、Recall@5，不冒称Recall@3。',
              '- Recall按题平均：命中的必需组数除以该题必需组总数。共同必需的两段各算一组；可替代的两段合为一组，任一命中即满足。',
              '- MRR采用首个必需或可替代片段的倒数排名；命中第一段不等于同时覆盖全部必需段。完整覆盖率另列。',
              '- G19两段齐全率只对EV037、EV038计分；任一命中的主口径仍单独保留。补充依据不计入分母。',
              '- 同一向量配置在两个分词运行中重复出现，经一致性核对后仅统计一次。共9组配置、每组30题，不是90道不同题。',
              '- 各k共用同题最大k的检索计时，本记录不提供各k独立速度比较；没有准确的生成上下文token或生成费用实验。',
              '- 参数选择仍须结合人工引用支持和小组选择理由。未自动修改默认配置，也未触发最终测试。', '']
    (target / '开发检索参数对照.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(target), 'configurations': len(summaries), 'rows': len(records), 'external_model_calls': 0}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
