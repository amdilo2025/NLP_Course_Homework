"""Audit saved final outputs and calculate frozen mechanical metrics; no inference."""
from collections import Counter
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from freeze_final_configuration import verify_freeze,apply_reference_errata
from icbc_rag.common import file_hash,read_jsonl
from icbc_rag.prompts import build_messages
from icbc_rag.generation import validate_answer
from score_retrieval_comparison import score_ranking,summarize

RUN=ROOT/'outputs/final_test_runs/20261021_k3_once'
OUT=ROOT/'outputs/final_test_audit/20261022'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit():
    plan=load(ROOT/'outputs/final_test_plans/20261021_k3_once.json')
    freeze=ROOT/plan['freeze_directory']
    verify_freeze(freeze)
    if file_hash(freeze/'freeze_manifest.json')!=plan['freeze_manifest_sha256']:
        raise ValueError('冻结清单变化')
    meta=load(RUN/'run_metadata.json')
    ids=plan['query_ids']
    if (meta['state']!='completed_semantic_scoring_pending' or meta['reserved_ids']!=ids
        or meta['completed_ids']!=ids or meta['pending_query_id'] is not None
        or meta['external_model_calls']!=16 or not plan['approval']['approved']):
        raise ValueError('批次未完整完成或调用次数不符')
    rows=read_jsonl(ROOT/plan['base_test_file'])
    if file_hash(ROOT/plan['base_test_file'])!=plan['base_test_sha256']:
        raise ValueError('原测试文件变化')
    if [r['query_id'] for r in rows]!=ids:
        raise ValueError('测试题号不一致')
    corpus=read_jsonl(ROOT/'data/processed/corpus.jsonl')
    docs={d['chunk_id']:d for d in corpus}
    rid_to_chunk={f'R{i:02d}':d['chunk_id'] for i,d in enumerate(corpus,1)}
    policy=load(freeze/'scoring_protocol.json')
    errata=load(freeze/'reference_errata.json')
    effective=[apply_reference_errata(r,errata) for r in rows]
    if load(RUN/'scoring_references_only.json')!=effective:
        raise ValueError('评分副本未正确应用冻结更正')
    per_query=[];retrieval=[];usage=Counter();models=Counter();latencies=[];claim_count=0
    for row in rows:
        qid=row['query_id']
        answer=load(RUN/(qid+'_answer.json'))
        provider=load(RUN/(qid+'_provider.json'))
        request=load(RUN/(qid+'_request.json'))
        bm25=load(RUN/(qid+'_bm25.json'))
        if answer['query_id']!=qid or answer['query']!=row['query'] or bm25['query']!=row['query']:
            raise ValueError('保存结果的题目不符')
        hits=answer['retrieval']['results']
        originals=[docs[h['chunk_id']] for h in hits]
        for hit,original in zip(hits,originals):
            if any(hit[k]!=v for k,v in original.items()):
                raise ValueError('原文候选内容变化')
        messages=build_messages(row['query'],originals)
        if request!={'query_id':qid,'messages':messages} or answer['messages']!=messages:
            raise ValueError('模型输入不等于问题加固定原文')
        if provider!=answer['provider_response'] or validate_answer(provider['content'],hits)!=answer['answer']:
            raise ValueError('客户端原响应或引用机械核验不一致')
        if answer['generation_status']!='generated_citations_checked_semantics_unverified' or answer['external_model_calls']!=1:
            raise ValueError('本题状态或调用次数异常')
        claim_count+=len(answer['answer']['claims'])
        for key in ['prompt_tokens','completion_tokens','total_tokens']:
            value=provider['usage'][key]
            if not isinstance(value,int) or value<0:
                raise ValueError('用量字段异常')
            usage[key]+=value
        models[provider['returned_model']]+=1
        latencies.append(provider['latency_ms'])
        roles=policy['reference_roles_by_query'][qid]
        groups=[[rid_to_chunk[rid] for rid in group] for group in roles['required_groups']]
        full=[rid_to_chunk[rid] for rid in roles['full_coverage_diagnostic']]
        for method,h in [('bm25',bm25['results']),('dense',hits)]:
            ranked=[v['chunk_id'] for v in h]
            if len(ranked)>3 or not set(ranked)<=set(docs):
                raise ValueError('排名超长或来源不符')
            retrieval.append({'query_id':qid,'method':method,'top_k':3,'ranked_chunk_ids':ranked,
                              **score_ranking(ranked,groups,full)})
        per_query.append({'query_id':qid,'expected_type':row['answer_type'],
                          'generated_type':answer['answer']['status'],
                          'type_exact_match':int(row['answer_type']==answer['answer']['status']),
                          'claim_count':len(answer['answer']['claims']),
                          'provider_latency_ms':provider['latency_ms'],
                          'usage':provider['usage'],'mechanical_validation_passed':True,
                          'human_semantic_score':None})
    summary={'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
             'run_directory':RUN.relative_to(ROOT).as_posix(),'final_test_queries':16,
             'external_model_calls_in_final_run':16,'retries':0,'EV003_resent':False,
             'mechanically_accepted_responses':16,'claim_count':claim_count,
             'type_exact_matches':sum(r['type_exact_match'] for r in per_query),
             'type_denominator':16,'type_mismatch_query_ids':[r['query_id'] for r in per_query if not r['type_exact_match']],
             'expected_types':dict(Counter(r['answer_type'] for r in rows)),
             'generated_types':dict(Counter(r['generated_type'] for r in per_query)),
             'usage':dict(usage),'returned_models':dict(models),
             'provider_latency_ms':{'n':16,'mean':statistics.mean(latencies),'median':statistics.median(latencies)},
             'estimated_currency_cost':None,'cost_note':'未提供实际单价；不把未知费用记为0。',
             'freeze_unchanged':True,'test_file_unchanged':True,
             'gold_references_absent_from_model_inputs':True,
             'reference_errata_applied_only_for_scoring':['EV036'],
             'human_semantic_scoring_performed':False,
             'known_topic_overlap':policy['known_topic_overlap'],
             'retrieval_metrics':[{'method':m,'top_k':3,**summarize([r for r in retrieval if r['method']==m])} for m in ['bm25','dense']],
             'scoring_protocol_sha256':file_hash(freeze/'scoring_protocol.json'),
             'run_metadata_sha256':file_hash(RUN/'run_metadata.json'),
             'source_files_sha256':{p.name:file_hash(p) for p in sorted(RUN.glob('*.json'))},
             'metric_note':'类型精确匹配、检索指标与逐字引用检查各自统计，均不替代人工回答语义评分。'}
    OUT.mkdir(parents=True,exist_ok=False)
    for name,data in [('summary.json',summary),('per_query.json',per_query),('retrieval_per_query.json',retrieval)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_files_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    audit()
