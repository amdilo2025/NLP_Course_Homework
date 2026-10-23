"""Offline, same-corpus development ranking comparison; no formal scoring."""
import argparse
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.rag_packages').is_dir():sys.path.insert(0,str(ROOT/'.rag_packages'))
sys.path.insert(0,str(ROOT/'src'))
from icbc_baseline.bm25 import BM25Index,canonical_hash
from icbc_rag.common import read_jsonl,load_config,project_path,file_hash
from icbc_rag.embeddings import LocalBGE
from icbc_rag.retrieval import prepare_index
from icbc_rag.cli import development_inputs

def validate_ks(values,count):
    if not values or len(values)!=len(set(values)) or any(type(k) is not int or not 1<=k<=count for k in values):
        raise ValueError('top-k须为不重复且不超过语料数的正整数')
    return values

def main(argv=None):
    parser=argparse.ArgumentParser(description='只用固定开发题，离线比较BM25与向量排名，不调用大模型')
    parser.add_argument('--top-k',type=int,nargs='+',help='由小组指定参数；未填时只核对既有默认值，不自动选择最优参数')
    parser.add_argument('--tokenizer',choices=['char','char_bigram'],help='可选BM25分词方式，由小组指定')
    parser.add_argument('--reviewed-dir',type=Path,help='已核验快照；只能使用与v1相同的开发问题和最终测试原文件')
    args=parser.parse_args(argv)
    config=load_config(ROOT/'config/rag.json')
    bm=json.loads((ROOT/'config/baseline.json').read_text(encoding='utf-8'))
    queries,partition=development_inputs()
    snapshot=args.reviewed_dir or ROOT/'data/evaluation/reviewed/20261017_word_v1'
    if not snapshot.is_absolute():snapshot=ROOT/snapshot
    selected_manifest=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in selected_manifest['files'].items():
        if file_hash(snapshot/name)!=digest:raise ValueError('评估快照发生变化：'+name)
    if read_jsonl(snapshot/'development_queries.jsonl')!=queries or file_hash(snapshot/'test.jsonl')!=partition['test_sha256_before']:
        raise ValueError('新版本不能改变开发问题或最终测试文件')
    reviewed=read_jsonl(snapshot/'development.jsonl');byid={x['query_id']:x for x in reviewed}
    corpus=read_jsonl(project_path(config['corpus_file']))
    dataset=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    if canonical_hash(corpus)!=dataset['corpus_sha256'] or canonical_hash(corpus)!=config['corpus_sha256']:
        raise ValueError('本轮必须使用固定24片段评估语料，不接受交互扩展版')
    ks=validate_ks(args.top_k or [config['top_k']],len(corpus))
    mode=args.tokenizer or bm['tokenizer']
    bm_index=BM25Index(corpus,bm['k1'],bm['b'],mode)
    encoder=LocalBGE(config);dense,built=prepare_index(config,encoder)
    now=datetime.now(timezone(timedelta(hours=8)))
    target=ROOT/'outputs/retrieval_comparisons'/now.strftime('%Y%m%dT%H%M%S_%f')
    target.mkdir(parents=True,exist_ok=False)
    records=[];maximum=max(ks)
    for row in queries:
        start=time.perf_counter();b=bm_index.search(row['query'],maximum)
        bm_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter();vector,tokens=encoder.encode([row['query']],query=True)
        d=dense.search_vector(row['query'],vector[0],maximum)
        dense_ms=(time.perf_counter()-start)*1000
        refs=byid[row['query_id']]['reference_chunk_ids']
        for k in ks:
            def take(result):return [{'rank':h['rank'],'chunk_id':h['chunk_id'],'score':h['score']} for h in result['results'][:k]]
            records.append({'query_id':row['query_id'],'query':row['query'],'top_k':k,
              'bm25':take(b),'dense':take(d),'bm25_ms_for_max_k':bm_ms,'dense_ms_for_max_k':dense_ms,
              'timing_note':'同题多个k共用最大k的一次检索计时，不当成各k独立延迟实验',
              'submitted_reference_chunk_ids':refs,
              'reference_note':'依据关系见指定快照；本工具仅记录排名，统计由独立工具完成',
              'query_tokens':tokens[0],'is_formal_score':False})
    if file_hash(partition['test_file'])!=partition['test_sha256_before']:raise ValueError('最终测试文件变化')
    (target/'comparison.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records),encoding='utf-8')
    info={'created_at':now.isoformat(),'kind':'development_ranking_diagnostic','query_count':len(queries),
      'top_k_values':ks,'parameter_origin':'caller_supplied' if args.top_k else 'existing_default_only',
      'tokenizer':mode,'bm25_k1':bm['k1'],'bm25_b':bm['b'],'corpus_sha256':dense.corpus_sha256,
      'dataset_sha256':file_hash(snapshot/'development.jsonl'),'reviewed_snapshot':snapshot.relative_to(ROOT).as_posix(),
      'snapshot_manifest_sha256':file_hash(snapshot/'manifest.json'),'encoder':encoder.info,
      'encoder_fingerprint':encoder.fingerprint,'index_built':built,'external_model_calls':0,
      'final_test_queries_run':0,'test_file_unchanged':True,'formal_scoring_performed':False,
      'automatic_parameter_selection':False,'python_version':platform.python_version(),
      'results_sha256':file_hash(target/'comparison.jsonl'),'code_sha256':file_hash(Path(__file__))}
    (target/'run_metadata.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    rids={c['chunk_id']:f'R{i:02d}' for i,c in enumerate(corpus,1)}
    lines=['# 开发题检索对照','',
      '这是同一固定语料、同一组30道开发题的逐题排名记录，不是正式准确率或结果分析。没有调用DeepSeek，也没有运行16道最终测试题。',
      '',f'参数：top-k={ks}，BM25分词={mode}。未自动选择最优参数。参考依据可能包含互补、补充或替代片段，未直接据此计算总体Recall。','',
      '| 题号 | k | 问题 | BM25召回R编号 | 向量召回R编号 | 填写的参考依据 |',
      '| --- | ---: | --- | --- | --- | --- |']
    for x in records:
        label=lambda values:'、'.join(rids[y['chunk_id']] for y in values)
        lines.append(f'| {x["query_id"]} | {x["top_k"]} | {x["query"]} | {label(x["bm25"])} | {label(x["dense"])} | '+
          '、'.join(rids[c] for c in x['submitted_reference_chunk_ids'])+' |')
    lines+=['','R编号与原核验Word附录一致。原始相似度/BM25分数、配置、版本和耗时见同目录JSONL与版本清单。',
      '不同检索方法的分数尺度不同，不能直接把数值大小作为效果比较。正式评分规则、比较参数及结论由小组确定。','']
    (target/'检索对照.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'output':str(target),'development_queries':len(queries),'top_k':ks,'external_model_calls':0,'formal_scoring':False},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
