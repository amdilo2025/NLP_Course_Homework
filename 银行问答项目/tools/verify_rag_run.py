"""Check actual development preview evidence; never compute answer scores."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.rag_packages').is_dir():sys.path.insert(0,str(ROOT/'.rag_packages'))
sys.path.insert(0,str(ROOT/'src'))
from icbc_rag.common import file_hash,read_jsonl
from icbc_baseline.bm25 import canonical_hash
from icbc_rag.retrieval import DenseIndex

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();run=args.run.resolve()
    if not run.is_relative_to(ROOT/'outputs/rag_runs'):raise ValueError('运行目录必须位于outputs/rag_runs')
    meta=json.loads((run/'run_metadata.json').read_text(encoding='utf-8'))
    records=read_jsonl(run/'results.jsonl')
    snap=ROOT/'data/evaluation/reviewed/20261017_word_v1'
    manifest=json.loads((snap/'manifest.json').read_text(encoding='utf-8'))
    queries=read_jsonl(snap/'development_queries.jsonl');expected={r['query_id']:r['query'] for r in queries}
    if meta['mode']!='preview' or meta['query_count']!=30 or len(records)!=30:raise ValueError('本次应为30题开发预览')
    if len({r['query_id'] for r in records})!=30 or {r['query_id'] for r in records}!=set(expected):raise ValueError('运行题号与开发快照不一致')
    for record in records:
        if record['query']!=expected[record['query_id']]:raise ValueError('问题文本变化')
        if record['answer'] is not None or record['external_model_calls']!=0 or record['generation_status']!='preview_only_no_model_called':
            raise ValueError('预览不得含生成答案或模型调用')
        prompt=json.loads(record['messages'][1]['content'])
        if prompt['question']!=record['query']:raise ValueError('提示问题变化')
        hits=record['retrieval']['results']
        if [x['chunk_id'] for x in prompt['evidence']]!=[x['chunk_id'] for x in hits]:raise ValueError('提示证据与检索不一致')
        for evidence in prompt['evidence']:
            if set(evidence)!={'chunk_id','text','source_title','source_version','scope','citation_locator'}:raise ValueError('提示含额外字段')
    if meta['external_model_calls'] or meta['accepted_generated_answers'] or meta['formal_scoring_performed']:
        raise ValueError('本轮不得伪记模型调用、回答或效果评分')
    if set(expected)&set(manifest['question_ids']['test']):raise ValueError('开发集混入最终测试')
    for name,digest in manifest['files'].items():
        if file_hash(snap/name)!=digest:raise ValueError('返回Word或核验数据快照发生变化')
    if file_hash(snap/'test.jsonl')!=meta['test_sha256_before'] or not meta['test_file_unchanged']:
        raise ValueError('最终测试快照发生变化')
    corpus=read_jsonl(ROOT/meta['config']['corpus_file'])
    index=DenseIndex.load(ROOT/meta['config']['index_file'],corpus,meta['encoder_fingerprint'])
    model=ROOT/meta['config']['model_dir']
    weights=json.loads((model/'manifest.json').read_text(encoding='utf-8'))
    for name,info in weights['files'].items():
        if file_hash(model/name)!=info['sha256']:raise ValueError('模型文件哈希变化')
    if len(corpus)!=24 or canonical_hash(corpus)!=manifest['corpus_sha256']:raise ValueError('原文语料不符')
    if any(set(x)&{'reference_answer','reviewer','query_id'} for x in corpus):raise ValueError('索引不得混入评估答案或核验人')
    for name,digest in meta['code_sha256'].items():
        if file_hash(ROOT/'src/icbc_rag'/name)!=digest:raise ValueError('运行后源码变化，请核对版本：'+name)
    report={'run':run.relative_to(ROOT).as_posix(),'source_chunks':24,'embedding_dimension':512,
      'max_source_tokens':max(index.counts),'truncated_source_chunks':0,'development_queries':30,
      'final_test_queries_run':0,'generated_answers':0,'external_model_calls':0,
      'test_snapshot_unchanged':True,'human_answers_unchanged':True,
      'prompt_evidence_matches_retrieval':True,'formal_scoring_performed':False,
      'model':meta['encoder'],'run_metadata_sha256':file_hash(run/'run_metadata.json'),
      'results_sha256':file_hash(run/'results.jsonl'),
      'notice':'仅证实本地检索、预览和工程流程。尚无真实大模型回答、答案准确率、模型效果提升或语义引用正确性结论。'}
    (ROOT/'outputs/checks/rag_readiness_20261017.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
