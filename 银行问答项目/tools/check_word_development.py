"""Verify development-only run provenance and write inspectable per-query checks.

No answer scoring, tuning, parameter selection or final-test execution.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.import_word_review import read_jsonl
from src.icbc_baseline.bm25 import canonical_hash

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_development_run(dev,results,test_ids):
    expected={x['query_id']:x['query'] for x in dev}
    ids=[x['query_id'] for x in results]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
        raise ValueError('本次结果须恰好覆盖开发题，不得重复、缺失或混入其他题')
    if set(ids)&set(test_ids):raise ValueError('本次结果含最终测试题，停止汇总')
    for r in results:
        if r['query']!=expected[r['query_id']]:raise ValueError('运行问题与开发快照不一致')
        if r.get('retrieval_only') is not True:raise ValueError('须为仅检索运行')

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True,type=Path)
    parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args(argv)
    snap=args.snapshot.resolve();run=args.run.resolve()
    if not snap.is_relative_to(ROOT/'data/evaluation/reviewed') or not run.is_relative_to(ROOT/'outputs/runs'):
        raise ValueError('仅处理项目内核验版本和运行记录')
    manifest=json.loads((snap/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        if sha(snap/name)!=digest:raise ValueError('导入版本文件变化：'+name)
    metadata=json.loads((run/'run_metadata.json').read_text(encoding='utf-8'))
    if metadata['corpus_sha256']!=manifest['corpus_sha256']:raise ValueError('本次运行语料版本不一致')
    if metadata['external_model_calls']!=0:raise ValueError('预期离线基线运行')
    dev=read_jsonl(snap/'development.jsonl');results=read_jsonl(run/'retrieval.jsonl')
    verify_development_run(dev,results,manifest['question_ids']['test'])
    if metadata['query_count']!=len(dev):raise ValueError('运行元数据题数不一致')
    corpus=read_jsonl(ROOT/'data/processed/corpus.jsonl')
    rmap={c['chunk_id']:f'R{i:02d}' for i,c in enumerate(corpus,1)}
    byid={x['query_id']:x for x in results}
    checks=[]
    for d in dev:
        r=byid[d['query_id']]
        ranks={h['chunk_id']:h['rank'] for h in r['results']}
        checks.append({'query_id':d['query_id'],'query':d['query'],
          'retrieved_ids':[x['chunk_id'] for x in r['results']],
          'retrieved_reference_ids':[rmap[x['chunk_id']] for x in r['results']],
          'submitted_reference_ids':d['reference_ids'],
          'submitted_reference_ranks':{rid:ranks.get(cid) for rid,cid in zip(d['reference_ids'],d['reference_chunk_ids'])},
          'answer_review_status':d['answer_review_status'],'is_answer_score':False})
    summary={'checked_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
      'snapshot':snap.relative_to(ROOT).as_posix(),'run':run.relative_to(ROOT).as_posix(),
      'development_queries_run':len(results),'final_test_queries_run':0,'excluded_queries_run':0,
      'external_model_calls':0,'formal_scoring_performed':False,
      'source_word_sha256':manifest['source_sha256'],'corpus_sha256':manifest['corpus_sha256'],
      'development_sha256':sha(snap/'development.jsonl'),'test_sha256_before_and_after':sha(snap/'test.jsonl'),
      'test_file_unchanged':True,'run_output_sha256':sha(run/'retrieval.jsonl'),
      'notice':'片段列表尚未区分必需、补充和可替代依据；排名仅供人工查看，不计算答案准确率或总体召回率。',
      'queries':checks}
    output=ROOT/'outputs/checks/word_development_audit_20261017.json'
    output.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 50题导入核查与开发题检索记录','',
      '**日期：2026-10-17。** 已导入返回Word的50道题，保存原文件副本与SHA-256。原答案、用途和排除选择均未由AI修改。','',
      '## 采用情况','',
      '| 用途 | 题数 | 本轮操作 |','| --- | ---: | --- |',
      '| 开发 | 30 | 已实际运行原有BM25默认配置，仅返回原文片段 |',
      '| 最终测试 | 16 | 保存快照与哈希，未运行检索或生成 |',
      '| 排除 | 4 | EV041、EV042、EV049、EV050，保留原记录 |','',
      '开发题均为可回答题。最终测试题有8道可回答、2道需澄清、6道资料不足。范围外题按填写选择排除，因此不能用本集合宣称已评估范围外拒答能力。','',
      '根据用户本轮澄清，文档中冻结时间、开发回归、模型检查及运行次数等附带说明不作为已完成事实。本轮确实完成的程序运行以下述日志为准。核验人和复核人仅按返回文档记载，不额外推断工作量或贡献比例。','',
      '## 建议复核的答案措辞','',
      '以下是AI依据固定原文提出的核查提示，不是新确认答案，也未写回原文档。它们不影响问题本身用于开发检索，但正式答案评分前应明确采用文本。','',
      '| 题号 | 问题 | 建议 |','| --- | --- | --- |']
    findings=json.loads((snap/'quality_findings.json').read_text(encoding='utf-8'))
    for item in findings['answer_findings']:
        lines.append('| '+item['query_id']+' | '+item['issue']+' | '+item['suggestion']+' |')
    lines+=['','## 检索检查的范围','',
      f'实际运行目录：`{run.relative_to(ROOT).as_posix()}`。配置保持原值：k1=1.2，b=0.75，char_bigram，top_k=3，随机种子0；外部模型调用0次。没有根据这些结果自动调参。','',
      '下表只是原文片段的逐题排名核对。确认依据可能包含补充、互补或可替代片段，例如EV037的复核说明明确R24为补充。尚未确认统一标注口径，因此不把所有依据机械要求同时命中，也不计算总体Recall或答案准确率。','',
      '| 题号 | 原问题 | 实际召回的R编号 按排名 | 填写的参考依据及其排名 |',
      '| --- | --- | --- | --- |']
    for c in checks:
        positions='；'.join(k+'：'+('未进前三' if v is None else f'第{v}名') for k,v in c['submitted_reference_ranks'].items())
        lines.append('| '+c['query_id']+' | '+c['query']+' | '+'、'.join(c['retrieved_reference_ids'])+' | '+positions+' |')
    lines+=['','## 后续安排','',
      '1. 保留本轮Word原件与数据快照；建议复核的5处措辞另列，不用AI自动覆盖人工答案。',
      '2. 开发阶段继续使用30道开发题；16道最终测试题不进入提示示例、阈值选择或调参。',
      '3. 按原计划实现向量检索及大模型RAG，保存检索原文、生成回答、引用核验、调用量、费用与耗时。当前尚未实现或调用。',
      '4. 正式评分前，由小组确认参考依据的必需、补充或可替代关系和评分口径。EV025、EV026与之前的SMOKE05共享吞卡主题，不能据此宣称全新业务主题泛化；原用途暂保留。',
      '5. 方案及参数确定后统一开展最终评估，基于实际输出进行本人或小组的结果分析。不要把候选起草、AI核查建议或本轮片段排名直接当作本人原创实验设计和结果分析。','',
      '## 文件位置','',
      f'- 数据快照：`{snap.relative_to(ROOT).as_posix()}/`。',
      f'- 原始检索输出：`{run.relative_to(ROOT).as_posix()}/retrieval.jsonl`。',
      '- 本轮核查记录：`outputs/checks/word_development_audit_20261017.json`。',
      '- 工程验证：`outputs/checks/unittest_word_import_20261017.txt`，共30项通过；合成测试夹具不代表人工标注或系统效果。','']
    (ROOT.parent/'50题导入核查与开发检索记录_20261017.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['development_queries_run','final_test_queries_run','excluded_queries_run','test_file_unchanged','formal_scoring_performed']},ensure_ascii=False))

if __name__=='__main__':main()
