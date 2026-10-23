"""Lossless DOCX review import. Document text is data, never executable instructions."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.export_reviewed_eval import HEADERS, validate_rows, normalized
from src.icbc_baseline.bm25 import canonical_hash

NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
PREFIXES=['确认问题','确认答案','处理类型','确认依据','核验状态','已用于调试','核验人与日期','复核人与说明']

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def reference_ids(text, known):
    result=[]
    # Range expansion must precede token matching: R07–R15 includes R08..R14.
    def expand(m):
        first,last=int(m[1]),int(m[2])
        if not 1 <= first <= last <= len(known):
            raise ValueError('依据范围无效：'+m[0])
        return '、'.join(f'R{i:02d}' for i in range(first,last+1))
    expanded=re.sub(r'\bR(\d{2})\s*[–—－-]\s*R?(\d{2})\b',expand,text)
    for token in re.findall(r'\bR\d{2}\b',expanded):
        if token not in known: raise ValueError('未知依据编号：'+token)
        if token not in result:result.append(token)
    return result

def paragraphs(data):
    with ZipFile(__import__('io').BytesIO(data)) as z:
        if z.testzip():raise ValueError('Word ZIP校验失败')
        tree=ET.fromstring(z.read('word/document.xml'))
    if tree.findall('.//w:ins',NS) or tree.findall('.//w:del',NS):
        raise ValueError('文档仍含修订，请先明确采用哪个文本版本')
    return [''.join(e.text or '' for e in p.findall('.//w:t',NS)).strip()
            for p in tree.findall('.//w:body//w:p',NS)]

def parse_blocks(lines):
    header=[]; blocks=[];current=None;last=None
    for line in lines:
        if line.startswith('附录'):break
        m=re.match(r'^(EV\d{3})\s+(.+)$',line)
        if m:
            current={'question_id':m[1],'original_question':m[2],'raw_fields':{}}
            blocks.append(current);last=None;continue
        if current is None:header.append(line);continue
        found=False
        for key in PREFIXES:
            if line.startswith(key+'：'):
                if key in current['raw_fields']:raise ValueError('重复字段 '+current['question_id']+' '+key)
                current['raw_fields'][key]=line[len(key)+1:].strip()
                last=key;found=True;break
        if not found and line and last and not line.startswith(('G','本页','候选依据')) and set(line) != {'_'}:
            current['raw_fields'][last]+='\n'+line
    return header,blocks

def normalize_blocks(blocks,candidates,corpus,as_of):
    known={f'R{i:02d}':c['chunk_id'] for i,c in enumerate(corpus,1)}
    originals={c['question_id']:c for c in candidates}
    ids=[b['question_id'] for b in blocks]
    if len(ids)!=len(set(ids)) or set(ids)!=set(originals):
        raise ValueError('须完整包含本批50个题号且不得重复')
    rows=[];extra={}
    for b in blocks:
        qid=b['question_id'];c=originals[qid];f=b['raw_fields']
        if b['original_question'] != c['question_draft']:raise ValueError(qid+' 原题变化，无法保证对齐')
        if any(not f.get(k) for k in PREFIXES):raise ValueError(qid+' 核验字段缺失')
        state=re.fullmatch(r'(待核验|已确认|需修改|排除)\s+用途：\s*(开发|最终测试|测试|排除|待定)',f['核验状态'])
        who=re.fullmatch(r'(.+?)\s+(\d{4}-\d{2}-\d{2})',f['核验人与日期'])
        use=re.match(r'^(是|否)(?:[。\s]|$)',f['已用于调试'])
        if not state or not who or not use:raise ValueError(qid+' 状态、日期或调试使用字段格式不明确')
        checked=date.fromisoformat(who[2])
        if checked>as_of:raise ValueError(qid+' 核验日期晚于导入日期')
        rids=reference_ids(f['确认依据'],known)
        row=dict.fromkeys(HEADERS,'')
        row.update({'题目ID':qid,'题组':c['group_id'],'类别':c['category'],'问题草稿':c['question_draft'],
            '核验状态':state[1],'确认问题':f['确认问题'],'确认答案':f['确认答案'],'确认类型':f['处理类型'],
            '确认依据ID':'、'.join(known[x] for x in rids),'用途':'测试' if state[2]=='最终测试' else state[2],
            # Conservative import for leakage checks; a submission claim is not a verified run log.
            '已用于调试':use[1],'核验人':who[1].strip(),'核验日期':checked.isoformat(),
            '修改说明':f['复核人与说明']})
        if row['核验状态']=='排除' and row['用途']!='排除':raise ValueError(qid+' 排除状态与用途不一致')
        rows.append(row)
        extra[qid]={'submitted_review':f,'reference_ids':rids,'reference_chunk_ids':[known[x] for x in rids]}
    return rows,extra

def main(argv=None):
    parser=argparse.ArgumentParser(description='导入返回的Word核验表，不运行检索或最终测试')
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--as-of',required=True,type=date.fromisoformat)
    args=parser.parse_args(argv)
    target=args.output.resolve()
    if not target.is_relative_to(ROOT/'data/evaluation/reviewed'):raise ValueError('输出必须位于项目reviewed目录')
    if target.exists():raise ValueError('输出版本已存在，不覆盖')
    source=args.input.read_bytes()
    folder=ROOT/'data/evaluation'
    prep=json.loads((folder/'preparation_manifest_20261015.json').read_text(encoding='utf-8'))
    candidates=read_jsonl(folder/prep['candidate_file']);corpus=read_jsonl(ROOT/'data/processed/corpus.jsonl')
    if canonical_hash(candidates)!=prep['candidate_sha256'] or canonical_hash(corpus)!=prep['corpus_sha256']:
        raise ValueError('候选或固定语料版本变化')
    header,blocks=parse_blocks(paragraphs(source))
    rows,extras=normalize_blocks(blocks,candidates,corpus,args.as_of)
    smoke={normalized(x['query']) for x in read_jsonl(ROOT/'examples/smoke_queries.jsonl')}
    accepted=validate_rows(rows,candidates,corpus,smoke)
    byid={x['query_id']:x for x in accepted}
    findings=json.loads((ROOT/'config/word_review_findings_20261017.json').read_text(encoding='utf-8'))
    flagged={x['query_id'] for x in findings['answer_findings']}
    result=[]
    for row in rows:
        qid=row['题目ID']; e=extras[qid]
        record=byid.get(qid,{'query_id':qid,'group_id':row['题组'],'query':row['问题草稿'] if row['确认问题']=='保留原题' else row['确认问题'],
          'reference_answer':row['确认答案'],'answer_type':row['确认类型'],'split':'excluded',
          'reviewer':row['核验人'],'review_date':row['核验日期']})
        record.pop('gold_chunk_ids',None)
        record.pop('used_for_tuning',None)
        record.update(e)
        record.update({'review_status_submitted':row['核验状态'],
          'answer_review_status':'requires_recheck' if qid in flagged else 'submitted_confirmed' if record['split']!='excluded' else 'excluded',
          'reference_roles_status':'unspecified_supporting_or_alternative_not_automatically_all_required',
          'formal_scoring_eligible':False,'prior_run_claims_verified':False,
          'workflow_note':'按用户2026-10-17澄清，附带冻结时间、回归与运行次数不作为已完成事实；保留原文备查。'})
        result.append(record)
    target.mkdir(parents=True,exist_ok=False)
    (target/'source_completed.docx').write_bytes(source)
    def write_records(name,items):
        (target/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items),encoding='utf-8')
    write_records('all_submitted.jsonl',result)
    for split in ['development','test','excluded']:
        write_records(split+'.jsonl',[x for x in result if x['split']==split])
    write_records('development_queries.jsonl',[{'query_id':x['query_id'],'query':x['query']} for x in result if x['split']=='development'])
    (target/'quality_findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'import_date':args.as_of.isoformat(),'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
      'source_file':args.input.name,'source_sha256':sha(source),'corpus_sha256':prep['corpus_sha256'],
      'candidate_sha256':prep['candidate_sha256'],'record_count':len(result),
      'split_counts':dict(Counter(x['split'] for x in result)),
      'type_counts_by_split':{s:dict(Counter(x['answer_type'] for x in result if x['split']==s)) for s in ['development','test','excluded']},
      'question_ids':{'development':[x['query_id'] for x in result if x['split']=='development'],'test':[x['query_id'] for x in result if x['split']=='test']},
      'answer_recheck_ids':sorted(flagged),'formal_scoring_ready':False,'test_queries_run':0,
      'source_members_and_division':[x for x in header if x.startswith(('成员','本轮分工'))],
      'prior_workflow_assertions':'retained_as_source_text_not_verified_completion',
      'files':{p.name:sha(p.read_bytes()) for p in target.iterdir() if p.is_file()}}
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(target),'count':50,'splits':manifest['split_counts'],'answer_recheck_ids':sorted(flagged),'test_queries_run':0},ensure_ascii=False))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
