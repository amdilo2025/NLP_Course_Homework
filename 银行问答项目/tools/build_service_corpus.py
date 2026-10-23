"""Extend the interactive corpus with verified application paragraphs.

The frozen evaluation corpus and reviewed questions are never overwritten.
"""
import copy
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from icbc_baseline.bm25 import canonical_hash
from icbc_baseline.cleaning import source_text,parse_articles,comparable,article_pages,normalize_paragraph

def main():
    sources_dir=ROOT.parent/'公开资料/工商银行'
    sources={x['source_id']:x for x in json.loads((sources_dir/'verified_sources_20261015.json').read_text(encoding='utf-8'))}
    primary=source_text(sources_dir,sources['S001'])
    mirror=source_text(sources_dir,sources['S003'])
    p_articles=parse_articles(primary);m_articles=parse_articles(mirror)
    pages=article_pages(primary)
    original_path=ROOT/'data/processed/corpus.jsonl'
    original=[json.loads(x) for x in original_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    frozen=json.loads((ROOT/'data/evaluation/preparation_manifest_20261015.json').read_text(encoding='utf-8'))
    if canonical_hash(original)!=frozen['corpus_sha256']:raise ValueError('原评估语料已变化，停止扩展')
    docs=copy.deepcopy(original);added=[]
    source=sources['S001']
    for number in [7,8,9]:
        if comparable(p_articles[number]['text'])!=comparable(m_articles[number]['text']):
            raise ValueError(f'第{number}条PDF与HTML不一致')
        article=m_articles[number]
        paragraphs=[p.strip() for p in article['text'].splitlines() if p.strip()]
        for i,text in enumerate(paragraphs,1):
            text=normalize_paragraph(re.sub(r'^第[一二三四五六七八九十]+条\s*','',text))
            first,last=pages[number]
            label=f'PDF第{first}页' if first==last else f'PDF第{first}—{last}页'
            doc={'chunk_id':f'S001-A{number:02d}-P{i:02d}','source_id':'S001',
              'title':f'{article["label"]} / 原文第{i}段','text':text,
              'source_title':source['title'],'source_url':source['url'],'source_version':source['version'],
              'effective_date':source.get('effective_date',''),'scope':source['scope'],
              'citation_locator':f'{article["label"]}，原文第{i}段；{label}',
              'approved_record_ids':[], 'augmentation_record_id':f'APP-{number:02d}-{i:02d}',
              'review_status':'AI已核对PDF与HTML；人工独立复核尚未记录',
              'conditions':['须符合发卡条件并经银行审核，不能承诺一定办成。','指定线上渠道不等于所有卡种均可线上申请，也不提供未经核实的App菜单。','不推断年龄、完整补充材料、收费或当日领卡保证。'],
              'raw_file':source['raw_file'],'raw_sha256':source['sha256'],
              'text_kind':'清洗后的官方原文，不是AI摘要','article_number':number,'paragraph_number':i,
              'pdf_page_start':first,'pdf_page_end':last,'formatting_mirror_id':'S003',
              'transformations':['去除条号与排版空白；条号另存出处'],
              'corpus_profile':'interactive_service_20261020'}
            docs.append(doc);added.append(doc['chunk_id'])
    if len({x['chunk_id'] for x in docs})!=len(docs):raise ValueError('扩展片段ID重复')
    corpus_path=ROOT/'data/processed/corpus_service_20261020.jsonl'
    config_path=ROOT/'config/service_rag.json'
    config=json.loads((ROOT/'config/rag.json').read_text(encoding='utf-8'))
    config.update({'corpus_file':corpus_path.relative_to(ROOT).as_posix(),'corpus_sha256':canonical_hash(docs),
      'index_file':'data/processed/dense_index_service_20261020.json','run_dir':'outputs/rag_service_runs',
      'prompt_style':'service_steps_v2','expand_article_context':True,'evaluation_dataset_compatible':False})
    corpus_path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in docs),encoding='utf-8')
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
      'kind':'interactive_service_augmentation_not_frozen_evaluation',
      'original_chunks':len(original),'added_chunks':len(added),'total_chunks':len(docs),
      'added_chunk_ids':added,'cross_checked_articles':[7,8,9],
      'base_corpus_sha256':canonical_hash(original),'service_corpus_sha256':canonical_hash(docs),
      'source_pdf_sha256':source['sha256'],'new_external_model_calls':0,
      'reason':'用户手动提问怎么办理借记卡，原24片段未覆盖第八条申请和启用规则。',
      'note':'仅扩展交互试用版；旧语料和核验题不改，原有测试标签不自动适用于新语料。新增原文由AI核对，不伪记人工核验。'}
    (ROOT/'data/processed/service_augmentation_20261020.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
