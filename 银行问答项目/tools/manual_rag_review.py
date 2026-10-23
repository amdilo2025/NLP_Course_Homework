"""User-operated development collection. Never automatically call an API.

--prepare-only imports compatible existing answers and writes local materials.
In interactive mode, each *new* request requires the exact command '生成'.
"""
import argparse
import copy
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.rag_packages').is_dir():sys.path.insert(0,str(ROOT/'.rag_packages'))
sys.path.insert(0,str(ROOT/'src'))
from icbc_baseline.bm25 import canonical_hash
from icbc_rag.common import file_hash,load_config,read_jsonl,project_path
from icbc_rag.cli import development_inputs,execute,display
from icbc_rag.generation import ChatClient,ProviderError,validate_answer
from icbc_rag.prompts import system_for_style,privacy_issue

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)

def matching_cache(record,metadata,query,config,documents):
    if metadata.get('mode')!='api' or metadata.get('corpus_sha256')!=config['corpus_sha256']:
        return False
    if metadata.get('config_sha256')!=canonical_hash(config):return False
    if metadata.get('prompt_template_sha256')!=canonical_hash(system_for_style(config.get('prompt_style'))):return False
    if record.get('query')!=query or not record.get('answer') or record.get('generation_status')!='generated_citations_checked_semantics_unverified':return False
    messages=record.get('messages')
    if not messages or canonical_hash(messages)!=record.get('prompt_sha256'):return False
    if messages[0].get('content')!=system_for_style(config.get('prompt_style')):return False
    try:
        prompt=json.loads(messages[1]['content'])
        if prompt['question']!=query:return False
        hits=record['retrieval']['results'];byid={d['chunk_id']:d for d in documents}
        if [d['chunk_id'] for d in prompt['evidence']]!=[d['chunk_id'] for d in hits]:return False
        for hit in hits:
            if hit['chunk_id'] not in byid or hit['text']!=byid[hit['chunk_id']]['text']:return False
        for item in prompt['evidence']:
            if set(item)!={'chunk_id','text','source_title','source_version','scope','citation_locator'}:return False
            if any(item[k]!=byid[item['chunk_id']][k] for k in item):return False
        validated=validate_answer(record['provider_response']['content'],hits)
        if record['answer']!=validated:return False
    except (KeyError,TypeError,ValueError):return False
    return True

def find_cached_answers(queries,config,documents):
    wanted={x['query']:x['query_id'] for x in queries};found={}
    for folder in sorted((ROOT/'outputs/rag_runs').glob('*'),reverse=True):
        meta_file=folder/'run_metadata.json';answer_file=folder/'results.jsonl'
        if not meta_file.is_file() or not answer_file.is_file():continue
        try:
            meta=json.loads(meta_file.read_text(encoding='utf-8'))
            for record in read_jsonl(answer_file):
                qid=wanted.get(record.get('query'))
                if not qid or qid in found:continue
                if matching_cache(record,meta,record['query'],config,documents):
                    found[qid]={'query_id':qid,'query':record['query'],'collected_at':now(),
                      'origin':'reused_existing_real_response','new_external_model_calls':0,
                      'source_run':folder.relative_to(ROOT).as_posix(),
                      'source_result_sha256':file_hash(answer_file),'source_query_id':record.get('query_id'),
                      'result':record,'manual_notes':None,'manual_reviewer':None,'manual_reviewed_at':None}
        except (OSError,ValueError,KeyError,TypeError):continue
    return found

def generation_allowed(command,offline=False,already_saved=False,pending=False):
    return command.strip()=='生成' and not offline and not already_saved and not pending

def prepare(config):
    queries,partition=development_inputs()
    documents=read_jsonl(project_path(config['corpus_file']))
    snapshot=ROOT/'data/evaluation/reviewed/20261017_word_v1'
    manifest=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    if config.get('evaluation_dataset_compatible') is False or config['corpus_sha256']!=manifest['corpus_sha256'] or canonical_hash(documents)!=manifest['corpus_sha256']:
        raise ValueError('采集器只接受固定24片段和30道开发题，不能混入交互扩展语料')
    for name,digest in manifest['files'].items():
        if file_hash(snapshot/name)!=digest:raise ValueError('核验数据快照发生变化')
    # Environment overrides would alter the model/protocol; require an explicit new config instead.
    for env_name,default_name in [(config['api']['base_url_env'],'default_base_url'),(config['api']['model_env'],'default_model')]:
        if os.environ.get(env_name) and os.environ[env_name].rstrip('/')!=config['api'].get(default_name,'').rstrip('/'):
            raise ValueError('本机API地址或模型覆盖与固定配置不同，请先登记新配置版本')
    fingerprint=canonical_hash({'config':config,'queries':queries,'prompt':system_for_style(config.get('prompt_style'))})
    folder=ROOT/'outputs/manual_reviews'/('development_'+fingerprint[:12])
    folder.mkdir(parents=True,exist_ok=True)
    meta_file=folder/'manifest.json'
    if meta_file.exists():
        info=json.loads(meta_file.read_text(encoding='utf-8'))
        if info['session_fingerprint']!=fingerprint:raise ValueError('采集会话配置冲突')
    else:
        info={'created_at':now(),'session_fingerprint':fingerprint,'config':config,
          'corpus_sha256':config['corpus_sha256'],'query_ids':[x['query_id'] for x in queries],
          'test_sha256_before':partition['test_sha256_before'],'formal_scoring_performed':False,
          'answers_saved':0,'reused_answers':0,'new_api_attempts':0,'pending_attempts':[]}
        write_json(meta_file,info)
    allowed={x['query_id']:x['query'] for x in queries}
    saved={}
    for path in (folder/'answers').glob('EV*.json'):
        item=json.loads(path.read_text(encoding='utf-8'))
        if item['query_id'] not in allowed:raise ValueError('采集文件出现非开发题')
        if item.get('query')!=allowed[item['query_id']] or item.get('result',{}).get('query')!=item['query']:
            raise ValueError('采集文件的问题与固定开发题不一致')
        saved[item['query_id']]=item
    for qid,item in find_cached_answers(queries,config,documents).items():
        if qid not in saved:
            saved[qid]=item;write_json(folder/'answers'/f'{qid}.json',item)
    return queries,partition,folder,info,saved

def save_progress(folder,info,saved,queries,partition):
    if file_hash(partition['test_file'])!=info['test_sha256_before']:raise ValueError('最终测试快照发生变化')
    info.update({'updated_at':now(),'answers_saved':len(saved),
      'reused_answers':sum(x['origin']=='reused_existing_real_response' for x in saved.values()),
      'new_api_attempts':sum(x.get('new_external_model_calls',0) for x in saved.values())+len(info.get('pending_attempts',[])),
      'test_file_unchanged':True,'final_test_queries_run':0,'formal_scoring_performed':False})
    write_json(folder/'manifest.json',info)
    lines=['# 开发题手动采集进度','',
      '只用于开发题采集与人工备注，不是正式效果评分。交互入口只有输入“生成”才发新请求，回车不调用；另有2026-10-21用户限定授权的代理采集，授权及实际次数见outputs/agent_collection_plans。已有兼容回答优先复用。',
      '',f'已保存回答 {len(saved)} / {len(queries)}；其中复用旧回答 {info["reused_answers"]}。新请求记录 {info["new_api_attempts"]} 次。',
      '未完成的请求可能已被平台计费，不自动重试。','',
      '| 题号 | 问题 | 状态 | 人工备注 |','| --- | --- | --- | --- |']
    for q in queries:
        item=saved.get(q['query_id'])
        state='未采集' if item is None else '复用已有回答' if item['origin']=='reused_existing_real_response' else item['result']['generation_status']
        note=(item or {}).get('manual_notes') or '未填写'
        lines.append('| '+q['query_id']+' | '+q['query']+' | '+state+' | '+note.replace('|','／').replace('\n',' ')+' |')
    lines+=['','这是自动生成的进度索引，请在采集程序中填写备注，不直接编辑此表。',
      '完整模型回答、来源、token用量及原始调用来源存于answers目录；它们没有被自动判分。','']
    (folder/'采集进度.md').write_text('\n'.join(lines),encoding='utf-8')

def main(argv=None):
    parser=argparse.ArgumentParser(description='由用户逐题触发真实调用；启动和准备不会调用API')
    parser.add_argument('--prepare-only',action='store_true',help='只整理已有回答与待采集列表，无API客户端')
    parser.add_argument('--offline',action='store_true',help='禁用真实生成，仅查看已有结果')
    args=parser.parse_args(argv)
    config=load_config(ROOT/'config/rag.json')
    queries,partition,folder,info,saved=prepare(config)
    client=None;encoder=None;index=None
    try:
        save_progress(folder,info,saved,queries,partition)
        print(f'固定24片段，30道开发题；已复用或保存{len(saved)}份回答。启动不调用API。')
        print('记录位置：'+str(folder.relative_to(ROOT)))
        if args.prepare_only:return 0
        print('输入题号查看或采集；输入列表查看题目，exit退出。')
        byid={x['query_id']:x for x in queries}
        while True:
            try:qid=input('\n开发题号> ').strip().upper()
            except EOFError:break
            if qid.lower() in {'exit','quit','退出'}:break
            if qid in {'列表','LIST','?'}:
                for q in queries:print(q['query_id']+' '+q['query']+(' [已保存，不重复调用]' if q['query_id'] in saved else ''))
                continue
            if qid not in byid:print('仅接受本批30道开发题号；最终测试题不会进入此入口。');continue
            row=byid[qid];print(row['query'])
            if qid in saved:display(saved[qid]['result'])
            else:
                pending=qid in info.get('pending_attempts',[])
                if pending:print('此题有未完成请求，可能已计费；请核对平台记录，本入口不重复提交。');continue
                try:command=input('输入“生成”才调用1次API；回车跳过> ')
                except EOFError:break
                if not generation_allowed(command,args.offline,False,pending):
                    print('未调用API。');continue
                # The only path to a new provider request is this explicit user command.
                try:
                    if encoder is None:
                        from icbc_rag.embeddings import LocalBGE
                        from icbc_rag.retrieval import prepare_index
                        encoder=LocalBGE(config);index,_=prepare_index(config,encoder)
                    if client is None:client=ChatClient(config['api'])
                except (ValueError,OSError,ProviderError) as exc:
                    print('尚未发送请求：'+str(exc));continue
                info.setdefault('pending_attempts',[]).append(qid)
                save_progress(folder,info,saved,queries,partition)
                result=execute(row,index,encoder,'api',client)
                item={'query_id':qid,'query':row['query'],'collected_at':now(),'origin':'user_confirmed_new_request',
                  'new_external_model_calls':result['external_model_calls'],'result':result,
                  'manual_notes':None,'manual_reviewer':None,'manual_reviewed_at':None}
                write_json(folder/'answers'/f'{qid}.json',item)
                saved[qid]=item;info['pending_attempts'].remove(qid)
                save_progress(folder,info,saved,queries,partition);display(result)
            try:note=input('人工备注（可留空；不会自动判分）> ').strip()
            except EOFError:break
            if note:
                if privacy_issue(note):print('备注疑似含敏感信息，未保存；请移除后再写。');continue
                reviewer=input('备注填写人（可留空）> ').strip()
                saved[qid].update({'manual_notes':note,'manual_reviewer':reviewer or None,'manual_reviewed_at':now()})
                write_json(folder/'answers'/f'{qid}.json',saved[qid])
                save_progress(folder,info,saved,queries,partition)
    except KeyboardInterrupt:
        print('\n已停止。中断中的请求可能已计费，不自动重试。')
    finally:save_progress(folder,info,saved,queries,partition)
    return 0

if __name__=='__main__':raise SystemExit(main())
