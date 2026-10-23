"""Vector retrieval, explicit prompt preview and optional real API generation."""
import argparse
from datetime import datetime,timezone,timedelta
import json
import platform
import sys
import time
from icbc_baseline.bm25 import canonical_hash
from .common import ROOT,project_path,file_hash,read_jsonl,load_config
from .prompts import build_messages,privacy_issue,system_for_style
from .generation import ChatClient,ProviderError,validate_answer

def development_inputs():
    folder=ROOT/'data/evaluation/reviewed/20261017_word_v1'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    # Only hash the held-out file; its contents never become retrieval or prompt input.
    for name in ['development.jsonl','development_queries.jsonl','test.jsonl']:
        if file_hash(folder/name)!=manifest['files'][name]:raise ValueError('评估版本文件发生变化：'+name)
    rows=read_jsonl(folder/'development_queries.jsonl')
    ids=[x['query_id'] for x in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(manifest['question_ids']['development']) or set(ids)&set(manifest['question_ids']['test']):
        raise ValueError('开发输入必须完整且不得含最终测试题')
    return rows,{'development_queries_sha256':file_hash(folder/'development_queries.jsonl'),
      'test_sha256_before':file_hash(folder/'test.jsonl'),'test_file':folder/'test.jsonl'}

def execute(row,index,encoder,mode,client=None):
    query=row['query'];result={'query_id':row['query_id'],'query':query,'mode':mode,
      'answer':None,'generation_status':'not_requested','external_model_calls':0}
    if not isinstance(query,str) or not query.strip() or len(query)>2000:
        raise ValueError('问题长度须为1至2000字符')
    issue=privacy_issue(query)
    if issue:
        # Do not record sensitive input or include it in provider requests.
        return {**result,'query':'[输入含疑似敏感信息，未记录]','generation_status':'input_blocked','notice':issue}
    start=time.perf_counter()
    vectors,counts=encoder.encode([query],query=True)
    retrieval=index.search_vector(query,vectors[0],encoder.config['top_k'])
    context=retrieval['results']
    if encoder.config.get('expand_article_context'):
        from .retrieval import article_context
        context=article_context(index.documents,retrieval['results'])
        retrieval['context_results']=context
        retrieval['context_note']='补充同来源同条款的已有相邻段落；不作为新的相似度排名'
    result.update({'retrieval':retrieval,'query_tokens':counts[0],
      'retrieval_ms':(time.perf_counter()-start)*1000})
    if mode=='retrieve':return result
    messages=build_messages(query,context,style=encoder.config.get('prompt_style'))
    result['messages']=messages
    result['prompt_sha256']=canonical_hash(messages)
    if mode=='preview':
        result['generation_status']='preview_only_no_model_called'
        return result
    if client is None:raise ProviderError('真实生成模式缺少API客户端')
    before=client.attempts
    try:
        output=client.complete(messages)
        result['provider_response']=output
        try:
            result['answer']=validate_answer(output['content'],context)
            result['generation_status']='generated_citations_checked_semantics_unverified'
        except (ValueError,TypeError) as exc:
            result['generation_status']='rejected_by_citation_or_schema_check'
            result['notice']=str(exc)
    except ProviderError as exc:
        result['generation_status']='provider_error';result['notice']=str(exc)
    result['external_model_calls']=client.attempts-before
    return result

def display(result,details=False):
    print('\n问题：'+result['query'])
    if 'retrieval' in result and (details or result['mode']!='api'):
        for h in result['retrieval']['results']:
            print(f"[{h['rank']}] {h['chunk_id']}  相似度={h['score']:.4f}")
            print('出处：'+h['citation_locator']);print(h['text']);print(h['source_url'])
        print('相似度是排序分数，不是回答正确率。')
        seed_ids={h['chunk_id'] for h in result['retrieval']['results']}
        siblings=[h for h in result['retrieval'].get('context_results',[]) if h['chunk_id'] not in seed_ids]
        if siblings:
            print('\n补充的同条款上下文：')
            for h in siblings:
                print(h['chunk_id']+'  '+h['citation_locator']);print(h['text'])
    if result['mode']=='preview' and 'messages' in result:
        print('\n模型请求预览：')
        print(json.dumps({'messages':result['messages']},ensure_ascii=False,indent=2))
        print('当前仅预览，尚未调用大模型，未生成回答。')
    answer=result.get('answer')
    if answer:
        print('\n答复：')
        sources={}
        for n,claim in enumerate(answer['claims'],1):
            labels=[]
            for ref in claim['citations']:
                if ref['chunk_id'] not in sources:sources[ref['chunk_id']]=(len(sources)+1,ref)
                label=sources[ref['chunk_id']][0]
                if label not in labels:labels.append(label)
            marker=''.join(f'[{i}]' for i in labels)
            print(f'{n}. {claim["text"]} {marker}')
        if answer['message']:
            print(('\n需要确认：' if answer['claims'] else '')+answer['message'])
        if sources:
            print('\n依据：');urls=set()
            for number,ref in sources.values():
                print(f'[{number}] {ref["source_title"]}，{ref["citation_locator"]}')
                if ref['source_url'] not in urls:
                    print(ref['source_url']);urls.add(ref['source_url'])
                if details:print('原文引文：'+ref['quote'])
        if details:print(answer['validation_note'])
        else:print('\n输入 /详情 查看检索原文与引用检查，不会再次调用模型。')
    elif result['mode']=='api' and not result.get('notice'):
        print('本次没有可展示的模型回答。')
    if result.get('notice'):print(result['notice'])

def main(argv=None):
    parser=argparse.ArgumentParser(description='本地中文向量检索与RAG；默认仅预览，不调用API')
    parser.add_argument('--config',type=str,default=str(ROOT/'config/rag.json'))
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--query');source.add_argument('--development',action='store_true')
    source.add_argument('--build-only',action='store_true')
    parser.add_argument('--mode',choices=['retrieve','preview','api'],default='preview')
    parser.add_argument('--rebuild-index',action='store_true')
    parser.add_argument('--details',action='store_true',help='同时展开检索原文和引用检查')
    args=parser.parse_args(argv)
    try:
        config=load_config(args.config)
        if args.development and config.get('evaluation_dataset_compatible') is False:
            raise ValueError('此配置是扩展交互语料，不能直接使用旧评估标签；正式评估请使用config/rag.json并核对版本。')
        # Fail early, without loading a model or sending requests, when API is absent.
        client=ChatClient(config['api']) if args.mode=='api' else None
        from .embeddings import LocalBGE
        from .retrieval import prepare_index
        started=time.perf_counter();encoder=LocalBGE(config)
        index,built=prepare_index(config,encoder,rebuild=args.rebuild_index)
        load_ms=(time.perf_counter()-started)*1000
        print(f'本地向量检索就绪：{len(index.documents)}个原文片段，{index.vectors.shape[1]}维，CPU，ONNX int8。')
        print('模式：'+args.mode+'；'+('允许真实API请求' if client else '无外部模型调用'))
        if args.build_only:
            print('索引已校验并保存。');return 0
        partition={}
        if args.development:rows,partition=development_inputs()
        elif args.query is not None:rows=[{'query_id':'USER001','query':args.query}]
        else:rows=None
        now=datetime.now(timezone(timedelta(hours=8)))
        folder=project_path(config['run_dir'])/now.strftime('%Y%m%dT%H%M%S_%f')
        folder.mkdir(parents=True,exist_ok=False)
        actual=[]
        try:
            with (folder/'results.jsonl').open('w',encoding='utf-8') as stream:
                def run(row):
                    answer=execute(row,index,encoder,args.mode,client)
                    actual.append(answer)
                    stream.write(json.dumps(answer,ensure_ascii=False)+'\n');stream.flush()
                    display(answer,details=args.details)
                if rows is not None:
                    for row in rows:run(row)
                else:
                    print('输入公开业务规则问题；/详情 查看上次原文，exit退出。')
                    while True:
                        try:query=input('\n问题> ').strip()
                        except EOFError:break
                        if query.lower() in {'exit','quit','退出'}:break
                        if query.lower() in {'/详情','/原文','/details','/sources'}:
                            if actual:display(actual[-1],details=True)
                            else:print('还没有可查看的问题。')
                            continue
                        if query:
                            try:run({'query_id':f'USER{len(actual)+1:03d}','query':query})
                            except ValueError as exc:print('本题未执行：'+str(exc))
        finally:
            unchanged=None
            if partition:unchanged=file_hash(partition['test_file'])==partition['test_sha256_before']
            meta={'started_at':now.isoformat(),'run_kind':'development_'+args.mode if args.development else 'user_'+args.mode,
              'mode':args.mode,'query_count':len(actual),'query_ids':[x['query_id'] for x in actual],
              'config':config,'config_sha256':canonical_hash(config),'corpus_sha256':index.corpus_sha256,
              'encoder':encoder.info,'encoder_fingerprint':encoder.fingerprint,'model_load_and_index_ms':load_ms,
              'index_built':built,'prompt_template_sha256':canonical_hash(system_for_style(config.get('prompt_style'))),
              'code_sha256':{p.name:file_hash(p) for p in (ROOT/'src/icbc_rag').glob('*.py')},
              'query_inputs_sha256':canonical_hash([{'query_id':x['query_id'],'query':x['query']} for x in actual]),
              'python_version':platform.python_version(),'external_model_calls':client.attempts if client else 0,
              'api_response_successes':client.successes if client else 0,
              'accepted_generated_answers':sum(x['answer'] is not None for x in actual),
              'test_file_unchanged':unchanged,'formal_scoring_performed':False,
              'development_queries_sha256':partition.get('development_queries_sha256'),
              'test_sha256_before':partition.get('test_sha256_before')}
            (folder/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        if partition and not unchanged:raise ValueError('最终测试快照变化，停止使用本次输出')
        print('\n本次记录：'+str(folder.relative_to(ROOT)))
        if any(x['generation_status'] in {'provider_error','rejected_by_citation_or_schema_check'} for x in actual):return 1
        return 0
    except (ValueError,OSError,KeyError,ImportError,ProviderError) as exc:
        print('未完成：'+str(exc),file=sys.stderr)
        return 1
