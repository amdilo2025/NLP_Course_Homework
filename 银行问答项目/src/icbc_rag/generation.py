"""Explicit chat-completions adapter and mechanical citation validation.

Citation membership and quote existence do not establish entailment.
"""
import json
import math
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

STATUSES={'可回答','资料不足','需澄清','范围外'}

class ProviderError(RuntimeError):pass

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise ProviderError('API重定向已停止，避免密钥被发送到其他地址')

def unique_object(pairs):
    out={}
    for key,value in pairs:
        if key in out:raise ValueError('模型JSON包含重复字段')
        out[key]=value
    return out

def compact(text):return re.sub(r'\s+','',unicodedata.normalize('NFC',text))

def validate_answer(raw,hits):
    data=json.loads(raw,object_pairs_hook=unique_object)
    if not isinstance(data,dict) or set(data)!={'status','message','claims'}:raise ValueError('模型输出字段不符合约定')
    status=data['status'];message=data['message'];claims=data['claims']
    if not isinstance(status,str) or status not in STATUSES or not isinstance(message,str) or len(message)>4000:
        raise ValueError('模型状态或说明无效')
    if not isinstance(claims,list) or len(claims)>20:raise ValueError('模型事实列表无效')
    if status=='可回答' and (not claims or message.strip()):raise ValueError('可回答时须逐条给出有引用的事实，不得附无引用说明')
    if status!='可回答' and not message.strip():raise ValueError('未回答时须说明原因或提出澄清问题')
    sources={h['chunk_id']:h for h in hits}
    checked=[]
    for claim in claims:
        if not isinstance(claim,dict) or set(claim)!={'text','evidence'} or not isinstance(claim['text'],str) or not claim['text'].strip():
            raise ValueError('事实条目无效')
        evidence=claim['evidence']
        if not isinstance(evidence,list) or not 1<=len(evidence)<=10:raise ValueError('每条事实必须有证据')
        seen=set();citations=[]
        for ref in evidence:
            if not isinstance(ref,dict) or set(ref)!={'chunk_id','quote'} or not isinstance(ref['chunk_id'],str) or not isinstance(ref['quote'],str):
                raise ValueError('引用结构无效')
            cid=ref['chunk_id'];quote=compact(ref['quote'])
            if cid not in sources:raise ValueError('引用不属于本轮检索证据')
            if (cid,quote) in seen:raise ValueError('同一事实含重复引用')
            seen.add((cid,quote))
            if len(quote)<8 or quote not in compact(sources[cid]['text']):raise ValueError('引文过短或不是对应原文的连续摘录')
            source=sources[cid]
            citations.append({'chunk_id':cid,'quote':ref['quote'],'source_title':source['source_title'],
              'citation_locator':source['citation_locator'],'source_url':source['source_url']})
        checked.append({'text':claim['text'],'citations':citations})
    return {'status':status,'message':message,'claims':checked,
      'citation_membership_checked':True,'quotes_checked':True,
      'semantic_support_status':'requires_human_review',
      'validation_note':'已检查本轮来源与逐字引文；未自动证明事实、条件、例外或否定关系被完整支持。'}

class ChatClient:
    def __init__(self,settings,environment=None,opener=None):
        env=os.environ if environment is None else environment
        key=env.get(settings['key_env'],'').strip()
        base=env.get(settings['base_url_env'],settings.get('default_base_url','')).strip().rstrip('/')
        model=env.get(settings['model_env'],settings.get('default_model','')).strip()
        if not key and environment is None and settings.get('credential_file'):
            from .credentials import load_key,CredentialError
            from .common import project_path
            try:key=load_key(project_path(settings['credential_file']),base) or ''
            except CredentialError as exc:raise ProviderError(str(exc)) from None
        if not key or not base or not model:
            raise ProviderError('尚未配置API服务。请在本机设置ICBC_RAG_BASE_URL、ICBC_RAG_MODEL、ICBC_RAG_API_KEY；当前可使用preview模式。')
        parsed=urllib.parse.urlsplit(base)
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ProviderError('API地址必须为不含凭据、查询参数或片段的HTTPS地址')
        self.endpoint=base if parsed.path.endswith('/chat/completions') else base+'/chat/completions'
        self.key=key;self.model=model;self.settings=settings;self.provider_host=parsed.hostname
        self.opener=opener or urllib.request.build_opener(NoRedirect())
        self.attempts=0;self.successes=0

    def complete(self,messages):
        payload={'model':self.model,'messages':messages,'temperature':self.settings['temperature'],
          'max_tokens':self.settings['max_output_tokens'],'stream':False}
        if self.settings['json_mode']:payload['response_format']={'type':'json_object'}
        if self.settings.get('thinking_mode') is not None:
            if self.provider_host!='api.deepseek.com' or self.settings['thinking_mode'] not in {'enabled','disabled'}:
                raise ProviderError('thinking_mode仅用于DeepSeek官方且须为enabled或disabled')
            payload['thinking']={'type':self.settings['thinking_mode']}
        request=urllib.request.Request(self.endpoint,data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),
          headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},method='POST')
        self.attempts+=1;start=time.perf_counter()
        try:
            with self.opener.open(request,timeout=self.settings['timeout_seconds']) as response:
                body=response.read(2*1024*1024+1)
            if len(body)>2*1024*1024:raise ProviderError('API响应超过大小限制')
            data=json.loads(body)
            choice=data['choices'][0]
            if choice.get('finish_reason')!='stop':raise ProviderError('模型未正常完成输出；不把截断内容作为答案')
            content=choice['message']['content']
            if not isinstance(content,str) or not content.strip():raise ProviderError('模型返回空内容')
            usage=data.get('usage')
            if not isinstance(usage,dict):usage=None
            else:
                usage={k:v for k,v in usage.items() if k in {'prompt_tokens','completion_tokens','total_tokens'} and type(v) is int and v>=0}
            cost=None
            inp=self.settings['input_price_per_million'];out=self.settings['output_price_per_million']
            if usage and 'prompt_tokens' in usage and 'completion_tokens' in usage and inp is not None and out is not None:
                if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in [inp,out]):
                    raise ProviderError('费用单价配置无效')
                cost={'amount':(usage['prompt_tokens']*inp+usage['completion_tokens']*out)/1_000_000,
                  'currency':self.settings['currency'],'kind':'estimate_at_user_configured_flat_rates'}
            self.successes+=1
            return {'content':content,'usage':usage,'estimated_cost':cost,
              'provider_host':self.provider_host,'requested_model':self.model,'returned_model':data.get('model'),
              'latency_ms':(time.perf_counter()-start)*1000}
        except urllib.error.HTTPError as exc:
            raise ProviderError('API请求失败，HTTP状态 '+str(exc.code)+'；未自动重试') from None
        except (urllib.error.URLError,TimeoutError,OSError) as exc:
            raise ProviderError('API连接失败或超时；未自动重试，可能需要核对服务端调用记录') from None
        except (KeyError,IndexError,TypeError,json.JSONDecodeError):
            raise ProviderError('API响应结构不符合chat-completions接口') from None
