"""Authenticated read of official model IDs. Never print or record the key."""
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from icbc_rag.common import load_config
from icbc_rag.generation import ChatClient,ProviderError

def main():
    config=load_config(ROOT/'config/rag.json')
    client=ChatClient(config['api'])
    if client.provider_host!='api.deepseek.com':raise ProviderError('连接检查只允许DeepSeek官方')
    result={'checked_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'provider_host':client.provider_host,
      'endpoint':'https://api.deepseek.com/models','requested_model':client.model,'generations_requested':0}
    request=urllib.request.Request(result['endpoint'],headers={'Authorization':'Bearer '+client.key})
    try:
        with client.opener.open(request,timeout=30) as response:value=json.load(response)
        result.update({'status':'authenticated','model_ids':[x['id'] for x in value['data']]})
        result['selected_model_available']=client.model in result['model_ids']
    except urllib.error.HTTPError as exc:
        result.update({'status':'http_error','http_status':exc.code})
    except (urllib.error.URLError,TimeoutError,OSError):result['status']='connection_error'
    (ROOT/'outputs/checks/deepseek_connection_20261020.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result.get('selected_model_available') else 1

if __name__=='__main__':raise SystemExit(main())
