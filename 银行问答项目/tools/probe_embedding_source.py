"""Read public model metadata only; no project data is uploaded."""
import json
import urllib.request

url='https://huggingface.co/api/models/Xenova/bge-small-zh-v1.5'
with urllib.request.urlopen(url,timeout=30) as response:
    value=json.load(response)
print(json.dumps({'sha':value['sha'],'files':[x['rfilename'] for x in value['siblings']]},ensure_ascii=False))
