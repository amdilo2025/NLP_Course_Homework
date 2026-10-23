"""Download pinned public ONNX/tokenizer data. Never download executable model code."""
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO='Xenova/bge-small-zh-v1.5'
REVISION='75c43b069aac4d136ba6bc1122f995fedcfd2781'
BASE_REVISION='7999e1d3359715c523056ef9478215996d62a620'
DEST=ROOT/'models/bge-small-zh-v1.5-onnx-int8'
FILES=['onnx/model_quantized.onnx','tokenizer.json','config.json','tokenizer_config.json','README.md','quantize_config.json']

def download(url,target,expected_sha256=None):
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.part')
    digest=hashlib.sha256();size=0
    request=urllib.request.Request(url,headers={'User-Agent':'icbc-course-project/1.0'})
    with urllib.request.urlopen(request,timeout=60) as response,temporary.open('wb') as output:
        while True:
            block=response.read(1024*1024)
            if not block:break
            size+=len(block)
            if size>150*1024*1024:raise ValueError('Unexpected model file size')
            output.write(block);digest.update(block)
    actual=digest.hexdigest()
    if expected_sha256 and actual!=expected_sha256:raise ValueError('Model LFS SHA256 mismatch')
    temporary.replace(target)
    print(json.dumps({'file':target.relative_to(DEST).as_posix(),'bytes':size,'sha256':actual}),flush=True)
    return {'sha256':actual,'bytes':size,'url':url,'upstream_lfs_sha256':expected_sha256}

def main():
    if (DEST/'manifest.json').is_file():
        manifest=json.loads((DEST/'manifest.json').read_text(encoding='utf-8'))
        if manifest['revision']!=REVISION:raise ValueError('Existing model revision differs')
        for name,info in manifest['files'].items():
            if hashlib.sha256((DEST/name).read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Existing model hash mismatch')
        print('Pinned local model verified; no download needed.');return
    api=f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}/onnx?expand=true'
    with urllib.request.urlopen(api,timeout=30) as response:tree=json.load(response)
    model_entry=next(x for x in tree if x['path']=='onnx/model_quantized.onnx')
    expected=model_entry['lfs']['oid']
    if not len(expected)==64:raise ValueError('Missing SHA256 for model weights')
    files={}
    for name in FILES:
        files[name]=download(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',DEST/name,expected if name.startswith('onnx/') else None)
    files['base_model_README.md']=download(f'https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/{BASE_REVISION}/README.md',DEST/'base_model_README.md')
    manifest={'downloaded_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
      'repo':REPO,'revision':REVISION,'base_model':'BAAI/bge-small-zh-v1.5','base_revision':BASE_REVISION,
      'license':'MIT per base model card','variant':'ONNX dynamic int8 quantized conversion',
      'pooling':'CLS','normalization':'L2','remote_python_code_executed':False,'files':files}
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved '+str(DEST/'manifest.json'),flush=True)

if __name__=='__main__':main()
