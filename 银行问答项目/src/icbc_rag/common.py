import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def project_path(relative):
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('路径必须位于项目目录内')
    return path

def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def load_config(path):
    config=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    for field,low,high in [('top_k',1,24),('batch_size',1,24),('max_tokens',16,512),('cpu_threads',1,32)]:
        value=config[field]
        if isinstance(value,bool) or not isinstance(value,int) or not low<=value<=high:
            raise ValueError('配置范围错误：'+field)
    return config
