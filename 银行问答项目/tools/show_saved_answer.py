"""Display a hash-verified saved development answer, with zero network calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from freeze_final_configuration import verify_freeze
from icbc_rag.common import file_hash


def available_answers():
    pointer = json.loads((ROOT / 'config/evaluation_freeze.json').read_text(encoding='utf-8'))
    folder = (ROOT / pointer['directory']).resolve()
    if not folder.is_relative_to(ROOT) or file_hash(folder / 'freeze_manifest.json') != pointer['freeze_manifest_sha256']:
        raise ValueError('冻结指针或清单不符')
    manifest = verify_freeze(folder)
    prefix = 'outputs/manual_reviews/development_a7f94cb8e0ab/answers/'
    return {Path(name).stem: (ROOT / name, digest)
            for name, digest in manifest['protected_files'].items()
            if name.startswith(prefix) and name.endswith('.json')}


def display(qid, answers, details=False):
    if qid not in answers:
        raise ValueError('只能查看已保存的29道开发回答；缺失题和最终测试题不可查看')
    path, digest = answers[qid]
    if file_hash(path) != digest:
        raise ValueError('已保存回答哈希变化')
    saved = json.loads(path.read_text(encoding='utf-8'))
    result = saved['result']
    print('\n【已有真实调用记录回放】当前不联网，不生成新回答，不新增API费用。')
    print('题号：' + qid + '；原采集记录时间：' + str(saved.get('collected_at', '见原记录')))
    print('问题：' + result['query'])
    print('\n1. 当时检索到的银行原文（固定24片段评估版）')
    for hit in result['retrieval']['results']:
        print(f"[{hit['rank']}] {hit['chunk_id']}，相似度={hit['score']:.4f}，{hit['citation_locator']}")
        if details:
            print(hit['text'])
    messages = result['messages']
    payload = json.loads(messages[1]['content'])
    print('\n2. 当时提供给模型的输入')
    print('问题：' + payload['question'])
    print('证据编号：' + '、'.join(d['chunk_id'] for d in payload['evidence']))
    print('系统提示要求仅依据证据回答并逐条提供原文引文。')
    print('\n3. 当时模型生成的回答及逐字引文')
    answer = result.get('answer')
    if answer:
        print('回答类型：' + answer['status'])
        for claim in answer['claims']:
            print('· ' + claim['text'])
            for citation in claim['citations']:
                print('  依据 ' + citation['chunk_id'] + '：' + citation['quote'])
        if answer.get('message'):
            print(answer['message'])
        print('机械检查：来源编号与逐字引文；语义是否完整支持须另查人工复核记录。')
    else:
        print('该记录没有通过机械检查的回答，保留原状态：' + result['generation_status'])
    provider = result.get('provider_response', {})
    print('\n原调用模型：' + str(provider.get('returned_model', provider.get('requested_model', '未记录'))))
    print('原调用token：' + str(provider.get('usage', {}).get('total_tokens', '未记录')) + '；当前新增token：0')
    print('原始证据：' + path.relative_to(ROOT).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-id')
    parser.add_argument('--details', action='store_true')
    args = parser.parse_args()
    answers = available_answers()
    if args.query_id:
        display(args.query_id.strip().upper(), answers, args.details)
        return 0
    print('免费查看已有RAG真实回答；这是历史记录回放，不是现场生成。')
    print('可查看题号：' + ' '.join(sorted(answers)))
    print('输入题号（如EV009），回车查看；输入exit退出。')
    while True:
        try:
            qid = input('\n题号> ').strip().upper()
        except EOFError:
            return 0
        if qid in {'EXIT', 'QUIT', '退出'}:
            return 0
        try:
            display(qid, answers, args.details)
        except ValueError as exc:
            print(str(exc))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print('未能打开记录：' + str(exc), file=sys.stderr)
        raise SystemExit(1)
