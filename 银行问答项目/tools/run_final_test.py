"""One-shot final evaluation. Default is offline preflight, never inference."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.rag_packages'), str(ROOT / 'src'), str(ROOT / 'tools')]
from freeze_final_configuration import verify_freeze, apply_reference_errata
from icbc_rag.common import project_path, file_hash, read_jsonl, load_config
from icbc_rag.prompts import build_messages

PLAN = 'outputs/final_test_plans/20261021_k3_once.json'
RUN = 'outputs/final_test_runs/20261021_k3_once'
IDS = ['EV001','EV002','EV007','EV008','EV025','EV026','EV031','EV032',
       'EV035','EV036','EV043','EV044','EV045','EV046','EV047','EV048']
ACCEPTED = 'generated_citations_checked_semantics_unverified'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save(path, value):
    """Flush before atomic replace; resolve paths before any paid request."""
    path = Path(path).resolve()
    temp = path.with_suffix(path.suffix + '.part')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def preflight(plan_path):
    plan_path = project_path(str(plan_path))
    plan = load(plan_path)
    freeze = project_path(plan['freeze_directory'])
    pointer = load(ROOT / 'config/evaluation_freeze.json')
    digest = file_hash(freeze / 'freeze_manifest.json')
    if digest != plan['freeze_manifest_sha256'] or digest != pointer['freeze_manifest_sha256']:
        raise ValueError('冻结版本不一致')
    verify_freeze(freeze)
    config = load_config(freeze / 'rag_config.json')
    baseline = load(freeze / 'baseline_config.json')
    if plan['query_ids'] != IDS or plan['query_count'] != 16:
        raise ValueError('测试题号或数量变化')
    expected = {'max_new_model_calls':16, 'max_calls_per_query':1,
                'max_output_tokens_per_call':1200, 'retries':0, 'corpus_count':24,
                'rag_retriever':'dense', 'rag_top_k':3,
                'stop_on_first_failed_request_or_invalid_response':True,
                'EV003_resend_allowed':False, 'reuse_development_questions':False,
                'reference_answers_must_never_enter_model_prompt':True}
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError('调用范围或限制发生变化')
    if config['top_k'] != 3 or config['api']['max_output_tokens'] != 1200:
        raise ValueError('RAG参数与授权计划不一致')
    for key in ['top_k','tokenizer','k1','b']:
        if baseline[key] != plan['baseline'][key]:
            raise ValueError('BM25参数不一致')
    if file_hash(project_path(plan['base_test_file'])) != plan['base_test_sha256']:
        raise ValueError('测试文件变化')
    errata = load(freeze / 'reference_errata.json')
    if (plan['base_test_file'] != errata['base_test_file'] or
        plan['base_test_sha256'] != errata['base_test_sha256'] or
        project_path(plan['reference_errata_file']) != freeze / 'reference_errata.json'):
        raise ValueError('测试参考更正版本不一致')
    if plan['execution_started'] or project_path(RUN).exists():
        raise ValueError('本次正式测试已经开始或有遗留目录，禁止自动重跑')
    if plan['external_model_calls'] != 0 or plan['final_test_queries_run'] != 0:
        raise ValueError('调用历史非空，禁止重跑')
    if not project_path(config['index_file']).is_file():
        raise ValueError('冻结索引缺失，禁止自动重建')
    return plan_path, plan, freeze, config, baseline


def require_approval(plan):
    approval = plan.get('approval') or {}
    if (approval.get('actor') != 'human_user' or approval.get('approved') is not True or
        approval.get('max_new_model_calls') != 16 or not approval.get('source_message') or
        not approval.get('approved_at') or
        approval.get('runner_sha256') != file_hash(Path(__file__)) or
        approval.get('freeze_manifest_sha256') != plan['freeze_manifest_sha256']):
        raise ValueError('缺少针对冻结版本16题正式测试的明确授权')


class JournaledClient:
    """Store the provider response before engine validation or result packaging."""
    def __init__(self, client, folder, state, persist, documents, limit=16):
        self.client, self.folder, self.state, self.persist = client, folder, state, persist
        self.documents = {d['chunk_id']:d for d in documents}
        self.limit = limit
        self.query = None

    @property
    def attempts(self):
        return self.client.attempts

    def complete(self, messages):
        qid = self.query['query_id']
        if qid in self.state['reserved_ids'] or len(self.state['reserved_ids']) >= self.limit:
            raise ValueError('题目重复或调用次数达到上限')
        payload = json.loads(messages[1]['content'])
        evidence = payload['evidence']
        ids = [e['chunk_id'] for e in evidence]
        if len(ids) > 3 or len(ids) != len(set(ids)):
            raise ValueError('检索片段数量或唯一性不符')
        docs = [self.documents[cid] for cid in ids]
        if messages != build_messages(self.query['query'], docs):
            raise ValueError('模型输入必须仅含冻结提示、当前问题和原文片段')
        save(self.folder / (qid + '_request.json'), {'query_id':qid,'messages':messages})
        self.state['reserved_ids'].append(qid)
        self.state['pending_query_id'] = qid
        self.persist()  # Fail closed before sending if durable storage fails.
        try:
            output = self.client.complete(messages)
            save(self.folder / (qid + '_provider.json'), output)
            return output
        finally:
            self.state['external_model_calls'] = self.attempts
            self.persist()


def run_batch(rows, folder, engine, baseline, index, encoder, client, state, persist):
    """Injectable core; tests use synthetic queries and fake providers only."""
    for source in rows:
        row = {'query_id':source['query_id'], 'query':source['query']}
        client.query = row
        save(folder / (row['query_id'] + '_bm25.json'), baseline.search(row['query'], top_k=3))
        result = engine(row, index, encoder, 'api', client)
        save(folder / (row['query_id'] + '_answer.json'), result)
        state['completed_ids'].append(row['query_id'])
        state['pending_query_id'] = None
        persist()
        print(row['query_id'] + ': ' + result['generation_status'], flush=True)
        if result['generation_status'] != ACCEPTED or result['external_model_calls'] != 1:
            state['state'] = 'stopped_after_first_failure_no_retry'
            persist()
            return False
    state['state'] = 'completed_semantic_scoring_pending'
    persist()
    return True


def execute_plan(plan_path):
    plan_path, plan, freeze, config, baseline_config = preflight(plan_path)
    require_approval(plan)  # Must precede reading held-out questions or credentials.
    rows = read_jsonl(project_path(plan['base_test_file']))
    if [r['query_id'] for r in rows] != IDS:
        raise ValueError('测试输入题号顺序不一致')
    from icbc_baseline.bm25 import BM25Index
    from icbc_rag.cli import execute
    from icbc_rag.embeddings import LocalBGE
    from icbc_rag.retrieval import prepare_index
    from icbc_rag.generation import ChatClient
    folder = project_path(RUN)
    folder.mkdir(parents=True, exist_ok=False)  # One-shot lock; never resume automatically.
    state = {'state':'started','started_at':now(),'reserved_ids':[], 'completed_ids':[],
             'pending_query_id':None,'external_model_calls':0, 'semantic_scoring_performed':False,
             'runner_sha256':file_hash(Path(__file__)),
             'freeze_manifest_sha256':plan['freeze_manifest_sha256']}
    plan['execution_started'] = True

    def persist():
        state['updated_at'] = now()
        save(folder / 'run_metadata.json', state)
        plan.update(state=state['state'], external_model_calls=state['external_model_calls'],
                    final_test_queries_run=len(state['completed_ids']))
        save(plan_path, plan)

    persist()
    try:
        encoder = LocalBGE(config)
        index, built = prepare_index(config, encoder, rebuild=False)
        if built or len(index.documents) != 24:
            raise ValueError('禁止变更冻结索引或语料')
        baseline = BM25Index(index.documents, k1=baseline_config['k1'], b=baseline_config['b'],
                             tokenizer=baseline_config['tokenizer'])
        provider = ChatClient(config['api'])
        client = JournaledClient(provider, folder, state, persist, index.documents)
        done = run_batch(rows, folder, execute, baseline, index, encoder, client, state, persist)
        # Scoring-only copy is created after generation, never fed to retrieval/model.
        errata = load(freeze / 'reference_errata.json')
        save(folder / 'scoring_references_only.json', [apply_reference_errata(r, errata) for r in rows])
        verify_freeze(freeze)
        state['freeze_unchanged_after'] = True
        state['test_sha256_after'] = file_hash(project_path(plan['base_test_file']))
        persist()
        return 0 if done else 1
    except BaseException as exc:
        state['state'] = 'interrupted_no_automatic_retry'
        state['error_type'] = type(exc).__name__
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=PLAN)
    parser.add_argument('--execute-approved-plan', action='store_true')
    args = parser.parse_args()
    if args.execute_approved_plan:
        return execute_plan(args.plan)
    _, plan, _, _, _ = preflight(args.plan)
    receipt = {'checked_at':now(), 'state':'offline_preflight_passed_not_executed',
               'runner_sha256':file_hash(Path(__file__)),
               'freeze_manifest_sha256':plan['freeze_manifest_sha256'],
               'query_count':16,'external_model_calls':0,'test_queries_read_for_inference':0,
               'approval_required':True}
    save(ROOT / 'outputs/checks/final_test_preflight_20261021.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('未执行完毕：' + type(exc).__name__, file=sys.stderr)
        sys.exit(1)
