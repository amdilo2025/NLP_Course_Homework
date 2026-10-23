"""Bounded development collection. Default prepares a local plan; execution requires recorded user approval."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / '.rag_packages').is_dir():
    sys.path.insert(0, str(ROOT / '.rag_packages'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from tools.manual_rag_review import prepare, save_progress, write_json, now
from icbc_baseline.bm25 import canonical_hash
from icbc_rag.common import file_hash, load_config
from icbc_rag.cli import execute
from icbc_rag.generation import ChatClient


def create_plan(config):
    queries, partition, session, info, saved = prepare(config)
    pending = set(info.get('pending_attempts', []))
    selected = [r for r in queries if r['query_id'] not in saved and r['query_id'] not in pending]
    plan = {'created_at': now(), 'kind': 'bounded_development_collection',
            'session': session.relative_to(ROOT).as_posix(), 'session_fingerprint': info['session_fingerprint'],
            'config_sha256': canonical_hash(config), 'test_sha256': partition['test_sha256_before'],
            'existing_answer_hashes': {qid: file_hash(session / 'answers' / f'{qid}.json') for qid in saved},
            'query_ids': [r['query_id'] for r in selected], 'skipped_pending': sorted(pending),
            'max_new_api_calls': len(selected), 'max_output_tokens_per_call': config['api']['max_output_tokens'],
            'retry_count': 0, 'stop_on_first_failed_result': True,
            'approval': None, 'execution_started': False, 'external_model_calls': 0,
            'attempted_query_ids': [], 'completed_query_ids': [], 'final_test_queries_run': 0}
    return plan


def validate_plan(plan, config, queries, info, session):
    if plan.get('kind') != 'bounded_development_collection' or plan['config_sha256'] != canonical_hash(config):
        raise ValueError('计划配置不一致')
    if plan['session_fingerprint'] != info['session_fingerprint'] or plan['test_sha256'] != info['test_sha256_before']:
        raise ValueError('计划对应的会话或测试版本不一致')
    approved = plan.get('approval') or {}
    if approved.get('actor') != 'human_user' or approved.get('approved') is not True or not approved.get('source_message_summary'):
        raise ValueError('尚未记录用户对本次付费调用的明确授权')
    if not approved.get('approved_at'):
        raise ValueError('缺少授权记录时间')
    allowed = {r['query_id'] for r in queries}
    ids = plan['query_ids']
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= allowed:
        raise ValueError('计划只能包含不重复的固定开发题')
    if type(plan['max_new_api_calls']) is not int or plan['max_new_api_calls'] != len(ids) or plan['max_new_api_calls'] > 30:
        raise ValueError('调用上限异常')
    if plan['retry_count'] != 0 or plan['stop_on_first_failed_result'] is not True:
        raise ValueError('不得重试或失败后继续消耗token')
    if plan['max_output_tokens_per_call'] != config['api']['max_output_tokens']:
        raise ValueError('输出token上限变化')
    if plan['execution_started'] or plan['attempted_query_ids'] or plan['external_model_calls']:
        raise ValueError('此计划已经启动，禁止重启或自动重试')
    for qid, digest in plan['existing_answer_hashes'].items():
        if file_hash(session / 'answers' / f'{qid}.json') != digest:
            raise ValueError('原有回答已变化，不自动覆盖或重复生成')


def run_approved_plan(path, client_factory=ChatClient, engine_factory=None, execute_fn=execute):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    # Resolve all provenance paths before constructing a client or sending anything.
    plan_relative_path = path.relative_to(ROOT.resolve()).as_posix()
    plan = json.loads(path.read_text(encoding='utf-8'))
    config = load_config(ROOT / 'config/rag.json')
    queries, partition, session, info, saved = prepare(config)
    validate_plan(plan, config, queries, info, session)
    if file_hash(partition['test_file']) != plan['test_sha256']:
        raise ValueError('测试文件发生变化')
    rows = {r['query_id']: r for r in queries}
    def default_engine(settings):
        from icbc_rag.embeddings import LocalBGE
        from icbc_rag.retrieval import prepare_index
        encoder = LocalBGE(settings)
        index, _ = prepare_index(settings, encoder)
        return encoder, index
    encoder = index = client = None
    plan['execution_started'] = True
    plan['started_at'] = now()
    write_json(path, plan)
    try:
        for qid in plan['query_ids']:
            if qid in saved or qid in info.get('pending_attempts', []):
                continue
            if plan['external_model_calls'] >= plan['max_new_api_calls']:
                break
            if encoder is None:
                encoder, index = (engine_factory or default_engine)(config)
                client = client_factory(config['api'])
            info.setdefault('pending_attempts', []).append(qid)
            plan['attempted_query_ids'].append(qid)
            # Persist before sending: an interrupted request is never silently repeated.
            save_progress(session, info, saved, queries, partition)
            write_json(path, plan)
            result = execute_fn(rows[qid], index, encoder, 'api', client)
            # Store the complete result before subsequent metadata/formatting work.
            write_json(session / 'request_results' / f'{path.stem}_{qid}.json', result)
            plan['external_model_calls'] += result['external_model_calls']
            item = {'query_id': qid, 'query': rows[qid]['query'], 'collected_at': now(),
                    'origin': 'user_approved_agent_request', 'approval_plan': plan_relative_path,
                    'new_external_model_calls': result['external_model_calls'], 'result': result,
                    'manual_notes': None, 'manual_reviewer': None, 'manual_reviewed_at': None}
            write_json(session / 'answers' / f'{qid}.json', item)
            saved[qid] = item
            info['pending_attempts'].remove(qid)
            plan['completed_query_ids'].append(qid)
            save_progress(session, info, saved, queries, partition)
            write_json(path, plan)
            print(qid + ' ' + result['generation_status'], flush=True)
            if result['generation_status'] != 'generated_citations_checked_semantics_unverified':
                plan['stopped_on_failed_query'] = qid
                break
    except BaseException as exc:
        plan['stopped_on_exception_type'] = type(exc).__name__
        plan['pending_request_ids'] = list(info.get('pending_attempts', []))
        raise
    finally:
        plan['finished_at'] = now()
        plan['test_file_unchanged'] = file_hash(partition['test_file']) == plan['test_sha256']
        plan['human_semantic_scoring_performed'] = False
        write_json(path, plan)
        save_progress(session, info, saved, queries, partition)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-approved-plan', type=Path)
    args = parser.parse_args(argv)
    if args.execute_approved_plan:
        result = run_approved_plan(args.execute_approved_plan)
    else:
        plan = create_plan(load_config(ROOT / 'config/rag.json'))
        folder = ROOT / 'outputs/agent_collection_plans'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'development_20261021.json'
        if path.exists():
            raise ValueError('计划已存在，不覆盖授权或历史执行状态')
        write_json(path, plan)
        result = {'plan': str(path), 'max_new_api_calls': plan['max_new_api_calls'], 'external_model_calls': 0,
                  'approval_recorded': False, 'existing_answers_reused': len(plan['existing_answer_hashes'])}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
