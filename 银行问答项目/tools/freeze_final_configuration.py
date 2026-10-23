"""Record the user's final choices and immutable evaluation hashes. No inference or API calls."""
import argparse
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icbc_rag.common import file_hash, read_jsonl
from icbc_baseline.bm25 import canonical_hash


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def verify_snapshot(path):
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        if file_hash(path/name) != digest:
            raise ValueError('快照变化：' + name)
    return manifest


def text_hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def apply_reference_errata(record, errata):
    """Apply the pre-test reference correction for scoring only; never alter the input row."""
    updated = copy.deepcopy(record)
    item = errata['corrections'].get(record['query_id'])
    if item is None:
        return updated
    if record.get('split') != 'test' or record['answer_type'] != item['answer_type_unchanged']:
        raise ValueError('参考更正不得改变分集或处理类型')
    if text_hash(record['reference_answer']) != item['old_reference_answer_sha256']:
        raise ValueError('原参考答案不匹配，停止套用更正')
    updated['reference_answer_original'] = record['reference_answer']
    updated['reference_answer'] = item['new_reference_answer']
    updated['reference_erratum_applied'] = 'EV036_pre_test_correction_20261021'
    return updated


def verify_environment(config, environment):
    for env_key, default_key in [('base_url_env', 'default_base_url'), ('model_env', 'default_model')]:
        value = environment.get(config['api'][env_key])
        if value and value.strip().rstrip('/') != config['api'][default_key].strip().rstrip('/'):
            raise ValueError('环境变量覆盖了已冻结API地址或模型')


def verify_freeze(folder, environment=None):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'freeze_manifest.json').read_text(encoding='utf-8'))
    if manifest['state'] != 'frozen':
        raise ValueError('配置尚未冻结')
    for name, digest in manifest['files'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or file_hash(path) != digest:
            raise ValueError('冻结副本变化：' + name)
    for name, digest in manifest['protected_files'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or file_hash(path) != digest:
            raise ValueError('受保护运行文件或证据变化：' + name)
    config = json.loads((folder/'rag_config.json').read_text(encoding='utf-8'))
    verify_environment(config, os.environ if environment is None else environment)
    return manifest


def build(target):
    target = Path(target)
    if not target.is_absolute():
        target = ROOT/target
    target = target.resolve()
    stage = target.with_name(target.name+'.staging')
    if not target.is_relative_to(ROOT.resolve()) or not stage.is_relative_to(ROOT.resolve()):
        raise ValueError('冻结目录必须位于项目内')
    if target.exists() or stage.exists():
        raise ValueError('不覆盖已有冻结版本或未完成暂存目录')
    pending_path = ROOT/'data/evaluation/finalization/20261021_decisions_pending.json'
    pending = json.loads(pending_path.read_text(encoding='utf-8'))
    if not pending['protocol_adoption']['confirmed'] or not pending['protocol_adoption']['freeze_exception_explicitly_approved']:
        raise ValueError('未确认评分口径及缺失例外')
    reviewed = ROOT/'data/evaluation/reviewed/20261021_settings_v2'
    v1 = ROOT/'data/evaluation/reviewed/20261017_word_v1'
    human = ROOT/'data/evaluation/human_reviews/20261021_returned_v1'
    reviewed_meta = verify_snapshot(reviewed)
    verify_snapshot(v1)
    human_meta = verify_snapshot(human)
    if file_hash(human/'manifest.json') != pending['human_review_manifest_sha256']:
        raise ValueError('主评确认版本与定稿记录不一致')
    if human_meta['reviewed_queries'] != 29 or human_meta['cross_review_queries'] != 6:
        raise ValueError('缺少主评或抽检记录')
    confirmations = read_jsonl(human/'reviews.jsonl')
    answer_hashes = {}
    for row in confirmations:
        relative = f'outputs/manual_reviews/development_a7f94cb8e0ab/answers/{row["query_id"]}.json'
        if file_hash(ROOT/relative) != row['source_answer_sha256']:
            raise ValueError('已经确认的模型回答发生变化')
        answer_hashes[relative] = row['source_answer_sha256']
    rag = json.loads((ROOT/'config/rag.json').read_text(encoding='utf-8'))
    baseline = json.loads((ROOT/'config/baseline.json').read_text(encoding='utf-8'))
    if rag['top_k'] != 3 or baseline['top_k'] != 3 or baseline['tokenizer'] != 'char_bigram' or baseline['k1'] != 1.2 or baseline['b'] != .75:
        raise ValueError('当前参数与本人确认的最终版本不一致')
    verify_environment(rag, os.environ)
    corpus = read_jsonl(ROOT/rag['corpus_file'])
    if len(corpus) != 24 or canonical_hash(corpus) != reviewed_meta['corpus_sha256'] or rag['corpus_sha256'] != reviewed_meta['corpus_sha256']:
        raise ValueError('最终评估必须使用固定24片段')
    policy = json.loads((reviewed/'evaluation_policy.json').read_text(encoding='utf-8'))
    # Only an administrative reference row is read here, never sent to retrieval or a model.
    reference = next(r for r in read_jsonl(reviewed/'all_submitted.jsonl') if r['query_id'] == 'EV036')
    if reference['split'] != 'test':
        raise ValueError('EV036分集不符')
    proposed = policy['deferred_revisions']['EV036']['proposed_answer']
    frozen_at = datetime.now(timezone(timedelta(hours=8))).isoformat()
    errata = {'created_at': frozen_at, 'adopted_before_final_test': True,
              'base_test_file': 'data/evaluation/reviewed/20261017_word_v1/test.jsonl',
              'base_test_sha256': file_hash(v1/'test.jsonl'),
              'application': '仅在人工评分/评估读取参考答案时套用，不进入模型提示、检索语料或生成输入',
              'corrections': {'EV036': {'old_reference_answer': reference['reference_answer'],
                              'old_reference_answer_sha256': text_hash(reference['reference_answer']),
                              'new_reference_answer': proposed,
                              'answer_type_unchanged': reference['answer_type'],
                              'reference_ids_unchanged': reference['reference_ids'],
                              'basis_chunk_ids': reference['reference_chunk_ids'],
                              'basis_note': '按R19保留证件过期、合理期限内未更新且没有合理理由、有权中止、至提供有效证件等限定',
                              'approval': '用户确认采用沿用k=3与EV036收窄措辞这两个选择'}}}
    apply_reference_errata(reference, errata)  # Check correction compatibility before writing.
    decisions = {'confirmed_at': frozen_at, 'stage': 'final_choices_adopted',
                 'source_message': '采用这两个选择',
                 'context': '采用助手建议的向量k=3及EV036收窄措辞，原文件保留',
                 'protocol_adoption': pending['protocol_adoption'],
                 'final_rag_selection': {'retriever': 'dense', 'top_k': 3,
                                        'basis': '沿用已有29份生成和人工确认记录；预算与已验证版本一致性考虑，不宣称最优参数'},
                 'EV036_reference_decision': '采纳已提供的收窄措辞，独立更正，不覆盖原测试文件',
                 'final_test_authorized': False, 'final_test_queries_run': 0}
    scoring = copy.deepcopy(policy)
    scoring.update({'status': 'frozen_by_explicit_user_adoption', 'frozen_at': frozen_at,
                    'development_response_scoring_queries': 29, 'development_retrieval_queries': 30,
                    'missing_response_policy': pending['protocol_adoption'],
                    'core_point_definition_source': 'outputs/ai_reviews/semantic_initial_20261021/assessments.json，经返回MD逐题确认及本轮整体采纳',
                    'core_and_reference_extensions_separate': True,
                    'test_reference_errata_file': 'reference_errata.json',
                    'cross_review_query_ids': ['EV004','EV010','EV014','EV023','EV037','EV040'],
                    'development_review_mode': 'AI初评后主评确认及6题抽检，不宣称独立盲评',
                    'test_review_mode': '两名成员分别评分全部16题，保留分歧；不得冒用姓名自动填分',
                    'final_test_requires_separate_authorization': True,
                    'final_test_execution_limit': '冻结后一次正式测试；不自动重试，不据最终测试回调参数',
                    'known_topic_overlap': 'EV025、EV026与旧SMOKE05同属吞卡主题，不能宣称所有测试主题未接触'})
    scoring['deferred_revisions']['EV036'].update({'adopted': True, 'reason': '本轮用户明确采纳，作为测试前参考更正'})
    scoring['final_test_allowed'] = False
    protected_paths = [ROOT/'run_baseline.py', ROOT/'run_rag.py', Path(__file__).resolve(),
                       *sorted((ROOT/'src').rglob('*.py')), ROOT/'config/rag.json', ROOT/'config/baseline.json',
                       ROOT/rag['corpus_file'], ROOT/rag['index_file'],
                       ROOT/rag['model_dir']/rag['model_file'], ROOT/rag['model_dir']/rag['tokenizer_file'],
                       ROOT/'requirements-rag-lock.txt', reviewed/'manifest.json', v1/'manifest.json',
                       human/'manifest.json', human/'reviews.jsonl', human/'scores.jsonl', human/'summary.json',
                       reviewed/'evaluation_policy.json', v1/'test.jsonl', reviewed/'test.jsonl',
                       ROOT/'outputs/ai_reviews/semantic_initial_20261021/assessments.json',
                       ROOT/'outputs/retrieval_metrics/20261021T103426_539747/summary.json']
    protected = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in protected_paths}
    protected.update(answer_hashes)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    shutil.copyfile(ROOT/'config/rag.json', stage/'rag_config.json')
    shutil.copyfile(ROOT/'config/baseline.json', stage/'baseline_config.json')
    write_json(stage/'decisions.json', decisions)
    write_json(stage/'reference_errata.json', errata)
    write_json(stage/'scoring_protocol.json', scoring)
    freeze = {'schema_version': 1, 'state': 'frozen', 'frozen_at': frozen_at,
              'scope': '原24片段评估版配置、原文、模型及编码器、引擎代码、评分协议与测试前参考更正',
              'selected_retriever': 'dense', 'top_k': 3,
              'baseline': {'method':'BM25','top_k':3,'tokenizer':'char_bigram','k1':1.2,'b':.75},
              'corpus_sha256': rag['corpus_sha256'], 'corpus_count': 24,
              'model_revision': rag['model_revision'], 'generation_api_model': rag['api']['default_model'],
              'generation_parameters': {'temperature': rag['api']['temperature'], 'max_output_tokens': rag['api']['max_output_tokens'],
                                        'thinking_mode': rag['api']['thinking_mode'], 'json_mode': rag['api']['json_mode']},
              'development_retrieval_queries': 30, 'development_scored_responses': 29,
              'EV003_exception_approved': True, 'EV003_resend_allowed': False,
              'EV036_reference_erratum_adopted': True,
              'base_test_sha256': file_hash(v1/'test.jsonl'), 'test_query_ids': reviewed_meta['question_ids']['test'],
              'test_query_count': 16, 'test_queries_run_before_freeze': 0,
              'final_test_authorized': False, 'external_model_calls_this_freeze': 0,
              'prior_pending_record_sha256': file_hash(pending_path),
              'files': {p.name: file_hash(p) for p in sorted(stage.iterdir()) if p.is_file()},
              'protected_files': protected}
    write_json(stage/'freeze_manifest.json', freeze)
    verify_freeze(stage)
    stage.rename(target)
    pointer = {'state':'frozen', 'frozen_at':frozen_at, 'directory':target.relative_to(ROOT).as_posix(),
               'freeze_manifest_sha256':file_hash(target/'freeze_manifest.json'), 'final_test_authorized':False}
    write_json(ROOT/'config/evaluation_freeze.json', pointer)
    return pointer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--target', type=Path, default=ROOT/'data/evaluation/frozen/20261021_k3_v1')
    args = parser.parse_args(argv)
    if args.verify:
        folder = args.verify if args.verify.is_absolute() else ROOT/args.verify
        m = verify_freeze(folder)
        result = {'state':m['state'], 'integrity_valid':True, 'top_k':m['top_k'], 'test_authorized':m['final_test_authorized']}
    else:
        result = build(args.target)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
