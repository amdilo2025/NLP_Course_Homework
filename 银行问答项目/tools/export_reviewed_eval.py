"""Read the review XLSX with the standard library; export only explicit reviews."""
import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icbc_baseline.bm25 import canonical_hash

DEFAULT_BOOK = ROOT / 'data/evaluation/评估题目待核验_20261015.xlsx'
HEADERS = ['题目ID','题组','类别','问题草稿','AI参考要点（待核）','候选依据ID','核验提示','核验状态',
           '确认问题','确认答案','确认类型','确认依据ID','用途','已用于调试','核验人','核验日期','修改说明']
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
TYPES = {'可回答','资料不足','需澄清','范围外'}


def normalized(text):
    return re.sub(r'[\W_]+', '', text).lower()


def read_review_rows(book):
    with zipfile.ZipFile(book) as z:
        if z.testzip() is not None:
            raise ValueError('Excel文件校验失败')
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            for item in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', NS):
                strings.append(''.join(t.text or '' for t in item.findall('.//s:t', NS)))
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        targets = {r.attrib['Id']: r.attrib['Target'] for r in rels}
        rid = None
        for sheet in wb.findall('s:sheets/s:sheet', NS):
            if sheet.attrib['name'] == '待核验题目':
                rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        if rid is None:
            raise ValueError('缺少“待核验题目”工作表')
        target = targets[rid]
        member = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
        xml = ET.fromstring(z.read(member))
        row_values = {}
        for row in xml.findall('s:sheetData/s:row', NS):
            n = int(row.attrib['r'])
            values = {}
            for cell in row.findall('s:c', NS):
                letters = re.match('[A-Z]+', cell.attrib['r']).group()
                if cell.find('s:f', NS) is not None and n >= 6 and letters in ('H','I','J','K','L','M','N','O','P','Q'):
                    raise ValueError(f'人工填写区不读取公式，请在{cell.attrib["r"]}填普通值')
                if cell.attrib.get('t') == 's':
                    value = strings[int(cell.findtext('s:v', '0', NS))]
                elif cell.attrib.get('t') == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
                else:
                    value = cell.findtext('s:v', '', NS)
                values[letters] = value.strip()
            row_values[n] = values
        columns = [chr(65 + i) for i in range(len(HEADERS))]
        if [row_values.get(5, {}).get(c, '') for c in columns] != HEADERS:
            raise ValueError('表头结构改变；请保留模板第5行表头')
        return [dict(zip(HEADERS, [values.get(c, '') for c in columns]))
                for n, values in sorted(row_values.items()) if n >= 6 and values.get('A')]


def review_date(value):
    if not value:
        raise ValueError('缺少核验日期')
    if re.fullmatch(r'\d+(?:\.\d+)?', value):
        value = (datetime(1899, 12, 30) + timedelta(days=float(value))).date().isoformat()
    return date.fromisoformat(value).isoformat()


def validate_rows(rows, candidates, corpus, smoke_questions):
    originals = {c['question_id']: c for c in candidates}
    known_chunks = {d['chunk_id'] for d in corpus}
    errors, accepted, seen_ids = [], [], set()
    split_by_group, tuned_groups = {}, set()
    for row in rows:
        if row.get('已用于调试') == '是':
            tuned_groups.add(row.get('题组'))
    for row in rows:
        qid = row['题目ID']
        if qid in seen_ids or qid not in originals:
            errors.append(f'{qid}：题号重复或不在本次候选快照中')
            continue
        seen_ids.add(qid)
        original = originals[qid]
        if row['题组'] != original['group_id'] or row['问题草稿'] != original['question_draft']:
            errors.append(f'{qid}：请保留原题组和草稿；改写应填写“确认问题”列')
            continue
        status = row['核验状态']
        if status not in {'待核验','已确认','需修改','排除'}:
            errors.append(f'{qid}：核验状态无效')
            continue
        if status != '已确认' or row['用途'] == '排除':
            continue
        before = len(errors)
        question = row['确认问题']
        if question == '保留原题':
            question = original['question_draft']
        answer, answer_type = row['确认答案'], row['确认类型']
        split, used = row['用途'], row['已用于调试']
        refs = [s for s in re.split(r'[,，;；、\s]+', row['确认依据ID']) if s]
        if not question or not answer or answer in {'待补充','待核验','同上','同AI','AI草案'}:
            errors.append(f'{qid}：缺少独立确认的问题或答案')
        if answer_type not in TYPES:
            errors.append(f'{qid}：缺少有效的人工确认类型')
        if answer_type == '可回答' and not refs:
            errors.append(f'{qid}：可回答题必须确认原文片段ID')
        if len(refs) != len(set(refs)) or any(r not in known_chunks for r in refs):
            errors.append(f'{qid}：依据ID重复或不在固定语料中')
        if split not in {'开发','测试'} or used not in {'是','否'}:
            errors.append(f'{qid}：用途或调试使用情况尚未确认')
        if not row['核验人']:
            errors.append(f'{qid}：缺少核验人')
        try:
            checked_on = review_date(row['核验日期'])
        except (ValueError, OverflowError):
            checked_on = ''
            errors.append(f'{qid}：核验日期应为YYYY-MM-DD或有效Excel日期')
        if split == '测试' and (used == '是' or original['group_id'] in tuned_groups or normalized(question) in smoke_questions):
            errors.append(f'{qid}：本题/同题组已用于调试或与已有功能示例重复，不进入测试集')
        if split in {'开发','测试'}:
            split_by_group.setdefault(original['group_id'], set()).add(split)
        if len(errors) == before:
            accepted.append({'query_id':qid,'group_id':original['group_id'],'query':question,
                             'reference_answer':answer,'answer_type':answer_type,
                             'gold_chunk_ids':refs,'split':'development' if split == '开发' else 'test',
                             'reviewer':row['核验人'],'review_date':checked_on,
                             'used_for_tuning':used == '是','edit_note':row['修改说明'],
                             'draft_origin':'AI候选，经表中明确人工确认'})
    for group, splits in split_by_group.items():
        if len(splits) > 1:
            errors.append(f'{group}：同题组跨开发/测试，需由小组调整划分')
    questions = [normalized(q['query']) for q in accepted]
    if len(questions) != len(set(questions)):
        errors.append('确认问题存在重复，需去重')
    if errors:
        raise ValueError('\n'.join(errors))
    return accepted


def main(argv=None):
    parser = argparse.ArgumentParser(description='检查人工核验表；默认不导出正式数据')
    parser.add_argument('--workbook', type=Path, default=DEFAULT_BOOK)
    parser.add_argument('--export', action='store_true', help='显式导出已确认且字段完整的行')
    args = parser.parse_args(argv)
    try:
        folder = ROOT / 'data/evaluation'
        manifest = json.loads((folder / 'preparation_manifest_20261015.json').read_text(encoding='utf-8'))
        candidates = [json.loads(s) for s in (folder / manifest['candidate_file']).read_text(encoding='utf-8').splitlines() if s.strip()]
        corpus = [json.loads(s) for s in (ROOT / 'data/processed/corpus.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        if canonical_hash(corpus) != manifest['corpus_sha256'] or canonical_hash(candidates) != manifest['candidate_sha256']:
            raise ValueError('固定语料或候选快照已改变，请先核对评估版本')
        smoke = {normalized(json.loads(s)['query']) for s in (ROOT / 'examples/smoke_queries.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()}
        rows = read_review_rows(args.workbook)
        accepted = validate_rows(rows, candidates, corpus, smoke)
        print(f'读取{len(rows)}题；字段完整且明确人工确认{len(accepted)}题；未执行效果评分。')
        if not args.export:
            print('当前为检查模式，没有导出或自动分配开发/测试题。')
            return 0
        if not accepted:
            raise ValueError('没有可导出的已确认题目；不会把AI草稿当作金标准')
        now = datetime.now(timezone(timedelta(hours=8)))
        target = folder / 'reviewed' / now.strftime('%Y%m%dT%H%M%S_%f')
        target.mkdir(parents=True, exist_ok=False)
        for split in ('development','test'):
            selected = [r for r in accepted if r['split'] == split]
            (target / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected), encoding='utf-8')
        info = {'created_at':now.isoformat(),'workbook':args.workbook.name,
                'corpus_sha256':manifest['corpus_sha256'],'records_sha256':canonical_hash(accepted),
                'confirmed_count':len(accepted),'development_count':sum(r['split']=='development' for r in accepted),
                'test_count':sum(r['split']=='test' for r in accepted)}
        (target / 'manifest.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'已导出：{target}')
        return 0
    except (ValueError, OSError, KeyError, zipfile.BadZipFile, ET.ParseError) as exc:
        print(f'核验表检查未通过：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
