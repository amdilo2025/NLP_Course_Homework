"""Build a source-only corpus from the verified local public snapshots."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .bm25 import canonical_hash

ARTICLE = re.compile(r"第[一二三四五六七八九十]+条")
CHAPTER = re.compile(r"^第[一二三四五六七八九十]+章")
LOCK_SENTENCE = "若持卡人连续输错密码超过3次，发卡银行将对借记卡实施锁定（电子现金除外）。"
FOOTNOTE_ONE = "具体以实际业务为准。"


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_file(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"Source path escapes its directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Source file missing: {path}")
    return path


def source_text(base: Path, source: dict) -> str:
    if not source.get("file_verified") or not source.get("raw_file") or not source.get("text_file"):
        raise ValueError(f"Source has no verified local original: {source['source_id']}")
    raw = safe_file(base, source["raw_file"])
    text = safe_file(base, source["text_file"])
    if digest_file(raw) != source["sha256"]:
        raise ValueError(f"Original snapshot hash mismatch: {source['source_id']}")
    # The collector hashes the decoded UTF-8 string before write_text. Windows
    # can store CRLF on disk, so verify its logical text after universal-newline
    # decoding; keep the original PDF/HTML check strictly byte-for-byte.
    content = text.read_text(encoding="utf-8")
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != source["text_sha256"]:
        raise ValueError(f"Extracted text hash mismatch: {source['source_id']}")
    return content


def normalize_paragraph(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u00a0", " ")
    return re.sub(r"[ \t\u3000]+", " ", text).strip()


def parse_articles(text: str) -> dict[int, dict]:
    lines = []
    for line in text.splitlines():
        line = normalize_paragraph(line)
        if not line or re.fullmatch(r"--- PDF第\d+页 ---", line) or CHAPTER.match(line):
            continue
        # Horizontal space is essential here: \s would cross a newline and
        # accidentally delete the sentence following a stand-alone footnote 1.
        if re.match(r"^[12][ \t]+.+", line):
            continue
        lines.append(line)
    body = "\n".join(lines)
    markers = list(ARTICLE.finditer(body))
    if len(markers) != 36:
        raise ValueError(f"Expected 36 charter articles, found {len(markers)}")
    return {i + 1: {"label": marker.group(),
                    "text": body[marker.start():markers[i + 1].start() if i + 1 < len(markers) else len(body)].strip()}
            for i, marker in enumerate(markers)}


def comparable(text: str) -> str:
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    text = text.replace("正常支持1", "正常支持")
    return text.replace("。", ".").replace("“", '"').replace("”", '"')


def article_pages(pdf_text: str) -> dict[int, list[int]]:
    headings = list(ARTICLE.finditer(pdf_text))
    pages = list(re.finditer(r"--- PDF第(\d+)页 ---", pdf_text))
    if len(headings) != 36:
        raise ValueError("Cannot locate the PDF article headings")
    def page_at(position):
        return max(int(m.group(1)) for m in pages if m.start() <= position)
    result = {}
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(pdf_text)
        segment = pdf_text[heading.start():end]
        # A page marker immediately before the next article is not evidence
        # that the previous article continues onto that new page.
        masked = re.sub(r"--- PDF第\d+页 ---", lambda m: " " * len(m.group()), segment)
        masked = re.sub(r"(?m)^第[一二三四五六七八九十]+章[^\n]*", lambda m: " " * len(m.group()), masked)
        meaningful_end = heading.start() + len(masked.rstrip()) - 1
        result[i + 1] = [page_at(heading.start()), page_at(meaningful_end)]
    return result


def build_corpus(source_dir: Path, config: dict) -> tuple[list[dict], dict, list[dict]]:
    source_dir = source_dir.resolve()
    manifest_path = safe_file(source_dir, config["source_manifest"])
    knowledge_path = safe_file(source_dir, config["knowledge_file"])
    sources = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = {s["source_id"]: s for s in sources}
    if len(registry) != len(sources):
        raise ValueError("Duplicate source IDs in manifest")
    records = [json.loads(line) for line in knowledge_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("Duplicate knowledge record IDs")
    approved, quarantine = [], []
    allowed_sources = {config["canonical_source_id"], *config["additional_source_ids"]}
    for record in records:
        if record.get("default_include") is True:
            if record["source_id"] not in allowed_sources or record["status"] != "可纳入（限所载范围）":
                raise ValueError(f"Unexpected approved record: {record['record_id']}")
            approved.append(record)
        else:
            quarantine.append({"kind": "knowledge_record", "record_id": record["record_id"],
                               "source_id": record["source_id"], "topic": record["topic"],
                               "reason": record["status"], "locator": record["locator"]})
    canonical_id, mirror_id = config["canonical_source_id"], config["formatting_mirror_id"]
    canonical, mirror = registry[canonical_id], registry[mirror_id]
    if canonical.get("document_group") != mirror.get("document_group"):
        raise ValueError("Formatting mirror is not in the canonical document group")
    pdf_text, mirror_text = source_text(source_dir, canonical), source_text(source_dir, mirror)
    primary_articles, mirror_articles = parse_articles(pdf_text), parse_articles(mirror_text)
    page_map = article_pages(pdf_text)
    approved_by_article = {}
    for record in approved:
        if record["source_id"] == canonical_id:
            label_match = ARTICLE.search(record["locator"])
            if not label_match:
                raise ValueError(f"Missing article in approved locator: {record['record_id']}")
            approved_by_article.setdefault(label_match.group(), []).append(record)
    documents = []
    seen_text = set()
    cross_checked = []

    def append_doc(chunk_id, source, text, title, linked_records, locator, **extra):
        cleaned = normalize_paragraph(text)
        if not cleaned:
            raise ValueError(f"Empty chunk: {chunk_id}")
        key = comparable(cleaned)
        if key in seen_text:
            quarantine.append({"kind": "duplicate_chunk", "chunk_id": chunk_id, "reason": "重复原文段落"})
            return
        seen_text.add(key)
        documents.append({"chunk_id": chunk_id, "source_id": source["source_id"],
                          "title": title, "text": cleaned,
                          "source_title": source["title"], "source_url": source["url"],
                          "source_version": source["version"],
                          "effective_date": source.get("effective_date", ""),
                          "scope": source["scope"], "citation_locator": locator,
                          "approved_record_ids": [r["record_id"] for r in linked_records],
                          "conditions": sorted({r["conditions"] for r in linked_records}),
                          "raw_file": source["raw_file"], "raw_sha256": source["sha256"],
                          "text_kind": "清洗后的官方原文，不是AI摘要", **extra})

    for number_string, paragraph_indices in config["article_paragraphs"].items():
        number = int(number_string)
        article = mirror_articles[number]
        linked = approved_by_article.get(article["label"], [])
        if not linked:
            raise ValueError(f"Article not approved by the audited records: {number}")
        if comparable(primary_articles[number]["text"]) != comparable(article["text"]):
            raise ValueError(f"PDF/HTML article disagreement: {number}")
        cross_checked.append(number)
        paragraphs = [p.strip() for p in article["text"].splitlines() if p.strip()]
        for i in paragraph_indices:
            if not isinstance(i, int) or i < 0 or i >= len(paragraphs):
                raise ValueError(f"Invalid paragraph selector: article {number}, {i}")
            paragraph = re.sub(r"^第[一二三四五六七八九十]+条\s*", "", paragraphs[i])
            transformations = ["去除条号与排版空白；条号另存出处"]
            if number == 22 and i == 0:
                if not paragraph.startswith(LOCK_SENTENCE):
                    raise ValueError("Expected password-threshold sentence changed; manual recheck required")
                paragraph = paragraph[len(LOCK_SENTENCE):]
                transformations.append("移除已登记冲突的锁定次数句；保留锁定影响范围")
                quarantine.append({"kind": "conflict_sentence", "source_id": canonical_id,
                                   "article": article["label"], "record_id": "K011",
                                   "reason": "锁定次数表述与旧FAQ存在冲突；不进入默认索引"})
            if number == 6 and i == 1:
                if "正常支持1。" not in paragraph or FOOTNOTE_ONE not in mirror_text:
                    raise ValueError("Expected expiry footnote changed")
                paragraph = paragraph.replace("正常支持1。", "正常支持。")
                paragraph += "（原文脚注1：具体以实际业务为准。）"
                transformations.append("将脚注1关联回所属段落，保留其限制条件")
            p_start, p_end = page_map[number]
            pages_label = f"PDF第{p_start}页" if p_start == p_end else f"PDF第{p_start}—{p_end}页"
            append_doc(f"S001-A{number:02d}-P{i + 1:02d}", canonical, paragraph,
                       f"{article['label']} / 原文第{i + 1}段", linked,
                       f"{article['label']}，原文第{i + 1}段；{pages_label}",
                       article_number=number, paragraph_number=i + 1,
                       pdf_page_start=p_start, pdf_page_end=p_end,
                       formatting_mirror_id=mirror_id, transformations=transformations)
        for i in range(len(paragraphs)):
            if i not in paragraph_indices:
                quarantine.append({"kind": "unselected_paragraph", "source_id": canonical_id,
                                   "article": article["label"], "paragraph_number": i + 1,
                                   "reason": "不在本批已核定的业务摘要覆盖范围"})

    for source_id in config["additional_source_ids"]:
        source = registry[source_id]
        text = source_text(source_dir, source)
        linked = [r for r in approved if r["source_id"] == source_id]
        if not linked:
            raise ValueError(f"No approved record for source: {source_id}")
        if source_id == "S002":
            start, last = text.index("尊敬的客户："), "2026年4月27日"
            end = text.index(last, start) + len(last)
            text = " ".join(text[start:end].split())
            locator = "修订公告正文、附件标题与落款；去除导航"
        elif source_id == "S018":
            text = " ".join(text[text.index("人工服务"):].split())
            locator = "人工服务至在线留言的公开说明；去除导航和下载工具目录"
        else:
            raise ValueError(f"No source-specific cleaner for {source_id}")
        append_doc(f"{source_id}-BODY-01", source, text, source["title"], linked, locator,
                   transformations=["定位公开正文区，去除页面导航、营销及无关目录"])

    active_sources = sorted({d["source_id"] for d in documents})
    for source in sources:
        if source["source_id"] not in active_sources:
            quarantine.append({"kind": "source_not_indexed", "source_id": source["source_id"],
                               "reason": source["decision"],
                               "note": "仅供排版核对，不作为第二份文档索引" if source["source_id"] == mirror_id else source["note"]})
    if any("超过3次" in d["text"] for d in documents):
        raise ValueError("Conflicting threshold leaked into the corpus")
    if not documents:
        raise ValueError("No approved paragraphs survived cleaning")
    report = {"schema_version": 1, "policy_version": config["cleaning_policy_version"],
              "source_manifest_sha256": digest_file(manifest_path),
              "knowledge_file_sha256": digest_file(knowledge_path),
              "config_sha256": canonical_hash(config), "corpus_sha256": canonical_hash(documents),
              "approved_summary_records": len(approved), "isolated_summary_records": len(records) - len(approved),
              "active_source_ids": active_sources, "formatting_mirror_not_counted": mirror_id,
              "cross_checked_charter_articles": cross_checked,
              "chunk_count": len(documents), "character_count": sum(len(d["text"]) for d in documents),
              "quarantine_count": len(quarantine),
              "notes": ["原件不修改；PDF与HTML选定条款交叉核对后按自然段提取。",
                        "28条已核摘要是筛选与元数据，不是28份原始文档或测试答案。",
                        "金额、时限、例外与脚注保留；锁定次数句单独隔离。",
                        "此报告为清洗与工程核查记录，不是检索效果评估。"]}
    return documents, report, quarantine


def write_corpus(output_dir: Path, documents: list[dict], report: dict, quarantine: list[dict]):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "corpus.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in documents), encoding="utf-8")
    (output_dir / "quarantine.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in quarantine), encoding="utf-8")
    (output_dir / "cleaning_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
