"""Command line entry, reproducible builds and source-traceable run records."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import platform
import random
import sys
import time

from .bm25 import BM25Index, canonical_hash
from .cleaning import build_corpus, write_corpus
from .tokenizer import MODES

PROJECT = Path(__file__).resolve().parents[2]


def output_path(relative: str) -> Path:
    path = (PROJECT / relative).resolve()
    if not path.is_relative_to(PROJECT):
        raise ValueError("Output directory must remain inside the project")
    return path


def read_queries(path: Path) -> list[dict]:
    queries, seen = [], set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError(f"Invalid query at line {line_number}")
        query_id = row.get("query_id", f"Q{line_number:03d}")
        if not isinstance(query_id, str) or not query_id or query_id in seen:
            raise ValueError(f"Missing or duplicate query_id at line {line_number}")
        if len(row["query"]) > 2000:
            raise ValueError(f"Query too long at line {line_number}")
        seen.add(query_id)
        queries.append({"query_id": query_id, "query": row["query"]})
    if not queries:
        raise ValueError("Query file is empty")
    return queries


def print_result(result: dict):
    print(f"\n问题：{result['query']}")
    if not result["results"]:
        print("没有字面命中；这不等于已经完成业务拒答判断。")
        return
    for hit in result["results"]:
        print(f"[{hit['rank']}] {hit['chunk_id']}  BM25={hit['score']:.4f}")
        print(f"出处：{hit['citation_locator']}")
        print(hit["text"][:210] + ("……" if len(hit["text"]) > 210 else ""))
        print(f"链接：{hit['source_url']}")
        if hit["match_note"].startswith("仅单字"):
            print(hit["match_note"])


def parser():
    p = argparse.ArgumentParser(description="工商银行普通借记卡：原文清洗与离线BM25检索")
    p.add_argument("--config", type=Path, default=PROJECT / "config/baseline.json")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--query", help="检索一个问题；返回原文候选和出处")
    modes.add_argument("--batch", type=Path, help="批量JSONL：每行含query，可含query_id")
    modes.add_argument("--interactive", action="store_true", help="连续输入问题，输入exit退出")
    modes.add_argument("--build-only", action="store_true", help="只清洗和构建索引")
    p.add_argument("--top-k", type=int, help="返回条数，1至100")
    p.add_argument("--tokenizer", choices=MODES, help="覆盖配置的字粒度分词方式")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if args.top_k is not None:
            config["top_k"] = args.top_k
        if args.tokenizer is not None:
            config["tokenizer"] = args.tokenizer
        if isinstance(config["top_k"], bool) or not isinstance(config["top_k"], int) or not 1 <= config["top_k"] <= 100:
            raise ValueError("top_k must be an integer in [1, 100]")
        random.seed(config["random_seed"])
        documents, report, quarantine = build_corpus((PROJECT / config["source_dir"]).resolve(), config)
        index = BM25Index(documents, config["k1"], config["b"], config["tokenizer"])
        output = output_path(config["output_dir"])
        write_corpus(output, documents, report, quarantine)
        index.save(output / "bm25_index.json")
        print(f"清洗完成：{len(documents)}个原文片段，来源{','.join(report['active_source_ids'])}；隔离{report['isolated_summary_records']}条资料摘要。")
        print(f"BM25索引：k1={index.k1}, b={index.b}, tokenizer={index.tokenizer}；无API调用。")
        print("结果为检索候选；BM25分数不是正确率或置信度。")

        if args.build_only:
            print("已保存 data/processed/ 下的语料、隔离记录、清洗报告和索引。")
            return 0
        run_kind = "interactive_retrieval" if args.interactive else "single_query" if args.query is not None else "unlabelled_batch" if args.batch else "engineering_smoke"
        if args.query is not None:
            queries = [{"query_id": "Q001", "query": args.query}]
        elif args.batch:
            queries = read_queries(args.batch)
        elif args.interactive:
            queries = []
        else:
            queries = read_queries(PROJECT / "examples/smoke_queries.jsonl")

        now = datetime.now(timezone(timedelta(hours=8)))
        run_id = now.strftime("%Y%m%dT%H%M%S_%f")
        run_dir = output_path(config["run_dir"]) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        count = 0
        with (run_dir / "retrieval.jsonl").open("w", encoding="utf-8") as log:
            def execute(row):
                nonlocal count
                start = time.perf_counter()
                result = index.search(row["query"], config["top_k"])
                result["query_id"] = row["query_id"]
                result["retrieval_ms"] = (time.perf_counter() - start) * 1000
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
                log.flush()
                count += 1
                print_result(result)
            if args.interactive:
                while True:
                    try:
                        query = input("\n请输入问题（exit退出）：").strip()
                    except EOFError:
                        break
                    if query.lower() in ("exit", "quit", "退出"):
                        break
                    if query:
                        try:
                            execute({"query_id": f"Q{count + 1:03d}", "query": query})
                        except ValueError as exc:
                            print(f"输入未处理：{exc}")
            else:
                for row in queries:
                    execute(row)
        metadata = {"run_id": run_id, "started_at": now.isoformat(), "timezone": "Asia/Shanghai",
                    "run_kind": run_kind, "query_count": count, "top_k": config["top_k"],
                    "k1": index.k1, "b": index.b, "tokenizer": index.tokenizer,
                    "random_seed": config["random_seed"], "python_version": platform.python_version(),
                    "config_sha256": canonical_hash(config), "corpus_sha256": index.corpus_sha256,
                    "source_manifest_sha256": report["source_manifest_sha256"],
                    "chunk_count": len(documents), "external_model_calls": 0,
                    "evaluation_status": "仅功能运行，未计算检索准确率、Recall或F1"}
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存本次逐题输出与运行配置：{run_dir.relative_to(PROJECT)}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        print("请检查README中的数据目录、文件哈希及配置说明；程序不会在缺少来源时伪造语料。", file=sys.stderr)
        return 1

