"""Engineering correctness checks, not the course project's evaluation set."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TEST_TMP = (ROOT / "outputs" / ".test_tmp").resolve()
if not TEST_TMP.is_relative_to(ROOT):
    raise ValueError("Test temporary directory must remain within the project")
TEST_TMP.mkdir(parents=True, exist_ok=True)

from icbc_baseline.bm25 import BM25Index
from icbc_baseline.cleaning import (article_pages, build_corpus, comparable,
                                    parse_articles, safe_file, source_text)
from icbc_baseline.tokenizer import tokenize


class ScoringTests(unittest.TestCase):
    def test_score_matches_independent_hand_calculation(self):
        docs = [{"chunk_id": "a", "text": "apple apple banana"},
                {"chunk_id": "b", "text": "banana carrot"}]
        result = BM25Index(docs).search("apple", 1)["results"][0]
        # N=2, df=1: numerator=2-1+0.5=1.5, denominator=1+0.5=1.5.
        idf = math.log(2)
        expected = idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2.5))
        self.assertEqual(result["chunk_id"], "a")
        self.assertAlmostEqual(result["score"], expected, places=12)

    def test_ties_are_stable_and_unknown_tokens_do_not_return_zero_scores(self):
        index = BM25Index([{"chunk_id": "b", "text": "apple"}, {"chunk_id": "a", "text": "apple"}])
        self.assertEqual([r["chunk_id"] for r in index.search("apple")["results"]], ["a", "b"])
        self.assertEqual(index.search("unknownzz")["results"], [])
        self.assertEqual(index.search("的了吗")["results"], [])

    def test_invalid_parameters_and_query_boundaries(self):
        for k1, b in [(0, 0.75), (float("nan"), 0.75), (1.2, 1.1)]:
            with self.assertRaises(ValueError):
                BM25Index([{"chunk_id": "a", "text": "apple"}], k1, b)
        index = BM25Index([{"chunk_id": "a", "text": "apple"}])
        for top_k in [0, 101, True]:
            with self.assertRaises(ValueError):
                index.search("apple", top_k)
        with self.assertRaises(ValueError):
            index.search(" ")

    def test_saved_index_detects_document_and_posting_tampering(self):
        index = BM25Index([{"chunk_id": "a", "text": "密码重置"}, {"chunk_id": "b", "text": "正式挂失"}])
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temp:
            p = Path(temp) / "index.json"
            index.save(p)
            self.assertEqual(index.search("密码重置"), BM25Index.load(p).search("密码重置"))
            saved = json.loads(p.read_text(encoding="utf-8"))
            altered = copy.deepcopy(saved)
            altered["documents"][0]["text"] = "modified"
            p.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                BM25Index.load(p)
            saved["doc_lengths"][0] += 1
            p.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "statistics mismatch"):
                BM25Index.load(p)


class CleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config/baseline.json").read_text(encoding="utf-8"))
        cls.source_dir = (ROOT / cls.config["source_dir"]).resolve()
        cls.docs, cls.report, cls.quarantine = build_corpus(cls.source_dir, cls.config)

    def test_isolation_and_canonical_source_deduplication(self):
        self.assertEqual(self.report["approved_summary_records"], 28)
        self.assertEqual(self.report["isolated_summary_records"], 12)
        self.assertEqual({d["source_id"] for d in self.docs}, {"S001", "S002", "S018"})
        self.assertEqual(len(self.docs), len({d["chunk_id"] for d in self.docs}))
        self.assertFalse(any("超过3次" in d["text"] for d in self.docs))
        self.assertTrue(any(d["article_number"] == 22 for d in self.docs if "article_number" in d))

    def test_expiry_footnote_and_conditional_clauses_survive(self):
        expiry = next(d for d in self.docs if d["chunk_id"] == "S001-A06-P02")
        self.assertIn("具体以实际业务为准", expiry["text"])
        self.assertIn("持卡人可在卡片服务有效期届满后或提前选择换卡", expiry["text"])
        loss = next(d for d in self.docs if d["chunk_id"] == "S001-A21-P02")
        self.assertIn("自挂失当日算起", loss["text"])
        self.assertIn("部分特殊卡种除外", loss["text"])

    def test_page_marker_does_not_extend_previous_article(self):
        chunk = next(d for d in self.docs if d["chunk_id"] == "S001-A25-P01")
        self.assertEqual([chunk["pdf_page_start"], chunk["pdf_page_end"]], [9, 9])

    def test_build_is_deterministic_and_every_chunk_has_provenance(self):
        docs, report, _ = build_corpus(self.source_dir, self.config)
        self.assertEqual(self.docs, docs)
        self.assertEqual(self.report["corpus_sha256"], report["corpus_sha256"])
        self.assertEqual(len(report["cross_checked_charter_articles"]), 14)
        for chunk in docs:
            self.assertTrue(chunk["approved_record_ids"])
            self.assertTrue(chunk["source_url"].startswith("https://"))
            self.assertTrue(chunk["citation_locator"])
            self.assertEqual(len(chunk["raw_sha256"]), 64)

    def test_source_hash_accepts_windows_newlines_but_rejects_changed_text(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temp:
            root = Path(temp)
            raw = b"original"
            (root / "raw.bin").write_bytes(raw)
            (root / "text.txt").write_bytes("甲\r\n乙\r\n".encode("utf-8"))
            source = {"source_id": "fixture", "file_verified": True, "raw_file": "raw.bin", "text_file": "text.txt",
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "text_sha256": hashlib.sha256("甲\n乙\n".encode("utf-8")).hexdigest()}
            self.assertEqual(source_text(root, source), "甲\n乙\n")
            (root / "text.txt").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "text hash mismatch"):
                source_text(root, source)
            with self.assertRaises(ValueError):
                safe_file(root, "../outside.txt")


class TokenizerTests(unittest.TestCase):
    def test_numbers_negation_and_punctuation_boundaries(self):
        tokens = tokenize("不支持。密码，挂失15天 ATM")
        self.assertIn("不", tokens)
        self.assertIn("不支", tokens)
        self.assertIn("密码", tokens)
        self.assertIn("15", tokens)
        self.assertIn("atm", tokens)
        self.assertNotIn("码挂", tokens)
        self.assertNotIn("15天", tokens)
        self.assertNotIn("密码", tokenize("密码", "char"))


if __name__ == "__main__":
    unittest.main()
