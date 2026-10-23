"""A small inverted-index BM25 implementation with positive IDF."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from .tokenizer import MODES, tokenize


def canonical_hash(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def validate_parameters(k1: float, b: float, mode: str):
    if not math.isfinite(k1) or k1 <= 0:
        raise ValueError("k1 must be finite and greater than zero")
    if not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("b must be finite and in [0, 1]")
    if mode not in MODES:
        raise ValueError(f"Unknown tokenizer: {mode}")


class BM25Index:
    SCHEMA_VERSION = 1

    def __init__(self, documents: list[dict], k1: float = 1.2, b: float = 0.75,
                 tokenizer: str = "char_bigram"):
        validate_parameters(k1, b, tokenizer)
        if not documents:
            raise ValueError("Cannot build an empty index")
        ids = [d["chunk_id"] for d in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs")
        self.documents = documents
        self.k1, self.b, self.tokenizer = k1, b, tokenizer
        self.postings = defaultdict(list)
        self.lengths = []
        for number, document in enumerate(documents):
            # Index the source paragraph only. AI summaries, topic labels, URLs,
            # and conditions are displayed as metadata, not retrieval evidence.
            tokens = tokenize(document["text"], tokenizer)
            self.lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self.postings[term].append((number, count))
        self.avgdl = sum(self.lengths) / len(self.lengths)
        if self.avgdl == 0:
            raise ValueError("All documents are empty after tokenization")
        self.corpus_sha256 = canonical_hash(documents)

    def search(self, query: str, top_k: int = 3) -> dict:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer in [1, 100]")
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query must contain 1 to 2000 characters")
        query_tokens = sorted(set(tokenize(query, self.tokenizer)))
        scores, matches = defaultdict(float), defaultdict(list)
        n = len(self.documents)
        for term in query_tokens:
            postings = self.postings.get(term, ())
            if not postings:
                continue
            df = len(postings)
            idf = math.log1p((n - df + 0.5) / (df + 0.5))
            for number, freq in postings:
                length_norm = 1 - self.b + self.b * self.lengths[number] / self.avgdl
                scores[number] += idf * freq * (self.k1 + 1) / (freq + self.k1 * length_norm)
                matches[number].append(term)
        ranking = sorted(scores, key=lambda i: (-scores[i], self.documents[i]["chunk_id"]))
        results = []
        for number in ranking[:top_k]:
            if scores[number] <= 0:
                continue
            found = list(matches[number])
            only_single_han = all(len(t) == 1 and "\u3400" <= t <= "\u9fff" for t in found)
            results.append({"rank": len(results) + 1, "score": scores[number],
                            "matched_tokens": found,
                            "match_note": "仅单字重合，需检查相关性" if only_single_han else "字面检索候选，需核对原文",
                            **self.documents[number]})
        return {"query": query, "query_tokens": query_tokens, "retrieval_only": True,
                "results": results, "no_token_match": not bool(results)}

    def save(self, path: Path):
        payload = {"schema_version": self.SCHEMA_VERSION, "k1": self.k1, "b": self.b,
                   "tokenizer": self.tokenizer, "documents": self.documents,
                   "corpus_sha256": self.corpus_sha256, "doc_lengths": self.lengths,
                   "avgdl": self.avgdl,
                   "postings": dict(sorted(self.postings.items()))}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("Unsupported index version")
        if canonical_hash(payload["documents"]) != payload["corpus_sha256"]:
            raise ValueError("Index document hash mismatch")
        # Rebuild from saved source paragraphs, then verify the saved statistics.
        # This small corpus favors transparent validation over serialized objects.
        index = cls(payload["documents"], payload["k1"], payload["b"], payload["tokenizer"])
        expected = {term: [list(p) for p in postings] for term, postings in index.postings.items()}
        if index.lengths != payload["doc_lengths"] or expected != payload["postings"] or not math.isclose(index.avgdl, payload["avgdl"]):
            raise ValueError("Index statistics mismatch; rebuild the index")
        return index
