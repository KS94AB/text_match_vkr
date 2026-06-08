from __future__ import annotations

import hashlib
from typing import List

from app.models import DocumentIn, PairwiseResult, SearchHit
from app.preprocessing import shingles
from app.services.base import BaseAnalyzer

try:
    from datasketch import MinHash, MinHashLSH
except Exception:  # pragma: no cover
    MinHash = None
    MinHashLSH = None


def _stable_hash(value: str, seed: int) -> int:
    return int(hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest(), 16)


def _simple_signature(items: set[str], num_perm: int = 64) -> list[int]:
    if not items:
        return [0] * num_perm
    signature = []
    for seed in range(num_perm):
        signature.append(min(_stable_hash(item, seed) for item in items))
    return signature


def _approximate_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    if not sig_a or not sig_b:
        return 0.0
    same = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return same / max(len(sig_a), len(sig_b), 1)


class MinHashLSHAnalyzer(BaseAnalyzer):
    name = "minhash_lsh"

    def compare_documents(self, documents: List[DocumentIn], threshold: float = 0.5, shingle_size: int = 5, **kwargs) -> List[PairwiseResult]:
        sets = {doc.doc_id: shingles(doc.text, n=shingle_size) for doc in documents}
        results: List[PairwiseResult] = []

        if MinHash is not None and MinHashLSH is not None:
            signatures = {}
            for doc_id, shingle_set in sets.items():
                mh = MinHash(num_perm=128)
                for item in shingle_set:
                    mh.update(item.encode("utf-8"))
                signatures[doc_id] = mh
            lsh = MinHashLSH(threshold=threshold, num_perm=128)
            for doc_id, mh in signatures.items():
                lsh.insert(doc_id, mh)

            for i in range(len(documents)):
                for j in range(i + 1, len(documents)):
                    left = documents[i].doc_id
                    right = documents[j].doc_id
                    score = signatures[left].jaccard(signatures[right])
                    verdict = "match" if score >= threshold else "no_match"
                    candidates = lsh.query(signatures[left])
                    intersection_size = len(sets[left] & sets[right])
                    union_size = len(sets[left] | sets[right])
                    results.append(
                        PairwiseResult(
                            left_id=left,
                            right_id=right,
                            score=round(float(score), 4),
                            verdict=verdict,
                            metadata={
                                "shingle_size": shingle_size,
                                "implementation": "datasketch_lsh",
                                "num_perm": 128,
                                "candidate": right in candidates,
                                "candidate_count": len(candidates),
                                "signature_jaccard": round(float(score), 4),
                                "shared_shingle_count": intersection_size,
                                "union_shingle_count": union_size,
                                "left_shingle_count": len(sets[left]),
                                "right_shingle_count": len(sets[right]),
                            },
                        )
                    )
            return sorted(results, key=lambda r: r.score, reverse=True)

        signatures = {doc_id: _simple_signature(shingle_set, num_perm=64) for doc_id, shingle_set in sets.items()}
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                left = documents[i].doc_id
                right = documents[j].doc_id
                score = _approximate_jaccard(signatures[left], signatures[right])
                verdict = "match" if score >= threshold else "no_match"
                intersection_size = len(sets[left] & sets[right])
                union_size = len(sets[left] | sets[right])
                results.append(
                    PairwiseResult(
                        left_id=left,
                        right_id=right,
                        score=round(score, 4),
                        verdict=verdict,
                        metadata={
                            "shingle_size": shingle_size,
                            "implementation": "stable_hash_fallback",
                            "num_perm": 64,
                            "signature_jaccard": round(score, 4),
                            "shared_shingle_count": intersection_size,
                            "union_shingle_count": union_size,
                            "left_shingle_count": len(sets[left]),
                            "right_shingle_count": len(sets[right]),
                        },
                    )
                )
        return sorted(results, key=lambda r: r.score, reverse=True)

    def search(self, documents: List[DocumentIn], query_text: str, top_k: int = 10, shingle_size: int = 5, **kwargs) -> List[SearchHit]:
        query_set = shingles(query_text, n=shingle_size)
        query_sig = _simple_signature(query_set, num_perm=64)
        hits: List[SearchHit] = []
        for doc in documents:
            doc_sig = _simple_signature(shingles(doc.text, n=shingle_size), num_perm=64)
            score = _approximate_jaccard(query_sig, doc_sig)
            hits.append(SearchHit(doc_id=doc.doc_id, score=round(score, 4), metadata={}))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]
