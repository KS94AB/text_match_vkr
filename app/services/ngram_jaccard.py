from __future__ import annotations

from typing import List

from app.models import DocumentIn, PairwiseResult, SearchHit
from app.preprocessing import token_ngrams, tokenize
from app.services.base import BaseAnalyzer

try:
    import textdistance
except Exception:  # pragma: no cover
    textdistance = None


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


class NgramJaccardAnalyzer(BaseAnalyzer):
    name = "ngram_jaccard"

    def compare_documents(self, documents: List[DocumentIn], threshold: float = 0.5, ngram_size: int = 3, **kwargs) -> List[PairwiseResult]:
        results: List[PairwiseResult] = []
        ngram_cache = {
            doc.doc_id: set(token_ngrams(tokenize(doc.text), n=ngram_size))
            for doc in documents
        }
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                left = documents[i].doc_id
                right = documents[j].doc_id
                if textdistance:
                    score = float(textdistance.jaccard.normalized_similarity(ngram_cache[left], ngram_cache[right]))
                else:
                    score = _jaccard(ngram_cache[left], ngram_cache[right])
                verdict = "match" if score >= threshold else "no_match"
                shared_ngrams = sorted(ngram_cache[left] & ngram_cache[right])
                union_ngrams = ngram_cache[left] | ngram_cache[right]
                fragment = shared_ngrams[0] if shared_ngrams else None
                results.append(
                    PairwiseResult(
                        left_id=left,
                        right_id=right,
                        score=round(score, 4),
                        verdict=verdict,
                        fragment=fragment,
                        metadata={
                            "ngram_size": ngram_size,
                            "jaccard": round(score, 4),
                            "shared_ngram_count": len(shared_ngrams),
                            "union_ngram_count": len(union_ngrams),
                            "left_ngram_count": len(ngram_cache[left]),
                            "right_ngram_count": len(ngram_cache[right]),
                            "shared_ngrams_preview": shared_ngrams[:20],
                        },
                    )
                )
        return sorted(results, key=lambda r: r.score, reverse=True)

    def search(self, documents: List[DocumentIn], query_text: str, top_k: int = 10, ngram_size: int = 3, **kwargs) -> List[SearchHit]:
        query_ngrams = set(token_ngrams(tokenize(query_text), n=ngram_size))
        hits: List[SearchHit] = []
        for doc in documents:
            doc_ngrams = set(token_ngrams(tokenize(doc.text), n=ngram_size))
            score = _jaccard(query_ngrams, doc_ngrams)
            hits.append(SearchHit(doc_id=doc.doc_id, score=round(score, 4), metadata={}))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]
