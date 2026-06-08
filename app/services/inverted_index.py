from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

from app.models import DocumentIn, PairwiseResult, SearchHit
from app.preprocessing import tokenize
from app.services.base import BaseAnalyzer

try:
    from whoosh import index
    from whoosh.fields import ID, TEXT, Schema
    from whoosh.filedb.filestore import RamStorage
    from whoosh.qparser import QueryParser
except Exception:  # pragma: no cover
    index = None
    ID = None
    TEXT = None
    Schema = None
    RamStorage = None
    QueryParser = None


def _build_inverted_index(documents: List[DocumentIn]) -> Dict[str, set[str]]:
    idx: Dict[str, set[str]] = defaultdict(set)
    for doc in documents:
        for token in set(tokenize(doc.text)):
            idx[token].add(doc.doc_id)
    return idx


def _overlap_score(tokens_a: list[str], tokens_b: list[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


class InvertedIndexAnalyzer(BaseAnalyzer):
    name = "inverted_index"

    def compare_documents(self, documents: List[DocumentIn], threshold: float = 0.5, **kwargs) -> List[PairwiseResult]:
        results: List[PairwiseResult] = []
        token_cache = {doc.doc_id: tokenize(doc.text) for doc in documents}
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                left = documents[i].doc_id
                right = documents[j].doc_id
                score = _overlap_score(token_cache[left], token_cache[right])
                verdict = "match" if score >= threshold else "no_match"
                left_terms = set(token_cache[left])
                right_terms = set(token_cache[right])
                shared_terms = sorted(left_terms & right_terms)
                union_terms = left_terms | right_terms
                results.append(
                    PairwiseResult(
                        left_id=left,
                        right_id=right,
                        score=round(score, 4),
                        verdict=verdict,
                        metadata={
                            "shared_terms": shared_terms[:30],
                            "shared_terms_count": len(shared_terms),
                            "union_terms_count": len(union_terms),
                            "left_terms_count": len(left_terms),
                            "right_terms_count": len(right_terms),
                            "score_formula": "shared_terms_count / union_terms_count",
                        },
                    )
                )
        return sorted(results, key=lambda r: r.score, reverse=True)

    def search(self, documents: List[DocumentIn], query_text: str, top_k: int = 10, **kwargs) -> List[SearchHit]:
        if Schema and RamStorage and QueryParser:
            storage = RamStorage()
            schema = Schema(doc_id=ID(stored=True, unique=True), content=TEXT(stored=True))
            ix = storage.create_index(schema)
            writer = ix.writer()
            for doc in documents:
                writer.add_document(doc_id=doc.doc_id, content=doc.text)
            writer.commit()

            with ix.searcher() as searcher:
                query = QueryParser("content", ix.schema).parse(query_text)
                results = searcher.search(query, limit=top_k)
                return [
                    SearchHit(doc_id=hit["doc_id"], score=round(float(hit.score), 4), metadata={})
                    for hit in results
                ]

        query_tokens = tokenize(query_text)
        hits: List[SearchHit] = []
        for doc in documents:
            score = _overlap_score(query_tokens, tokenize(doc.text))
            hits.append(SearchHit(doc_id=doc.doc_id, score=round(score, 4), metadata={}))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]
