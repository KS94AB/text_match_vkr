from __future__ import annotations

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import DocumentIn, PairwiseResult, SearchHit
from app.preprocessing import normalize_text
from app.services.base import BaseAnalyzer


class TfidfCosineAnalyzer(BaseAnalyzer):
    name = "tfidf_cosine"

    def _build_matrix(self, texts: list[str]):
        vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
        )
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return vectorizer, None
        return vectorizer, matrix

    def compare_documents(self, documents: List[DocumentIn], threshold: float = 0.5, **kwargs) -> List[PairwiseResult]:
        normalized_texts = [normalize_text(doc.text) for doc in documents]
        vectorizer, matrix = self._build_matrix(normalized_texts)
        vocabulary_size = len(getattr(vectorizer, "vocabulary_", {}))
        if matrix is None:
            similarity = [[0.0 for _ in documents] for _ in documents]
        else:
            similarity = cosine_similarity(matrix)

        results: List[PairwiseResult] = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                score = float(similarity[i][j])
                verdict = "match" if score >= threshold else "no_match"
                results.append(
                    PairwiseResult(
                        left_id=documents[i].doc_id,
                        right_id=documents[j].doc_id,
                        score=round(score, 4),
                        verdict=verdict,
                        metadata={
                            "vectorizer": "TfidfVectorizer",
                            "similarity_metric": "cosine_similarity",
                            "vocabulary_size": vocabulary_size,
                            "left_normalized_length": len(normalized_texts[i]),
                            "right_normalized_length": len(normalized_texts[j]),
                            "score_formula": "cosine_similarity(tfidf(left), tfidf(right))",
                        },
                    )
                )
        return sorted(results, key=lambda r: r.score, reverse=True)

    def search(self, documents: List[DocumentIn], query_text: str, top_k: int = 10, **kwargs) -> List[SearchHit]:
        texts = [normalize_text(query_text), *[normalize_text(doc.text) for doc in documents]]
        vectorizer, matrix = self._build_matrix(texts)
        vocabulary_size = len(getattr(vectorizer, "vocabulary_", {}))
        if matrix is None:
            scores = [0.0 for _ in documents]
        else:
            scores = cosine_similarity(matrix[0], matrix[1:]).ravel()

        hits = [
            SearchHit(
                doc_id=doc.doc_id,
                score=round(float(score), 4),
                metadata={
                    "vectorizer": "TfidfVectorizer",
                    "similarity_metric": "cosine_similarity",
                    "vocabulary_size": vocabulary_size,
                },
            )
            for doc, score in zip(documents, scores)
        ]
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]
