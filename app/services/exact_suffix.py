from __future__ import annotations

from typing import List

from app.models import DocumentIn, PairwiseResult, SearchHit
from app.preprocessing import normalize_text
from app.services.base import BaseAnalyzer

try:  # pragma: no cover
    from suffix_trees import STree
except Exception:  # pragma: no cover
    STree = None


def _longest_common_substring_dp(a: str, b: str) -> tuple[int, str]:
    if not a or not b:
        return 0, ""
    dp = [0] * (len(b) + 1)
    max_len = 0
    end_pos = 0
    for i, ca in enumerate(a, start=1):
        new_row = [0] * (len(b) + 1)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                new_row[j] = dp[j - 1] + 1
                if new_row[j] > max_len:
                    max_len = new_row[j]
                    end_pos = i
        dp = new_row
    fragment = a[end_pos - max_len:end_pos]
    return max_len, fragment.strip()


def _longest_common_substring(a: str, b: str) -> tuple[int, str, str]:
    if not a or not b:
        return 0, "", "empty"

    if STree is not None:
        try:
            tree = STree.STree([a, b])
            fragment = (tree.lcs() or "").strip()
            return len(fragment), fragment, "suffix_tree"
        except Exception:
            pass

    length, fragment = _longest_common_substring_dp(a, b)
    return length, fragment, "dynamic_programming_fallback"


class ExactSuffixAnalyzer(BaseAnalyzer):
    name = "suffix_exact"

    def compare_documents(self, documents: List[DocumentIn], threshold: float = 0.5, **kwargs) -> List[PairwiseResult]:
        results: List[PairwiseResult] = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                left = normalize_text(documents[i].text)
                right = normalize_text(documents[j].text)
                lcs_len, fragment, implementation = _longest_common_substring(left, right)
                denom = max(len(left), len(right), 1)
                score = lcs_len / denom
                verdict = "match" if score >= threshold else "no_match"
                results.append(
                    PairwiseResult(
                        left_id=documents[i].doc_id,
                        right_id=documents[j].doc_id,
                        score=round(score, 4),
                        verdict=verdict,
                        fragment=fragment[:250] if fragment else None,
                        metadata={
                            "longest_common_substring_length": lcs_len,
                            "implementation": implementation,
                            "fragment": fragment,
                            "fragment_preview": fragment[:250] if fragment else None,
                            "left_length": len(left),
                            "right_length": len(right),
                            "score_formula": "longest_common_substring_length / max(left_length, right_length)",
                        },
                    )
                )
        return sorted(results, key=lambda r: r.score, reverse=True)

    def search(self, documents: List[DocumentIn], query_text: str, top_k: int = 10, **kwargs) -> List[SearchHit]:
        query = normalize_text(query_text)
        hits: List[SearchHit] = []
        for doc in documents:
            lcs_len, fragment, implementation = _longest_common_substring(normalize_text(doc.text), query)
            score = lcs_len / max(len(query), len(doc.text), 1)
            hits.append(
                SearchHit(
                    doc_id=doc.doc_id,
                    score=round(score, 4),
                    metadata={
                        "fragment": fragment[:250],
                        "longest_common_substring_length": lcs_len,
                        "implementation": implementation,
                    },
                )
            )
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]
