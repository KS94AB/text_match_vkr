from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from app.models import DocumentIn, PairwiseResult, SearchHit


class BaseAnalyzer(ABC):
    name: str

    @abstractmethod
    def compare_documents(self, documents: List[DocumentIn], threshold: float = 0.5, **kwargs) -> List[PairwiseResult]:
        raise NotImplementedError

    @abstractmethod
    def search(self, documents: List[DocumentIn], query_text: str, top_k: int = 10, **kwargs) -> List[SearchHit]:
        raise NotImplementedError
