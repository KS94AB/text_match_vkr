from __future__ import annotations

from app.services.exact_suffix import ExactSuffixAnalyzer
from app.services.inverted_index import InvertedIndexAnalyzer
from app.services.minhash_lsh import MinHashLSHAnalyzer
from app.services.ngram_jaccard import NgramJaccardAnalyzer


METHOD_DETAILS = {
    "suffix_exact": {
        "title": "Точный поиск общих фрагментов",
        "description": "Поиск длинных дословных совпадений с использованием suffix-trees при наличии библиотеки и безопасного fallback.",
    },
    "minhash_lsh": {
        "title": "MinHash + LSH",
        "description": "Быстрый приближённый поиск похожих документов по шинглам и вероятностным сигнатурам.",
    },
    "inverted_index": {
        "title": "Обратный индекс",
        "description": "Отбор кандидатов и сравнение по пересечению терминов, удобно для коллекций документов.",
    },
    "ngram_jaccard": {
        "title": "N-граммы + коэффициент Жаккара",
        "description": "Простое и наглядное сравнение частичных совпадений по множествам N-грамм.",
    },
}


ANALYZERS = {
    "suffix_exact": ExactSuffixAnalyzer(),
    "minhash_lsh": MinHashLSHAnalyzer(),
    "inverted_index": InvertedIndexAnalyzer(),
    "ngram_jaccard": NgramJaccardAnalyzer(),
}


def get_analyzer(method: str):
    try:
        return ANALYZERS[method]
    except KeyError as exc:
        raise ValueError(f"Unsupported method: {method}") from exc
