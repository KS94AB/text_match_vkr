from __future__ import annotations

import csv
import json
import time
from itertools import combinations
from pathlib import Path

from app.models import DocumentIn
from app.services.factory import ANALYZERS

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_documents.json"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

THRESHOLDS = {
    "suffix_exact": 0.095,
    "minhash_lsh": 0.3125,
    "inverted_index": 0.0877,
    "ngram_jaccard": 0.0208,
}


def load_documents() -> list[DocumentIn]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return [DocumentIn(**item) for item in payload["documents"]]


def load_labels() -> dict[tuple[str, str], int]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    labels = {}
    for item in payload["expected_similar_pairs"]:
        key = tuple(sorted((item["left_id"], item["right_id"])))
        labels[key] = 1
    return labels


def confusion_stats(method_name: str, documents: list[DocumentIn], labels: dict[tuple[str, str], int]) -> dict[str, float]:
    analyzer = ANALYZERS[method_name]
    threshold = THRESHOLDS[method_name]
    t0 = time.perf_counter()
    results = analyzer.compare_documents(documents, threshold=threshold)
    elapsed = (time.perf_counter() - t0) * 1000

    tp = fp = tn = fn = 0
    for result in results:
        key = tuple(sorted((result.left_id, result.right_id)))
        predicted = 1 if result.score >= threshold else 0
        actual = labels.get(key, 0)
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "method": method_name,
        "threshold": threshold,
        "runtime_ms": round(elapsed, 3),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main() -> None:
    documents = load_documents()
    labels = load_labels()
    summary = [confusion_stats(name, documents, labels) for name in ANALYZERS]
    summary = sorted(summary, key=lambda row: (-row["f1"], row["runtime_ms"]))

    (RESULT_DIR / "benchmark_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (RESULT_DIR / "benchmark_results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["method", "threshold", "runtime_ms", "precision", "recall", "f1", "tp", "fp", "tn", "fn"],
        )
        writer.writeheader()
        writer.writerows(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
