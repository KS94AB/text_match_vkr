from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.document_loader import load_documents_from_uploads
from app.models import AnalysisSummary, AnalyzeRequest, AnalyzeResponse, ExperimentMetrics, GroundTruth
from app.services.factory import ANALYZERS, METHOD_DETAILS, get_analyzer

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="Text Match Research Service",
    version="0.2.0",
    description="Учебный веб-сервис для сравнительного анализа методов поиска совпадений в текстах.",
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _pair_key(left_id: str, right_id: str) -> frozenset[str]:
    return frozenset((left_id, right_id))


def _safe_divide(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _build_method_specific_metrics(method: str, pairwise: list[Any]) -> dict[str, Any]:
    metadata_items = [result.metadata or {} for result in pairwise]

    if method == "suffix_exact":
        lengths = [
            float(metadata.get("longest_common_substring_length", 0))
            for metadata in metadata_items
        ]
        return {
            "average_common_fragment_length": _average(lengths),
            "max_common_fragment_length": int(max(lengths)) if lengths else 0,
        }

    if method == "minhash_lsh":
        candidate_counts = [
            float(metadata.get("candidate_count", 0))
            for metadata in metadata_items
            if "candidate_count" in metadata
        ]
        signature_scores = [
            float(metadata.get("signature_jaccard", 0))
            for metadata in metadata_items
            if "signature_jaccard" in metadata
        ]
        return {
            "total_lsh_candidates": int(sum(candidate_counts)) if candidate_counts else 0,
            "average_lsh_candidates": _average(candidate_counts),
            "average_signature_similarity": _average(signature_scores),
        }

    if method == "inverted_index":
        shared_terms = [
            float(metadata.get("shared_terms_count", 0))
            for metadata in metadata_items
            if "shared_terms_count" in metadata
        ]
        return {
            "average_shared_terms": _average(shared_terms),
        }

    if method == "ngram_jaccard":
        shared_ngrams = [
            float(metadata.get("shared_ngram_count", 0))
            for metadata in metadata_items
            if "shared_ngram_count" in metadata
        ]
        union_ngrams = [
            float(metadata.get("union_ngram_count", 0))
            for metadata in metadata_items
            if "union_ngram_count" in metadata
        ]
        return {
            "average_shared_ngrams": _average(shared_ngrams),
            "average_union_ngrams": _average(union_ngrams),
        }

    if method == "tfidf_cosine":
        vocabulary_sizes = [
            float(metadata.get("vocabulary_size", 0))
            for metadata in metadata_items
            if "vocabulary_size" in metadata
        ]
        return {
            "average_vocabulary_size": _average(vocabulary_sizes),
        }

    return {}


def _calculate_experiment_metrics(
    ground_truth: GroundTruth | None,
    pairwise: list[Any],
) -> ExperimentMetrics | None:
    if ground_truth is None:
        return None

    expected_by_pair: dict[frozenset[str], bool] = {}
    for pair in ground_truth.pairs:
        key = _pair_key(pair.left_id, pair.right_id)
        if len(key) != 2:
            raise HTTPException(status_code=400, detail="Эталонная пара должна содержать два разных идентификатора документов.")
        if key in expected_by_pair:
            raise HTTPException(status_code=400, detail=f"Дублирующаяся эталонная пара: {pair.left_id} / {pair.right_id}.")
        expected_by_pair[key] = pair.expected_match

    result_by_pair = {_pair_key(result.left_id, result.right_id): result for result in pairwise}
    tp = fp = fn = tn = 0
    evaluated_pair_count = 0

    for key, expected_match in expected_by_pair.items():
        result = result_by_pair.get(key)
        if result is None:
            continue
        predicted_match = result.verdict == "match"
        evaluated_pair_count += 1
        if predicted_match and expected_match:
            tp += 1
        elif predicted_match and not expected_match:
            fp += 1
        elif not predicted_match and expected_match:
            fn += 1
        else:
            tn += 1

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return ExperimentMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        evaluated_pair_count=evaluated_pair_count,
        labeled_pair_count=len(expected_by_pair),
        missing_result_count=len(expected_by_pair) - evaluated_pair_count,
        unlabeled_result_count=sum(1 for key in result_by_pair if key not in expected_by_pair),
    )


def _annotate_pairwise_with_ground_truth(
    ground_truth: GroundTruth | None,
    pairwise: list[Any],
) -> None:
    if ground_truth is None:
        return

    labels = {
        _pair_key(pair.left_id, pair.right_id): pair
        for pair in ground_truth.pairs
    }
    for result in pairwise:
        label = labels.get(_pair_key(result.left_id, result.right_id))
        if label is None:
            result.metadata["ground_truth"] = None
            continue
        predicted_match = result.verdict == "match"
        if predicted_match and label.expected_match:
            outcome = "TP"
        elif predicted_match and not label.expected_match:
            outcome = "FP"
        elif not predicted_match and label.expected_match:
            outcome = "FN"
        else:
            outcome = "TN"
        result.metadata.update(
            {
                "expected_match": label.expected_match,
                "scenario": label.scenario,
                "experiment_outcome": outcome,
            }
        )


async def _load_ground_truth_from_upload(upload: UploadFile | None) -> GroundTruth | None:
    if upload is None or not upload.filename:
        return None
    if Path(upload.filename).suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="Файл эталонной разметки должен быть в формате .json.")

    file_bytes = await upload.read()
    if not file_bytes:
        return None

    try:
        payload = json.loads(file_bytes.decode("utf-8-sig"))
        return GroundTruth.model_validate(payload)
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Не удалось декодировать JSON эталонной разметки как UTF-8.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Некорректный JSON эталонной разметки: {exc}") from exc


def run_analysis(request: AnalyzeRequest) -> AnalyzeResponse:
    if len(request.documents) < 2 and not request.query_text:
        raise HTTPException(status_code=400, detail="Для попарного сравнения требуется минимум два документа.")

    try:
        analyzer = get_analyzer(request.method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.perf_counter()

    pairwise = analyzer.compare_documents(
        request.documents,
        threshold=request.threshold,
        shingle_size=request.shingle_size,
        ngram_size=request.ngram_size,
    ) if len(request.documents) >= 2 else []

    search_results = analyzer.search(
        request.documents,
        query_text=request.query_text,
        top_k=request.top_k,
        shingle_size=request.shingle_size,
        ngram_size=request.ngram_size,
    ) if request.query_text else []

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    scores = [result.score for result in pairwise]
    pair_count = len(pairwise)
    match_count = sum(1 for result in pairwise if result.verdict == "match")
    no_match_count = sum(1 for result in pairwise if result.verdict == "no_match")
    summary = AnalysisSummary(
        method=request.method,
        document_count=len(request.documents),
        pair_count=pair_count,
        match_count=match_count,
        no_match_count=no_match_count,
        average_score=round(sum(scores) / pair_count, 4) if pair_count else 0.0,
        max_score=round(max(scores), 4) if scores else 0.0,
        min_score=round(min(scores), 4) if scores else 0.0,
        total_time_ms=elapsed_ms,
        average_time_per_pair_ms=round(elapsed_ms / pair_count, 2) if pair_count else 0.0,
        method_specific_metrics=_build_method_specific_metrics(request.method, pairwise),
    )
    experiment_metrics = _calculate_experiment_metrics(request.ground_truth, pairwise)
    _annotate_pairwise_with_ground_truth(request.ground_truth, pairwise)
    parameters = {
        "threshold": request.threshold,
        "shingle_size": request.shingle_size,
        "ngram_size": request.ngram_size,
        "top_k": request.top_k,
        "query_text_provided": bool(request.query_text),
        "experiment_mode": experiment_metrics is not None,
    }

    notes = [
        "Результаты предназначены для учебного и исследовательского сравнения методов.",
        "Для коротких документов чувствительность к порогу схожести рекомендуется подбирать эмпирически.",
    ]
    return AnalyzeResponse(
        method=request.method,
        document_count=len(request.documents),
        threshold=request.threshold,
        parameters=parameters,
        summary=summary,
        experiment_metrics=experiment_metrics,
        pairwise=pairwise,
        search_results=search_results,
        notes=notes,
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "method_details": METHOD_DETAILS,
            "default_method": "ngram_jaccard",
        },
    )


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/methods")
def list_methods() -> dict[str, list[str]]:
    return {"methods": list(ANALYZERS.keys())}


@app.get("/method-details")
def method_details() -> dict[str, Any]:
    return {"methods": METHOD_DETAILS}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    return run_analysis(request)


@app.post("/analyze-upload")
async def analyze_upload(
    method: str = Form("ngram_jaccard"),
    threshold: float = Form(0.5),
    shingle_size: int = Form(5),
    ngram_size: int = Form(3),
    top_k: int = Form(10),
    query_text: str = Form(""),
    files: list[UploadFile] = File(...),
    ground_truth_file: UploadFile | None = File(None),
) -> dict[str, Any]:
    documents = await load_documents_from_uploads(files)
    ground_truth = await _load_ground_truth_from_upload(ground_truth_file)
    payload = AnalyzeRequest(
        documents=documents,
        method=method,
        threshold=threshold,
        shingle_size=shingle_size,
        ngram_size=ngram_size,
        top_k=top_k,
        query_text=query_text or None,
        ground_truth=ground_truth,
    )

    analysis = run_analysis(payload)

    return {
        "analysis": analysis.model_dump(),
        "uploaded_documents": [doc.model_dump() for doc in documents],
        "elapsed_ms": analysis.summary.total_time_ms if analysis.summary else 0.0,
        "method_meta": METHOD_DETAILS[analysis.method],
    }
