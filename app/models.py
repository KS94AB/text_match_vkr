from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, Field


MethodName = Literal["suffix_exact", "minhash_lsh", "inverted_index", "ngram_jaccard", "tfidf_cosine"]


class DocumentIn(BaseModel):
    doc_id: str = Field(
        ...,
        description="Document identifier",
        validation_alias=AliasChoices("doc_id", "id"),
    )
    text: str = Field(..., description="Document text")
    title: Optional[str] = Field(default=None, description="Human-readable document title")


class GroundTruthPair(BaseModel):
    left_id: str
    right_id: str
    expected_match: bool
    scenario: Optional[str] = None


class GroundTruth(BaseModel):
    pairs: List[GroundTruthPair]


class AnalyzeRequest(BaseModel):
    documents: List[DocumentIn]
    method: MethodName = "ngram_jaccard"
    query_text: Optional[str] = None
    shingle_size: int = 5
    ngram_size: int = 3
    threshold: float = 0.5
    top_k: int = 10
    ground_truth: Optional[GroundTruth] = None


class PairwiseResult(BaseModel):
    left_id: str
    right_id: str
    score: float
    verdict: str
    fragment: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    doc_id: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisSummary(BaseModel):
    method: MethodName
    document_count: int
    pair_count: int
    match_count: int
    no_match_count: int
    average_score: float
    max_score: float
    min_score: float
    total_time_ms: float
    average_time_per_pair_ms: float
    method_specific_metrics: Dict[str, Any] = Field(default_factory=dict)


class ExperimentMetrics(BaseModel):
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    evaluated_pair_count: int
    labeled_pair_count: int
    missing_result_count: int = 0
    unlabeled_result_count: int = 0


class AnalyzeResponse(BaseModel):
    method: MethodName
    document_count: int
    threshold: float
    parameters: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[AnalysisSummary] = None
    experiment_metrics: Optional[ExperimentMetrics] = None
    pairwise: List[PairwiseResult] = Field(default_factory=list)
    search_results: List[SearchHit] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
