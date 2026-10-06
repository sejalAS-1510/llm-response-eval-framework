"""
tests/test_agent_eval_dataset.py
---------------------------------
Validation suite evaluating the multi-category benchmark test dataset across all 4 Judge Agents:
- Correct / Fully Grounded responses
- Factually Incorrect / Contradictory responses
- Irrelevant / Off-Topic responses
- Incomplete responses (Full vs. Partial vs. Substantially Incomplete)
- Hallucinated / Unsupported responses

Asserts:
1. Relevance Agent distinguishes on-topic vs. off-topic correctly.
2. Accuracy Agent's score and reasoning align with reference answers and evidence.
3. Hallucination Agent correctly flags unsupported claims and links them to missing source support.
4. Completeness Agent distinguishes full, partial, and substantially-incomplete responses correctly.
5. No agent's reasoning contradicts its own score (Self-Consistency across all records).
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List

# Ensure mock/offline evaluation mode is active for fast, deterministic evaluation
os.environ["MOCK_LLM"] = "1"

import pytest
from src.agents.orchestrator import EvaluationOrchestrator
from src.agents.schemas import (
    EvaluationResult,
    RelevanceResult,
    AccuracyResult,
    HallucinationResult,
    CompletenessResult,
)

DATASET_FILE = Path(__file__).resolve().parent.parent / "data" / "agent_evaluation_benchmark_dataset.json"


@pytest.fixture(scope="module")
def benchmark_dataset() -> List[Dict[str, Any]]:
    """Loads the curated benchmark test dataset."""
    assert DATASET_FILE.exists(), f"Benchmark dataset not found at {DATASET_FILE}"
    content = DATASET_FILE.read_text(encoding="utf-8")
    dataset = json.loads(content)
    assert len(dataset) >= 10, f"Expected at least 10 benchmark records, found {len(dataset)}"
    return dataset


@pytest.fixture(scope="module")
def orchestrator() -> EvaluationOrchestrator:
    return EvaluationOrchestrator()


@pytest.fixture(scope="module")
async def evaluated_dataset(benchmark_dataset, orchestrator) -> Dict[str, EvaluationResult]:
    """Pre-evaluates all benchmark records once for the test module."""
    results = {}
    for item in benchmark_dataset:
        res = await orchestrator.evaluate(
            question=item["question"],
            ai_response=item["ai_response"],
            reference_answer=item.get("reference_answer"),
            source_document=item.get("source_document"),
        )
        results[item["id"]] = res
    return results


# ===========================================================================
# TEST 1: DATASET STRUCTURE & CATEGORY COVERAGE
# ===========================================================================

def test_dataset_coverage(benchmark_dataset):
    """Verifies that the dataset covers all 5 requested evaluation categories."""
    categories = {item["category"] for item in benchmark_dataset}
    expected_categories = {"correct", "factually_incorrect", "irrelevant", "incomplete", "hallucinated"}
    assert expected_categories.issubset(categories), f"Missing categories: {expected_categories - categories}"

    # Verify each item has required keys
    for item in benchmark_dataset:
        assert "id" in item
        assert "category" in item
        assert "question" in item
        assert "ai_response" in item
        assert "reference_answer" in item
        assert "expected" in item


# ===========================================================================
# TEST 2: RELEVANCE AGENT DISTINGUISHES ON/OFF-TOPIC CORRECTLY
# ===========================================================================

@pytest.mark.asyncio
async def test_relevance_agent_distinguishes_on_off_topic(benchmark_dataset, evaluated_dataset):
    """
    Asserts: Relevance Agent clearly separates on-topic from off-topic responses.
    - On-topic items (correct, incomplete, contradictory, hallucinated) score >= 0.40
      and are classified as 'fully_relevant' or 'partially_relevant'.
    - Off-topic items (irrelevant recipe, unrelated architecture) score <= 0.20
      and are classified as 'unrelated' or 'off_topic'.
    - Clear score gap: min(on_topic) > max(off_topic).
    """
    on_topic_scores = []
    off_topic_scores = []

    for item in benchmark_dataset:
        item_id = item["id"]
        res = evaluated_dataset[item_id]
        rel: RelevanceResult = res.relevance

        is_on_topic = item["expected"].get("relevance_is_on_topic", True)
        if is_on_topic:
            on_topic_scores.append(rel.score)
            assert rel.score >= 0.40, f"Expected on-topic score >= 0.40 for {item_id}, got {rel.score}"
            assert rel.classification in ("fully_relevant", "partially_relevant"), (
                f"Expected on-topic classification for {item_id}, got {rel.classification}"
            )
        else:
            off_topic_scores.append(rel.score)
            assert rel.score <= 0.20, f"Expected off-topic score <= 0.20 for {item_id}, got {rel.score}"
            assert rel.classification in ("unrelated", "off_topic"), (
                f"Expected off-topic classification for {item_id}, got {rel.classification}"
            )

    assert len(on_topic_scores) >= 8
    assert len(off_topic_scores) >= 2
    # Verify strict mathematical separation
    assert min(on_topic_scores) > max(off_topic_scores), (
        f"Relevance overlap: min(on_topic)={min(on_topic_scores)} not greater than max(off_topic)={max(off_topic_scores)}"
    )


# ===========================================================================
# TEST 3: ACCURACY AGENT ALIGNS WITH REFERENCE ANSWERS AND EVIDENCE
# ===========================================================================

@pytest.mark.asyncio
async def test_accuracy_agent_aligns_with_reference_and_evidence(benchmark_dataset, evaluated_dataset):
    """
    Asserts: Accuracy Agent's score and reasoning accurately align with reference answers and evidence.
    - Correct responses: score >= 0.85, classification == 'correct', supporting evidence populated,
      and reasoning confirms alignment with reference.
    - Factually incorrect / Contradictory responses: score <= 0.50 (<= 0.25 for direct contradiction),
      classification in ('contradictory', 'incorrect', 'partially_correct'),
      supporting evidence cites conflicting reference context,
      and reasoning articulates mismatch/contradiction.
    """
    # 1. Correct responses check
    correct_items = [it for it in benchmark_dataset if it["category"] == "correct"]
    for it in correct_items:
        res = evaluated_dataset[it["id"]]
        acc: AccuracyResult = res.accuracy
        assert acc.score >= 0.85, f"Expected accuracy >= 0.85 for correct item {it['id']}, got {acc.score}"
        assert acc.classification == "correct"
        assert len(acc.supporting_evidence) > 0, f"Expected supporting evidence for {it['id']}"
        # Reasoning must affirm alignment
        reasoning_lower = acc.reasoning.lower()
        assert any(term in reasoning_lower for term in ("align", "correct", "reference", "ground")), (
            f"Reasoning does not affirm alignment: {acc.reasoning}"
        )

    # 2. Factually incorrect / Contradictory responses check
    incorr_items = [it for it in benchmark_dataset if it["category"] == "factually_incorrect"]
    for it in incorr_items:
        res = evaluated_dataset[it["id"]]
        acc: AccuracyResult = res.accuracy
        assert acc.score <= 0.50, f"Expected accuracy <= 0.50 for incorrect item {it['id']}, got {acc.score}"
        assert acc.classification in ("contradictory", "incorrect", "partially_correct")
        # Evidence must contain reference context
        assert len(acc.supporting_evidence) > 0
        # Reasoning must articulate factual discrepancy
        reasoning_lower = acc.reasoning.lower()
        assert any(term in reasoning_lower for term in ("contradict", "inaccurate", "unverified", "discrepancy", "mismatch")), (
            f"Reasoning does not articulate discrepancy: {acc.reasoning}"
        )

    # Specific check for direct acoustic contradiction INCORR-02
    res_sound = evaluated_dataset["INCORR-02"]
    assert res_sound.accuracy.score <= 0.25
    assert res_sound.accuracy.classification == "contradictory"
    assert "contradicts" in res_sound.accuracy.reasoning.lower()


# ===========================================================================
# TEST 4: HALLUCINATION AGENT FLAGS UNSUPPORTED CLAIMS & LINKS TO MISSING SOURCE
# ===========================================================================

@pytest.mark.asyncio
async def test_hallucination_agent_flags_unsupported_claims_and_links_missing_source(
    benchmark_dataset, evaluated_dataset
):
    """
    Asserts: Hallucination Agent correctly flags unsupported claims and links them to missing source support.
    - Hallucinated responses (HAL-01, HAL-02):
      - is_hallucinated is True.
      - hallucination_score > 0.
      - flagged_claims contains >= 1 claim with status 'unsupported' or 'contradicted'.
      - Flagged claim explanation links to missing context support (e.g. 'not mentioned in reference context',
        'ungrounded', 'unsupported').
    - Fully grounded responses (CORR-01, CORR-02):
      - is_hallucinated is False.
      - hallucination_score == 0.0.
      - flagged_claims is empty.
      - Reasoning confirms all statements appear grounded.
    """
    # 1. Hallucinated responses
    hal_items = [it for it in benchmark_dataset if it["category"] == "hallucinated"]
    for it in hal_items:
        res = evaluated_dataset[it["id"]]
        hal: HallucinationResult = res.hallucination
        assert hal.is_hallucinated is True, f"Expected is_hallucinated=True for {it['id']}"
        assert hal.hallucination_score > 0.0
        assert len(hal.flagged_claims) >= 1, f"Expected flagged claims for {it['id']}"

        for fc in hal.flagged_claims:
            assert fc.status in ("unsupported", "contradicted")
            exp_lower = fc.explanation.lower()
            assert any(
                term in exp_lower
                for term in ("not mentioned", "ungrounded", "unsupported", "fails to state", "contradict", "fabricated")
            ), f"Flagged claim explanation does not link to missing support: {fc.explanation}"

    # 2. Fully grounded responses
    grounded_items = [it for it in benchmark_dataset if it["category"] == "correct"]
    for it in grounded_items:
        res = evaluated_dataset[it["id"]]
        hal: HallucinationResult = res.hallucination
        assert hal.is_hallucinated is False, f"False positive hallucination on {it['id']}"
        assert hal.hallucination_score == 0.0
        assert len(hal.flagged_claims) == 0
        assert "grounded" in hal.reasoning.lower()


# ===========================================================================
# TEST 5: COMPLETENESS AGENT DISTINGUISHES FULL / PARTIAL / SUBSTANTIALLY INCOMPLETE
# ===========================================================================

@pytest.mark.asyncio
async def test_completeness_agent_distinguishes_levels_of_completeness(benchmark_dataset, evaluated_dataset):
    """
    Asserts: Completeness Agent distinguishes full, partial, and substantially-incomplete responses.
    - Full: score >= 0.85, classification == 'fully_complete', 0 missing aspects.
    - Partial: 0.40 <= score <= 0.75, classification in ('partially_complete', 'mostly_complete'),
      missing_aspects specifies missing sub-questions (e.g. population, currency).
    - Substantially Incomplete: score <= 0.35, classification == 'incomplete',
      missing_aspects contains multiple major omitted requirements (stages of mitosis).
    - Relative ordering: full_score > partial_score > substantially_incomplete_score.
    """
    # 1. Full completeness (CORR-01, CORR-02)
    res_full = evaluated_dataset["CORR-01"]
    comp_full: CompletenessResult = res_full.completeness
    assert comp_full.score >= 0.85
    assert comp_full.classification == "fully_complete"
    assert len(comp_full.missing_aspects) == 0
    assert "thoroughly addressed" in comp_full.reasoning.lower()

    # 2. Partial completeness (INCOMP-PARTIAL: Canberra given, population & currency omitted)
    res_partial = evaluated_dataset["INCOMP-PARTIAL"]
    comp_partial: CompletenessResult = res_partial.completeness
    assert 0.40 <= comp_partial.score <= 0.75
    assert comp_partial.classification in ("partially_complete", "mostly_complete")
    assert len(comp_partial.missing_aspects) >= 1
    # Check that missing aspects explicitly mention population or currency
    missing_str = " ".join(comp_partial.missing_aspects).lower()
    assert ("population" in missing_str) or ("currency" in missing_str), (
        f"Missing aspects did not identify population/currency: {comp_partial.missing_aspects}"
    )

    # 3. Substantially incomplete (INCOMP-SUBSTANTIAL: 4 stages omitted)
    res_incomp = evaluated_dataset["INCOMP-SUBSTANTIAL"]
    comp_incomp: CompletenessResult = res_incomp.completeness
    assert comp_incomp.score <= 0.35
    assert comp_incomp.classification == "incomplete"
    assert len(comp_incomp.missing_aspects) >= 3, (
        f"Expected at least 3 missing aspects for 4-stage mitosis question, got {len(comp_incomp.missing_aspects)}"
    )

    # 4. Strict relative ordering
    assert comp_full.score > comp_partial.score > comp_incomp.score, (
        f"Expected full ({comp_full.score}) > partial ({comp_partial.score}) > incomplete ({comp_incomp.score})"
    )


# ===========================================================================
# TEST 6: NO AGENT'S REASONING CONTRADICTS ITS OWN SCORE (SELF-CONSISTENCY)
# ===========================================================================

@pytest.mark.asyncio
async def test_no_agent_reasoning_contradicts_own_score(benchmark_dataset, evaluated_dataset):
    """
    Asserts: For EVERY record in the benchmark dataset, no agent's reasoning contradicts
    its own assigned score or classification.

    Consistency Rubric:
    - Relevance:
        * High score (>= 0.70) must NOT claim the response is unrelated or fails to address the question.
        * Low score (<= 0.30) must NOT claim the response completely and directly answers the question.
    - Accuracy:
        * High score (>= 0.80) must NOT state the response directly contradicts or is inaccurate.
        * Low score (<= 0.35) must NOT declare all statements align perfectly.
    - Hallucination:
        * Score 0.0 (grounded) must NOT declare unsupported or fabricated claims were identified.
        * Score > 0.0 (hallucinated) must NOT state all statements appear grounded with zero issues.
    - Completeness:
        * High score (>= 0.85) must NOT claim the response is substantially incomplete.
        * Low score (<= 0.35) must NOT claim all core aspects are thoroughly addressed.
    """
    for item in benchmark_dataset:
        item_id = item["id"]
        res: EvaluationResult = evaluated_dataset[item_id]

        rel_score = res.relevance.score
        rel_reasoning = res.relevance.reasoning.lower()

        acc_score = res.accuracy.score
        acc_reasoning = res.accuracy.reasoning.lower()

        hal_score = res.hallucination.hallucination_score
        hal_reasoning = res.hallucination.reasoning.lower()

        comp_score = res.completeness.score
        comp_reasoning = res.completeness.reasoning.lower()

        # -------------------------------------------------------------------
        # 1. Relevance Self-Consistency
        # -------------------------------------------------------------------
        if rel_score >= 0.70:
            assert "unrelated" not in rel_reasoning, f"[{item_id}] High relevance ({rel_score}) states 'unrelated': {rel_reasoning}"
            assert "does not address" not in rel_reasoning, f"[{item_id}] High relevance ({rel_score}) states 'does not address': {rel_reasoning}"
        if rel_score <= 0.30:
            assert "completely addresses" not in rel_reasoning, f"[{item_id}] Low relevance ({rel_score}) states 'completely addresses': {rel_reasoning}"

        # -------------------------------------------------------------------
        # 2. Accuracy Self-Consistency
        # -------------------------------------------------------------------
        if acc_score >= 0.80:
            assert "contradicts" not in acc_reasoning, f"[{item_id}] High accuracy ({acc_score}) states 'contradicts': {acc_reasoning}"
            assert "inaccurate" not in acc_reasoning, f"[{item_id}] High accuracy ({acc_score}) states 'inaccurate': {acc_reasoning}"
        if acc_score <= 0.30:
            assert "all factual statements align" not in acc_reasoning, f"[{item_id}] Low accuracy ({acc_score}) states 'all statements align': {acc_reasoning}"

        # -------------------------------------------------------------------
        # 3. Hallucination Self-Consistency
        # -------------------------------------------------------------------
        if hal_score == 0.0 and not res.hallucination.is_hallucinated:
            assert "unsupported" not in hal_reasoning, f"[{item_id}] Grounded response states 'unsupported': {hal_reasoning}"
            assert "contradicted" not in hal_reasoning, f"[{item_id}] Grounded response states 'contradicted': {hal_reasoning}"
            assert len(res.hallucination.flagged_claims) == 0
        if hal_score > 0.0 or res.hallucination.is_hallucinated:
            assert "all statements appear grounded" not in hal_reasoning, (
                f"[{item_id}] Hallucinated response states 'all statements appear grounded': {hal_reasoning}"
            )
            assert len(res.hallucination.flagged_claims) > 0

        # -------------------------------------------------------------------
        # 4. Completeness Self-Consistency
        # -------------------------------------------------------------------
        if comp_score >= 0.85:
            assert "substantially incomplete" not in comp_reasoning, f"[{item_id}] High completeness ({comp_score}) states 'substantially incomplete': {comp_reasoning}"
            assert "fails to address" not in comp_reasoning, f"[{item_id}] High completeness ({comp_score}) states 'fails to address': {comp_reasoning}"
        if comp_score <= 0.35:
            assert "thoroughly addressed" not in comp_reasoning, f"[{item_id}] Low completeness ({comp_score}) states 'thoroughly addressed': {comp_reasoning}"
