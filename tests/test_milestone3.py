"""
Milestone 3 Test & Validation Suite.
Tests:
1. M3.1 Completeness Judge Agent (fully complete, partially complete, incomplete).
2. M3.2 Verdict Agent (weighted scoring, Pass/Needs Improvement/Fail thresholds, critical overrides).
3. M3.4 Batch Evaluation Module (CSV parsing, empty question handling, 100+ records evaluation, batch stats, export).
4. End-to-End Orchestrator Integration (all 4 dimensions + overall verdict).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import logging
from src.agents.schemas import (
    RelevanceResult,
    AccuracyResult,
    HallucinationResult,
    CompletenessResult,
    ClaimEvaluation,
    EvaluationResult,
)
from src.agents.completeness_agent import CompletenessAgent
from src.agents.verdict_agent import VerdictAgent
from src.agents.orchestrator import EvaluationOrchestrator
from src.agents.batch_evaluator import BatchEvaluator

logging.basicConfig(level=logging.WARNING)


async def test_completeness_agent():
    print("\n--- 1. Testing M3.1 Completeness Judge Agent ---")
    agent = CompletenessAgent()

    # Case A: Fully Complete
    q_complete = "What is the capital of France and what is its official language?"
    a_complete = "The capital of France is Paris, and the official language is French."
    res_a = await agent.evaluate(q_complete, a_complete)
    print(f"Case A (Fully complete): score={res_a.score}, class={res_a.classification}, missing={res_a.missing_aspects}")
    assert res_a.score >= 0.75, f"Expected score >= 0.75, got {res_a.score}"
    assert res_a.classification in ("fully_complete", "mostly_complete")

    # Case B: Partially Complete (omits second sub-question)
    q_partial = "What is the capital of France and what is its official language?"
    a_partial = "The capital of France is Paris."
    res_b = await agent.evaluate(q_partial, a_partial)
    print(f"Case B (Partial omission): score={res_b.score}, class={res_b.classification}, missing={res_b.missing_aspects}")
    assert res_b.score <= 0.80, f"Expected lower score for missing language, got {res_b.score}"

    # Case C: Substantially Incomplete
    q_incomp = "Explain the detailed causes of World War I, including alliances, imperialism, and the assassination."
    a_incomp = "It started in Europe."
    res_c = await agent.evaluate(q_incomp, a_incomp)
    print(f"Case C (Substantially incomplete): score={res_c.score}, class={res_c.classification}, missing={res_c.missing_aspects}")
    assert res_c.score <= 0.50, f"Expected <= 0.50, got {res_c.score}"
    print("[PASS] Completeness Judge Agent passed all test cases!")


def test_verdict_agent():
    print("\n--- 2. Testing M3.2 Verdict Agent & Weighted Evaluation ---")
    verdict_agent = VerdictAgent(
        weight_accuracy=0.35,
        weight_completeness=0.25,
        weight_relevance=0.20,
        weight_groundedness=0.20,
    )

    # Mock inputs helper
    def make_results(rel_score, acc_score, acc_class, hal_score, is_hal, comp_score):
        return (
            RelevanceResult(score=rel_score, classification="fully_relevant", reasoning="Relevant"),
            AccuracyResult(score=acc_score, classification=acc_class, supporting_evidence=[], reasoning="Acc"),
            HallucinationResult(
                is_hallucinated=is_hal,
                hallucination_score=hal_score,
                total_claims=2,
                unsupported_claims_count=1 if is_hal else 0,
                flagged_claims=[],
                all_claims=[],
                reasoning="Hal reasoning",
            ),
            CompletenessResult(
                score=comp_score,
                classification="fully_complete" if comp_score >= 0.8 else "partially_complete",
                identified_requirements=["Q1"],
                addressed_aspects=["Q1"] if comp_score >= 0.8 else [],
                partially_addressed_aspects=[],
                missing_aspects=[] if comp_score >= 0.8 else ["Sub-question 2"],
                reasoning="Comp reasoning",
            ),
        )

    # 1. Standard Pass Case (all dimensions high, no hallucination)
    rel, acc, hal, comp = make_results(1.0, 1.0, "correct", 0.0, False, 1.0)
    v1 = verdict_agent.evaluate(rel, acc, hal, comp)
    print(f"Scenario 1 (High Quality): weighted_score={v1.weighted_score}, verdict={v1.verdict}")
    assert v1.verdict == "Pass", f"Expected Pass, got {v1.verdict}"
    assert v1.weighted_score >= 0.90

    # 2. Critical Contradiction Override (even with high relevance & completeness)
    rel, acc, hal, comp = make_results(1.0, 0.0, "contradictory", 0.0, False, 1.0)
    v2 = verdict_agent.evaluate(rel, acc, hal, comp)
    print(f"Scenario 2 (Contradiction Override): weighted_score={v2.weighted_score}, verdict={v2.verdict}")
    assert v2.verdict == "Fail", f"Expected Fail on contradiction, got {v2.verdict}"
    assert any("Contradiction" in issue for issue in v2.major_issues)

    # 3. Severe Hallucination Override (>= 50% hallucinated)
    rel, acc, hal, comp = make_results(1.0, 0.5, "partially_correct", 0.6, True, 1.0)
    v3 = verdict_agent.evaluate(rel, acc, hal, comp)
    print(f"Scenario 3 (Severe Hallucination): weighted_score={v3.weighted_score}, verdict={v3.verdict}")
    assert v3.verdict == "Fail", f"Expected Fail on severe hallucination, got {v3.verdict}"

    # 4. Moderate Hallucination Cap (Cannot Pass, capped at Needs Improvement)
    rel, acc, hal, comp = make_results(1.0, 0.85, "correct", 0.2, True, 0.9)
    v4 = verdict_agent.evaluate(rel, acc, hal, comp)
    print(f"Scenario 4 (Hallucination Cap): weighted_score={v4.weighted_score}, verdict={v4.verdict}")
    assert v4.verdict == "Needs Improvement", f"Expected Needs Improvement, got {v4.verdict}"

    # 5. Major Incompleteness Cap (Completeness < 0.5)
    rel, acc, hal, comp = make_results(1.0, 0.9, "correct", 0.0, False, 0.3)
    v5 = verdict_agent.evaluate(rel, acc, hal, comp)
    print(f"Scenario 5 (Major Incompleteness): weighted_score={v5.weighted_score}, verdict={v5.verdict}")
    assert v5.verdict == "Needs Improvement", f"Expected Needs Improvement, got {v5.verdict}"

    print("[PASS] Verdict Agent passed all weighted evaluation and override tests!")


async def test_batch_evaluator():
    print("\n--- 3. Testing M3.4 Batch Evaluation Module ---")
    orchestrator = EvaluationOrchestrator()
    evaluator = BatchEvaluator(orchestrator=orchestrator, concurrency_limit=10)

    # Test CSV parsing with varied column names and empty question
    csv_sample = """questions,ai responses,reference_answer
What is the capital of France?,Paris is the capital of France.,Paris is the capital of France.
,Water is a clear liquid that boils at 100 degrees Celsius.,Water boils at 100C.
Explain gravity.,Gravity is a fundamental force of attraction between masses.,Gravity pulls objects together.
Who wrote Hamlet?,William Shakespeare wrote the tragedy of Hamlet.,Shakespeare wrote Hamlet.
"""
    records, warnings = evaluator.parse_and_validate_csv(csv_sample)
    print(f"Parsed {len(records)} records with {len(warnings)} warning(s).")
    assert len(records) == 4
    assert records[1]["question"] == "[No Question Provided]", "Empty question should be handled gracefully"

    # Evaluate small batch
    summary = await evaluator.evaluate_batch(records, filename="test_sample.csv")
    print(f"Batch evaluated: id={summary.batch_id}, total={summary.statistics.total_records}, pass_rate={summary.statistics.pass_rate_percent}%")
    assert summary.statistics.total_records == 4
    assert summary.statistics.successful_records == 4
    assert len(summary.items) == 4

    # Test CSV export
    exported_csv = evaluator.export_summary_to_csv(summary)
    assert "Overall Weighted Score" in exported_csv
    assert "Verdict" in exported_csv

    # Test High-Volume (100+ Records) Batch Processing
    print("Testing 100+ records batch evaluation performance...")
    large_records = []
    for i in range(1, 105):
        large_records.append({
            "index": i,
            "question": f"Question {i}: What is the chemical formula for water?",
            "ai_response": f"Answer {i}: Water is chemical compound H2O.",
            "reference_answer": "Water is H2O.",
            "source_document": None,
        })

    large_summary = await evaluator.evaluate_batch(large_records, filename="large_100_batch.csv")
    print(f"100+ Batch evaluated: total={large_summary.statistics.total_records}, successful={large_summary.statistics.successful_records}")
    assert large_summary.statistics.total_records == 104
    assert large_summary.statistics.successful_records == 104
    print("[PASS] Batch Evaluation Module passed CSV validation, empty question handling, and 100+ record load tests!")



async def test_orchestrator_integration():
    print("\n--- 4. Testing End-to-End Orchestrator Integration ---")
    orchestrator = EvaluationOrchestrator()
    result = await orchestrator.evaluate(
        question="What is the speed of light, and why is it significant?",
        ai_response="The speed of light in vacuum is approximately 299,792,458 meters per second. It is significant because it represents the universal speed limit in physics.",
        reference_answer="The speed of light is ~300,000 km/s and represents the maximum speed limit in the universe.",
    )

    print(f"Orchestrator Result:")
    print(f"  Relevance: score={result.relevance.score} ({result.relevance.classification})")
    print(f"  Accuracy: score={result.accuracy.score} ({result.accuracy.classification})")
    print(f"  Hallucination: is_hal={result.hallucination.is_hallucinated} ({result.hallucination.hallucination_score})")
    print(f"  Completeness: score={result.completeness.score} ({result.completeness.classification})")
    print(f"  Verdict: verdict={result.verdict.verdict}, weighted_score={result.verdict.weighted_score}")

    assert result.relevance is not None
    assert result.accuracy is not None
    assert result.hallucination is not None
    assert result.completeness is not None
    assert result.verdict is not None
    assert result.verdict.verdict in ("Pass", "Needs Improvement", "Fail")
    print("[PASS] Orchestrator successfully integrated all Milestone 3 components!")



async def main():
    print("=" * 80)
    print(" MILESTONE 3: VALIDATION SUITE (M3.1, M3.2, M3.3, M3.4)")
    print("=" * 80)
    await test_completeness_agent()
    test_verdict_agent()
    await test_batch_evaluator()
    await test_orchestrator_integration()
    print("\n" + "=" * 80)
    print(" ALL MILESTONE 3 VALIDATION TESTS PASSED SUCCESSFULLY! ")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
