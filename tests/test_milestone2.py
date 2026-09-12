"""
Milestone 2.4: Agent Evaluation & Consistency Validation Suite.

Validates:
1. Relevance Judge Agent scoring and reasoning quality.
2. Accuracy Judge Agent factual verification and supporting evidence.
3. Hallucination Detection Agent claim decomposition, flagging, and explanations.
4. Scoring consistency, false positives, false negatives, and metrics.

Usage:
    python -m tests.test_milestone2
"""

import asyncio
import json
import logging
from pathlib import Path

from src.agents.orchestrator import EvaluationOrchestrator

logging.basicConfig(level=logging.WARNING)

BENCHMARK_FILE = Path(__file__).resolve().parent / "benchmark_test_cases.json"


async def run_validation():
    print("=" * 80)
    print(" MILESTONE 2: AGENT EVALUATION & CONSISTENCY VALIDATION (M2.4)")
    print("=" * 80)

    if not BENCHMARK_FILE.exists():
        print(f"Error: Benchmark file not found at {BENCHMARK_FILE}")
        return

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    orchestrator = EvaluationOrchestrator()

    total = len(test_cases)
    relevance_passes = 0
    accuracy_passes = 0
    hallucination_tp = 0
    hallucination_fp = 0
    hallucination_tn = 0
    hallucination_fn = 0

    print(f"Loaded {total} benchmark test cases across varied categories.\n")

    for tc in test_cases:
        tc_id = tc["id"]
        category = tc["category"]
        desc = tc["description"]
        expected = tc.get("expected", {})

        print(f"[{tc_id}] {category.upper()} - {desc}")
        print(f"  Q: {tc['question']}")
        print(f"  A: {tc['ai_response']}")

        # Run evaluation via orchestrator
        result = await orchestrator.evaluate(
            question=tc["question"],
            ai_response=tc["ai_response"],
            reference_answer=tc.get("reference_answer"),
            source_document=tc.get("source_document"),
        )

        rel = result.relevance
        acc = result.accuracy
        hal = result.hallucination

        # --- Relevance check ---
        rel_ok = True
        if "min_relevance_score" in expected and rel.score < expected["min_relevance_score"]:
            rel_ok = False
        if "max_relevance_score" in expected and rel.score > expected["max_relevance_score"]:
            rel_ok = False
        if rel_ok:
            relevance_passes += 1

        # --- Accuracy check ---
        acc_ok = True
        if "min_accuracy_score" in expected and acc.score < expected["min_accuracy_score"]:
            acc_ok = False
        if "max_accuracy_score" in expected and acc.score > expected["max_accuracy_score"]:
            acc_ok = False
        if acc_ok:
            accuracy_passes += 1

        # --- Hallucination check ---
        hal_expected = expected.get("is_hallucinated")
        if hal_expected is not None:
            if hal_expected and hal.is_hallucinated:
                hallucination_tp += 1
            elif not hal_expected and not hal.is_hallucinated:
                hallucination_tn += 1
            elif not hal_expected and hal.is_hallucinated:
                hallucination_fp += 1
            elif hal_expected and not hal.is_hallucinated:
                hallucination_fn += 1

        print(f"  -> Relevance: score={rel.score:.2f} ({rel.classification}) [Pass: {rel_ok}]")
        print(f"     Reasoning: {rel.reasoning}")
        print(f"  -> Accuracy: score={acc.score:.2f} ({acc.classification}) [Pass: {acc_ok}]")
        print(f"     Evidence : {acc.supporting_evidence[:2]}")
        print(f"     Reasoning: {acc.reasoning}")
        print(f"  -> Hallucination: is_hallucinated={hal.is_hallucinated}, score={hal.hallucination_score:.2f}")
        if hal.flagged_claims:
            print(f"     Flagged Claims ({len(hal.flagged_claims)}):")
            for c in hal.flagged_claims:
                print(f"       * [{c.status}] {c.claim}")
                print(f"         Why: {c.explanation}")
        print(f"     Reasoning: {hal.reasoning}")
        print("-" * 80)
        await asyncio.sleep(5)

    # --- Summary Metrics ---
    total_evals = total
    precision = (
        hallucination_tp / (hallucination_tp + hallucination_fp)
        if (hallucination_tp + hallucination_fp) > 0
        else 1.0
    )
    recall = (
        hallucination_tp / (hallucination_tp + hallucination_fn)
        if (hallucination_tp + hallucination_fn) > 0
        else 1.0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 1.0
    )

    print("\n" + "=" * 80)
    print(" VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Test Cases Evaluated : {total_evals}")
    print(f"Relevance Consistency Rate : {relevance_passes}/{total_evals} ({relevance_passes/total_evals*100:.1f}%)")
    print(f"Accuracy Consistency Rate  : {accuracy_passes}/{total_evals} ({accuracy_passes/total_evals*100:.1f}%)")
    print(f"Hallucination Detection    : TP={hallucination_tp}, TN={hallucination_tn}, FP={hallucination_fp}, FN={hallucination_fn}")
    print(f"  - Precision              : {precision:.2f}")
    print(f"  - Recall                 : {recall:.2f}")
    print(f"  - F1 Score               : {f1:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_validation())
