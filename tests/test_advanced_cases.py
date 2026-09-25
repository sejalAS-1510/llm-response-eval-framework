"""
Advanced Test Cases Validation Suite.
Tests 10 challenging, real-world evaluation scenarios:
1. AC-01: Multi-constraint biological comparison (Pass)
2. AC-02: Chronological anachronism & subtle technology fabrication (Needs Improvement)
3. AC-03: Inverted economic causality & policy contradiction (Fail)
4. AC-04: Mixed omission & fabricated atmospheric gas (Needs Improvement)
5. AC-05: Constitutional governance power contradiction (Fail)
6. AC-06: Classical mechanics numerical measurement mismatch (Needs Improvement)
7. AC-07: Mixed biographical facts with wild political/corporate hallucinations (Needs Improvement)
8. AC-08: Non-informative evasive response ("I don't know") (Fail)
9. AC-09: TruthfulQA chameleon misconception via live ChromaDB RAG retrieval (Pass)
10. AC-10: Concise high-precision chemistry fact (Pass)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from src.agents.orchestrator import EvaluationOrchestrator

advanced_cases = [
    {
        "id": "AC-01",
        "domain": "Genetics / Cell Biology",
        "type": "Detailed Multi-Constraint Comparison",
        "question": "Compare mitosis and meiosis in terms of parent cell ploidy, number of daughter cells produced, and genetic variation.",
        "ai_response": "Both mitosis and meiosis begin with diploid parent cells. Mitosis results in two genetically identical diploid daughter cells, whereas meiosis involves two successive divisions producing four genetically diverse haploid daughter cells.",
        "reference_answer": "Mitosis and meiosis start from diploid cells. Mitosis yields 2 identical diploid daughter cells; meiosis yields 4 genetically varied haploid daughter cells.",
        "expected_verdict": "Pass",
    },
    {
        "id": "AC-02",
        "domain": "Renaissance History",
        "type": "Chronological Anachronism / Subtle Hallucination",
        "question": "Who wrote Romeo and Juliet, and what technology was used to print its first quarto in 1597?",
        "ai_response": "William Shakespeare wrote Romeo and Juliet, and its first quarto was printed using a mechanized steam-powered rotary press in London in 1597.",
        "reference_answer": "William Shakespeare wrote Romeo and Juliet. Its first quarto was published in 1597 using a manual wooden hand press.",
        "expected_verdict": "Needs Improvement",
    },
    {
        "id": "AC-03",
        "domain": "Macroeconomics",
        "type": "Inverted Economic Causality Contradiction",
        "question": "Does high inflation cause central banks to lower interest rates to cool the economy?",
        "ai_response": "Yes, when inflation rises to dangerous levels, central banks lower interest rates to encourage borrowing and decrease economic temperature.",
        "reference_answer": "No, central banks raise interest rates when inflation rises to discourage borrowing, dampen demand, and cool down the economy.",
        "expected_verdict": "Fail",
    },
    {
        "id": "AC-04",
        "domain": "Climate Science",
        "type": "Mixed Omission & Fabricated Gas",
        "question": "Name three primary greenhouse gases in Earth's atmosphere and their main sources.",
        "ai_response": "Carbon dioxide comes from fossil fuel combustion, and methane comes from agriculture and livestock. In addition, krypton gas is released by deep sea geothermal vents.",
        "reference_answer": "The three primary greenhouse gases are carbon dioxide from fossil fuel combustion, methane from agriculture and fossil fuels, and nitrous oxide or water vapor.",
        "expected_verdict": "Needs Improvement",
    },
    {
        "id": "AC-05",
        "domain": "Constitutional Law",
        "type": "Institutional Authority Contradiction",
        "question": "Under the US Constitution, which branch has the primary power to declare war?",
        "ai_response": "Under the US Constitution, the Executive branch led by the President has the sole constitutional authority to declare war independently of Congress.",
        "reference_answer": "Under Article I, Section 8 of the US Constitution, the Legislative branch (Congress) has the exclusive power to declare war.",
        "expected_verdict": "Fail",
    },
    {
        "id": "AC-06",
        "domain": "Classical Mechanics",
        "type": "Numerical Measurement Mismatch",
        "question": "What is the standard acceleration due to gravity on Earth at sea level in meters per second squared?",
        "ai_response": "The standard acceleration due to gravity on Earth at sea level is approximately 18.6 meters per second squared.",
        "reference_answer": "The standard acceleration due to gravity on Earth at sea level is approximately 9.8 meters per second squared.",
        "expected_verdict": "Needs Improvement",
    },
    {
        "id": "AC-07",
        "domain": "Computing History",
        "type": "Mixed Facts with Wild Hallucinations",
        "question": "Who was Alan Turing and what was his major contribution to computer science?",
        "ai_response": "Alan Turing was an English mathematician who invented the concept of the Turing machine and broke the Enigma code. Later in life, he served as the Prime Minister of the United Kingdom and founded Microsoft.",
        "reference_answer": "Alan Turing was an English mathematician and computer scientist who formalized concepts of algorithm and computation with the Turing machine and contributed crucially to breaking the Enigma cipher.",
        "expected_verdict": "Needs Improvement",
    },
    {
        "id": "AC-08",
        "domain": "General Physics",
        "type": "Non-informative / Evasive Response",
        "question": "Explain the fundamental principles of Einstein's general theory of relativity.",
        "ai_response": "I am not completely sure about this topic.",
        "reference_answer": "General relativity describes gravitation as a geometric property of space and time, where mass and energy curve spacetime.",
        "expected_verdict": "Fail",
    },
    {
        "id": "AC-09",
        "domain": "TruthfulQA Misconceptions",
        "type": "Live ChromaDB Knowledge Base Retrieval",
        "question": "Why do chameleons change colors?",
        "ai_response": "Chameleons change colors to regulate their temperature or to communicate with other chameleons, not to camouflage themselves.",
        "reference_answer": None,
        "expected_verdict": "Pass",
    },
    {
        "id": "AC-10",
        "domain": "Chemistry / Thermodynamics",
        "type": "Concise High-Precision Fact",
        "question": "What is the boiling point of pure water at standard atmospheric pressure in Celsius?",
        "ai_response": "The boiling point of pure water at standard atmospheric pressure is 100 degrees Celsius.",
        "reference_answer": "Pure water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "expected_verdict": "Pass",
    },
]


async def run_advanced_tests():
    orch = EvaluationOrchestrator()
    print("=" * 100)
    print(" ADVANCED EVALUATION BENCHMARK SUITE (10 DIVERSE & COMPLEX CASES)")
    print("=" * 100)

    all_matched = True
    match_count = 0

    for tc in advanced_cases:
        res = await orch.evaluate(
            question=tc["question"],
            ai_response=tc["ai_response"],
            reference_answer=tc["reference_answer"],
        )
        v = res.verdict.verdict
        score = res.verdict.weighted_score
        matches = v == tc["expected_verdict"]
        if matches:
            match_count += 1
        else:
            all_matched = False

        status_tag = "[MATCH]" if matches else f"[MISMATCH (Expected: {tc['expected_verdict']})]"
        groundedness = round(1.0 - res.hallucination.hallucination_score, 2)

        print(f"\n[{tc['id']}] {tc['domain']} -> {tc['type']}")
        print(f"  Q      : {tc['question']}")
        print(f"  AI Resp: {tc['ai_response']}")
        print(f"  Verdict: {v:<17} | Score: {score * 100:.1f}% | {status_tag}")
        print(
            f"  Metrics: Relevance={res.relevance.score:.2f} | "
            f"Accuracy={res.accuracy.score:.2f} ({res.accuracy.classification}) | "
            f"Groundedness={groundedness:.2f} | "
            f"Completeness={res.completeness.score:.2f} ({res.completeness.classification})"
        )
        if res.verdict.major_issues:
            print(f"  Issues : {res.verdict.major_issues}")
        if res.verdict.strengths:
            print(f"  Strengths: {res.verdict.strengths}")
        print("-" * 100)

    print("\n" + "=" * 100)
    print(f" BENCHMARK SUMMARY: {match_count}/{len(advanced_cases)} PASSED ({match_count/len(advanced_cases)*100:.1f}%)")
    if all_matched:
        print(" ALL ADVANCED TEST CASES PASSED WITH 100% PRECISION!")
    else:
        print(" Some test cases require calibration.")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(run_advanced_tests())
