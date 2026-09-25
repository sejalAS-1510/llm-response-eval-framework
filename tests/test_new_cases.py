"""
Comprehensive validation script for New Test Cases across diverse domains.
Tests:
1. High-quality complete scientific explanation (Pass)
2. Multi-part history question with omitted aspects (Needs Improvement)
3. Subtle hallucination with fabricated entities (Needs Improvement)
4. Direct scientific contradiction (Fail)
5. Completely off-topic irrelevant prompt (Fail)
6. Numerical contradiction (Needs Improvement / Fail)
7. Automatic RAG ChromaDB fallback without reference answer (Pass)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from src.agents.orchestrator import EvaluationOrchestrator

test_cases = [
    {
        "id": "NC-01",
        "domain": "Biomedical / Immunology",
        "type": "High Quality Grounded Response",
        "question": "How does mRNA vaccination work, and what is its role in producing immunity?",
        "ai_response": "mRNA vaccines provide instructions to our cells to produce a harmless viral protein, known as the spike protein. The immune system recognizes this protein as foreign and produces antibodies and T-cell responses against it, creating immunity without exposing the body to the live virus.",
        "reference_answer": "mRNA vaccines instruct cells to make a harmless spike protein, triggering an immune response that produces antibodies to fight future infection.",
        "expected_verdict": "Pass",
    },
    {
        "id": "NC-02",
        "domain": "History / Science Policy",
        "type": "Multi-part Question with Omission",
        "question": "What was the Manhattan Project, when did it begin, who led the scientific laboratory, and what was the code name of the first nuclear test?",
        "ai_response": "The Manhattan Project was the American-led research program during World War II that developed the first nuclear weapons. The scientific laboratory at Los Alamos was directed by J. Robert Oppenheimer.",
        "reference_answer": "The Manhattan Project was a WWII research program begun in 1942, scientifically directed by J. Robert Oppenheimer, culminating in the first nuclear detonation code-named Trinity.",
        "expected_verdict": "Needs Improvement",
    },
    {
        "id": "NC-03",
        "domain": "Space Exploration",
        "type": "Subtle Hallucination (Invented Entity)",
        "question": "What are the planetary rovers operated by NASA on Mars, and which is the newest?",
        "ai_response": "NASA has operated several rovers on Mars including Sojourner, Spirit, Opportunity, Curiosity, and Perseverance. Perseverance is the newest rover, which landed with a miniature robotic explorer named Ares-X.",
        "reference_answer": "NASA Mars rovers are Sojourner, Spirit, Opportunity, Curiosity, and Perseverance. Perseverance landed in 2021 as the latest rover, carrying the Ingenuity helicopter.",
        "expected_verdict": "Needs Improvement",
    },
    {
        "id": "NC-04",
        "domain": "Physics / Acoustics",
        "type": "Direct Factual Contradiction",
        "question": "Is the speed of sound faster in air than in water?",
        "ai_response": "Yes, sound travels significantly faster in air than in water because air is less dense and offers less molecular resistance to sonic waves.",
        "reference_answer": "No, sound travels much faster in water (about 1,480 m/s) than in air (about 343 m/s) because water is denser and less compressible.",
        "expected_verdict": "Fail",
    },
    {
        "id": "NC-05",
        "domain": "Computer Science / Food",
        "type": "Completely Off-Topic / Irrelevant",
        "question": "What are the primary differences between SQL and NoSQL databases?",
        "ai_response": "To make a classic French omelet, whisk two eggs with salt, melt butter in a non-stick pan over medium heat, and roll the eggs gently when set.",
        "reference_answer": "SQL databases are relational and table-based with fixed schemas, while NoSQL databases are non-relational, distributed, and support flexible document or key-value structures.",
        "expected_verdict": "Fail",
    },
    {
        "id": "NC-06",
        "domain": "Medicine / Physiology",
        "type": "Numerical Contradiction",
        "question": "What is the normal human body temperature in degrees Celsius?",
        "ai_response": "Normal human body temperature is approximately 45 degrees Celsius under standard resting conditions.",
        "reference_answer": "Normal human body temperature is approximately 37 degrees Celsius (98.6 degrees Fahrenheit).",
        "expected_verdict": "Needs Improvement",
    },
    {
        "id": "NC-07",
        "domain": "Geography / General Knowledge",
        "type": "RAG ChromaDB Knowledge Base Fallback",
        "question": "When was the College of Engineering at the University of Notre Dame established?",
        "ai_response": "The College of Engineering at the University of Notre Dame was established in 1920.",
        "reference_answer": None,
        "expected_verdict": "Pass",
    },
]


async def run_new_tests():
    orch = EvaluationOrchestrator()
    print("=" * 95)
    print(" EVALUATION SUITE FOR NEW BENCHMARK TEST CASES")
    print("=" * 95)

    all_matched = True

    for tc in test_cases:
        res = await orch.evaluate(
            question=tc["question"],
            ai_response=tc["ai_response"],
            reference_answer=tc["reference_answer"],
        )
        v = res.verdict.verdict
        score = res.verdict.weighted_score
        matches = v == tc["expected_verdict"]
        if not matches:
            all_matched = False

        status_flag = "[MATCH]" if matches else f"[MISMATCH -> Expected: {tc['expected_verdict']}]"
        groundedness = round(1.0 - res.hallucination.hallucination_score, 2)

        print(f"\n[{tc['id']}] {tc['domain']} | {tc['type']}")
        print(f"  Query    : {tc['question']}")
        print(f"  AI Resp  : {tc['ai_response'][:100]}...")
        print(f"  Result   : Verdict = {v:<18} | Weighted Score = {score * 100:.1f}% | {status_flag}")
        print(f"  Scores   : Relevance: {res.relevance.score:.2f} | Accuracy: {res.accuracy.score:.2f} ({res.accuracy.classification}) | Groundedness: {groundedness:.2f} | Completeness: {res.completeness.score:.2f} ({res.completeness.classification})")
        if res.verdict.major_issues:
            print(f"  Issues   : {res.verdict.major_issues}")
        if res.verdict.strengths:
            print(f"  Strengths: {res.verdict.strengths}")
        print("-" * 95)

    print("\n" + "=" * 95)
    if all_matched:
        print(" SUCCESS: ALL NEW TEST CASES EVALUATED ACCURATELY AGAINST EXPECTED CRITERIA!")
    else:
        print(" REVIEW NEEDED: One or more test cases had unexpected verdict outputs.")
    print("=" * 95)


if __name__ == "__main__":
    asyncio.run(run_new_tests())
