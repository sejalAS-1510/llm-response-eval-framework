"""
scratch/run_model_comparison.py
Creates two distinct AI response datasets (System A: Model Alpha vs System B: Model Beta)
covering correct, incomplete, inaccurate, and unsupported responses.
Evaluates both using BatchEvaluator, records all KPIs and dimensional distributions,
and prints a formatted comparison table for the final demo.
"""

import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.batch_evaluator import BatchEvaluator
from src.agents.orchestrator import EvaluationOrchestrator
from src.reporting.report_service import ReportDataAggregator
from src.input_module.main import BATCH_STORE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BATCHES_DIR = DATA_DIR / "batches"
BATCHES_DIR.mkdir(parents=True, exist_ok=True)

# 16 Benchmark Questions with ground truth references
EVAL_PAIRS = [
    {
        "id": 1,
        "question": "What is the speed of light in a vacuum?",
        "reference": "The speed of light in a vacuum is exactly 299,792,458 meters per second.",
        "resp_a": "The speed of light in a vacuum is approximately 299,792,458 meters per second.", # Correct
        "resp_b": "The speed of light in vacuum is 299,792,458 m/s.", # Correct
    },
    {
        "id": 2,
        "question": "What is the chemical formula for water and what atoms compose it?",
        "reference": "The chemical formula for water is H2O, composed of two hydrogen atoms and one oxygen atom.",
        "resp_a": "Water has the chemical formula H2O, consisting of two hydrogen atoms covalently bonded to one oxygen atom.", # Correct
        "resp_b": "Water is H2O.", # Incomplete (missing atoms breakdown)
    },
    {
        "id": 3,
        "question": "Explain the process of photosynthesis and its primary products.",
        "reference": "Photosynthesis is the biological process where green plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
        "resp_a": "Photosynthesis is the biological process where chlorophyll absorbs sunlight to convert water and carbon dioxide into glucose and oxygen.", # Correct
        "resp_b": "Photosynthesis converts light, CO2, and water into chemical energy in the form of glucose and oxygen.", # Correct
    },
    {
        "id": 4,
        "question": "What are the three branches of the US government and their primary constitutional functions?",
        "reference": "The three branches are Legislative (makes laws), Executive (enforces laws), and Judicial (interprets laws).",
        "resp_a": "The three branches of the US federal government are the Legislative, Executive, and Judicial branches.", # Incomplete (missing functions)
        "resp_b": "The US government has three branches: the Legislative makes federal laws, the Executive administers and enforces laws, and the Judicial interprets laws under the Constitution.", # Correct
    },
    {
        "id": 5,
        "question": "When was the United Nations founded and where is its international headquarters located?",
        "reference": "The United Nations was founded on October 24, 1945, and is headquartered in New York City.",
        "resp_a": "The United Nations was founded on October 24, 1945 following the conclusion of World War II.", # Incomplete (missing headquarters)
        "resp_b": "The UN is headquartered in New York City.", # Incomplete (missing founding date)
    },
    {
        "id": 6,
        "question": "When was Apple founded, who were the co-founders, and where was it established?",
        "reference": "Apple was founded on April 1, 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne in Los Altos, California.",
        "resp_a": "Apple was founded on April 1, 1976 by Steve Jobs, Steve Wozniak, and Elon Musk in Silicon Valley with initial venture funding of $2.5 million.", # Hallucinated (Musk + $2.5M funding)
        "resp_b": "Apple was founded on April 1, 1976 by Steve Jobs and Steve Wozniak.", # Incomplete (omits Wayne/location, but grounded)
    },
    {
        "id": 7,
        "question": "What was the Apollo 11 mission, what year did it launch, and who walked on the moon?",
        "reference": "Apollo 11 was the American spaceflight that landed astronauts Neil Armstrong and Buzz Aldrin on the Moon in July 1969.",
        "resp_a": "Apollo 11 was a NASA spaceflight launched in July 1969. Astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins all walked on the lunar surface during a 48-hour excursion.", # Hallucinated (Collins walked + 48hr excursion)
        "resp_b": "Apollo 11 was the 1969 American mission that successfully landed Neil Armstrong and Buzz Aldrin on the Moon, while Michael Collins orbited in the command module.", # Correct
    },
    {
        "id": 8,
        "question": "Who wrote Romeo and Juliet and what year was it first published in quarto?",
        "reference": "William Shakespeare wrote Romeo and Juliet, and it was first printed as an unauthorized quarto in 1597.",
        "resp_a": "William Shakespeare wrote Romeo and Juliet, and it was first printed in 1597 using an industrial steam-driven press invented by Johannes Gutenberg.", # Hallucinated (steam press in 1597)
        "resp_b": "William Shakespeare wrote Romeo and Juliet, and the tragedy was first published in quarto format in 1597.", # Correct
    },
    {
        "id": 9,
        "question": "What are the primary greenhouse gases in Earth's atmosphere?",
        "reference": "The primary greenhouse gases are water vapor, carbon dioxide (CO2), methane (CH4), nitrous oxide (N2O), and ozone.",
        "resp_a": "The primary greenhouse gases are carbon dioxide, methane, nitrous oxide, and krypton gas emitted by deep-sea volcanic thermal vents.", # Hallucinated (krypton gas)
        "resp_b": "The main greenhouse gases in Earth's atmosphere are water vapor, carbon dioxide, methane, nitrous oxide, and ozone.", # Correct
    },
    {
        "id": 10,
        "question": "Does sound travel faster in air or in water, and why?",
        "reference": "Sound travels faster in water (about 1,480 m/s) than in air (about 343 m/s) because water is much denser and less compressible.",
        "resp_a": "Sound travels significantly faster in air than in water because air is less dense, allowing sound waves to move with much less molecular resistance.", # Contradictory / Inaccurate
        "resp_b": "Sound travels about four times faster in water than in air because water molecules are more tightly packed and transmit elastic vibrations more rapidly.", # Correct
    },
    {
        "id": 11,
        "question": "What is the normal resting human body temperature in Celsius and Fahrenheit?",
        "reference": "Normal human body temperature is approximately 37 degrees Celsius or 98.6 degrees Fahrenheit.",
        "resp_a": "Normal human body temperature is approximately 45 degrees Celsius (113 degrees Fahrenheit).", # Contradictory / Inaccurate (lethal temp!)
        "resp_b": "Normal human body temperature is around 39.5 degrees Celsius.", # Inaccurate (fever temp)
    },
    {
        "id": 12,
        "question": "Does high inflation cause central banks to raise or lower interest rates?",
        "reference": "Central banks typically raise interest rates during high inflation to increase borrowing costs and curb demand.",
        "resp_a": "Central banks lower interest rates during high inflation to stimulate spending and provide liquidity to households.", # Contradictory / Inaccurate
        "resp_b": "Central banks adjust monetary policy and interest rates in response to inflation trends.", # Incomplete / vague (evasive)
    },
    {
        "id": 13,
        "question": "What are the key differences between SQL and NoSQL databases?",
        "reference": "SQL databases are relational, table-based, and use structured schemas with ACID guarantees. NoSQL databases are non-relational, document/key-value/graph based, and offer flexible schemas with horizontal scalability.",
        "resp_a": "SQL databases are relational and structured with rigid schemas and ACID transactions. NoSQL databases are non-relational with dynamic schemas, distributed architectures, and horizontal scaling.", # Correct
        "resp_b": "SQL is relational and NoSQL is non-relational.", # Incomplete (extremely brief)
    },
    {
        "id": 14,
        "question": "How does mRNA vaccine technology stimulate immunity against viruses?",
        "reference": "mRNA vaccines deliver genetic instructions for host cells to synthesize a harmless viral antigen (such as a spike protein), prompting an immune response and antibody production without using live virus.",
        "resp_a": "mRNA vaccines insert synthetic nanobots that directly enter the cell nucleus to alter human genomic DNA permanently, creating artificial antibodies that circulate for 50 years.", # Hallucinated (nanobots + genomic DNA)
        "resp_b": "mRNA vaccines introduce genetic messenger RNA that instructs cells to produce a harmless viral spike protein, triggering the immune system to produce protective antibodies.", # Correct
    },
    {
        "id": 15,
        "question": "Explain Newton's three laws of motion.",
        "reference": "Newton's laws: (1) Law of inertia: objects stay at rest or in uniform motion unless an external force acts; (2) F = ma: force equals mass times acceleration; (3) Action-reaction: for every action there is an equal and opposite reaction.",
        "resp_a": "Newton's three laws: 1. Inertia - an object remains at rest or in motion unless acted upon by a net force. 2. Force = mass × acceleration (F=ma). 3. Action and reaction - for every action, there is an equal and opposite reaction.", # Correct
        "resp_b": "Newton's first law states that an object at rest stays at rest unless acted upon by a force.", # Incomplete (omitted laws 2 and 3)
    },
    {
        "id": 16,
        "question": "Who was Alan Turing and what were his major contributions to computing?",
        "reference": "Alan Turing was an English mathematician and computer scientist who formalized the concepts of algorithms and Turing machines, broke the German Enigma cipher at Bletchley Park, and developed the Turing Test.",
        "resp_a": "Alan Turing was an English computer scientist who cracked the Enigma cipher and subsequently served as Prime Minister of Great Britain from 1951 to 1955.", # Hallucinated / Contradiction (Prime Minister)
        "resp_b": "Alan Turing was a British mathematician who helped crack the Enigma code during WWII, conceptualized the Turing machine, and pioneered artificial intelligence.", # Correct
    },
]


def write_csv(filepath: Path, model_key: str):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["questions", "ai_responses", "reference_answer"])
        for item in EVAL_PAIRS:
            writer.writerow([item["question"], item[model_key], item["reference"]])
    print(f"Wrote {len(EVAL_PAIRS)} records to {filepath}")


async def main():
    csv_a = DATA_DIR / "system_a_eval_set.csv"
    csv_b = DATA_DIR / "system_b_eval_set.csv"

    write_csv(csv_a, "resp_a")
    write_csv(csv_b, "resp_b")

    orchestrator = EvaluationOrchestrator()
    evaluator = BatchEvaluator(orchestrator=orchestrator, concurrency_limit=5)

    print("\n--- EVALUATING SYSTEM A (Model Alpha: Fluent/Generative) ---")
    records_a, _ = evaluator.parse_and_validate_csv(csv_a.read_text(encoding="utf-8"))
    summary_a = await evaluator.evaluate_batch(records_a, filename="system_a_eval_set.csv")
    summary_a.batch_id = "batch-system-a-eval"
    BATCH_STORE[summary_a.batch_id] = summary_a
    (BATCHES_DIR / f"{summary_a.batch_id}.json").write_text(summary_a.model_dump_json(indent=2), encoding="utf-8")

    print("\n--- EVALUATING SYSTEM B (Model Beta: Conservative/RAG) ---")
    records_b, _ = evaluator.parse_and_validate_csv(csv_b.read_text(encoding="utf-8"))
    summary_b = await evaluator.evaluate_batch(records_b, filename="system_b_eval_set.csv")
    summary_b.batch_id = "batch-system-b-eval"
    BATCH_STORE[summary_b.batch_id] = summary_b
    (BATCHES_DIR / f"{summary_b.batch_id}.json").write_text(summary_b.model_dump_json(indent=2), encoding="utf-8")

    rep_service = ReportDataAggregator(storage_dir=BATCHES_DIR)
    rep_a = rep_service.get_report_data_by_batch_id(summary_a.batch_id)
    rep_b = rep_service.get_report_data_by_batch_id(summary_b.batch_id)

    print("\n" + "=" * 90)
    print("SIDE-BY-SIDE EVALUATION COMPARISON: SYSTEM A vs SYSTEM B")
    print("=" * 90)
    print(f"{'Metric / Dimension':<35} | {'System A (Model Alpha)':<25} | {'System B (Model Beta)':<25}")
    print("-" * 90)
    print(f"{'Total Evaluated Records':<35} | {rep_a.metadata.total_records:<25} | {rep_b.metadata.total_records:<25}")
    print(f"{'Pass Count (%)':<35} | {f'{rep_a.overall_stats.pass_count} ({rep_a.overall_stats.pass_percent}%)':<25} | {f'{rep_b.overall_stats.pass_count} ({rep_b.overall_stats.pass_percent}%)':<25}")
    print(f"{'Needs Improvement Count (%)':<35} | {f'{rep_a.overall_stats.needs_improvement_count} ({rep_a.overall_stats.needs_improvement_percent}%)':<25} | {f'{rep_b.overall_stats.needs_improvement_count} ({rep_b.overall_stats.needs_improvement_percent}%)':<25}")
    print(f"{'Fail Count (%)':<35} | {f'{rep_a.overall_stats.fail_count} ({rep_a.overall_stats.fail_percent}%)':<25} | {f'{rep_b.overall_stats.fail_count} ({rep_b.overall_stats.fail_percent}%)':<25}")
    print(f"{'Hallucination Rate (%)':<35} | {f'{rep_a.hallucination_frequency.rate_percent}% ({rep_a.hallucination_frequency.flagged_responses_count} responses)':<25} | {f'{rep_b.hallucination_frequency.rate_percent}% ({rep_b.hallucination_frequency.flagged_responses_count} responses)':<25}")
    print("-" * 90)
    print(f"{'Average Relevance Score':<35} | {f'{rep_a.overall_stats.average_relevance_score:.3f} ({rep_a.overall_stats.average_relevance_percent}%)':<25} | {f'{rep_b.overall_stats.average_relevance_score:.3f} ({rep_b.overall_stats.average_relevance_percent}%)':<25}")
    print(f"{'Average Accuracy Score':<35} | {f'{rep_a.overall_stats.average_accuracy_score:.3f} ({rep_a.overall_stats.average_accuracy_percent}%)':<25} | {f'{rep_b.overall_stats.average_accuracy_score:.3f} ({rep_b.overall_stats.average_accuracy_percent}%)':<25}")
    print(f"{'Average Completeness Score':<35} | {f'{rep_a.overall_stats.average_completeness_score:.3f} ({rep_a.overall_stats.average_completeness_percent}%)':<25} | {f'{rep_b.overall_stats.average_completeness_score:.3f} ({rep_b.overall_stats.average_completeness_percent}%)':<25}")
    print(f"{'Average Groundedness Score':<35} | {f'{rep_a.overall_stats.average_groundedness_score:.3f} ({rep_a.overall_stats.average_groundedness_percent}%)':<25} | {f'{rep_b.overall_stats.average_groundedness_score:.3f} ({rep_b.overall_stats.average_groundedness_percent}%)':<25}")
    print(f"{'Average Weighted Composite Score':<35} | {f'{rep_a.overall_stats.average_weighted_score:.3f} ({rep_a.overall_stats.average_weighted_percent}%)':<25} | {f'{rep_b.overall_stats.average_weighted_score:.3f} ({rep_b.overall_stats.average_weighted_percent}%)':<25}")
    print("=" * 90)

    print("\nSYSTEM A RECOMMENDATIONS:")
    for r in rep_a.recommendations:
        print(f"- [{r.priority.upper()}] {r.headline}: {r.plain_english}")

    print("\nSYSTEM B RECOMMENDATIONS:")
    for r in rep_b.recommendations:
        print(f"- [{r.priority.upper()}] {r.headline}: {r.plain_english}")

if __name__ == "__main__":
    asyncio.run(main())
