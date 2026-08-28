# Research notes (M1.1)

Notes from reading up on how LLM outputs get evaluated, before designing this system.

## How people evaluate LLM responses

Most evaluation setups don't try to produce one "quality" score. They break it into a few separate checks:

- **Relevance** – does the response actually answer what was asked
- **Accuracy** – are the facts in it correct
- **Faithfulness / groundedness** – are the claims backed by the given context (separate from whether they're *true* — a model can be right by accident, from its own memory, without the answer being grounded in the source)
- **Completeness** – does it cover what the question/reference expects
- **Hallucination** – basically the flip side of faithfulness: which claims aren't backed by anything

Two ways people score these automatically:
- Old-school **reference-based metrics** (BLEU/ROUGE, embedding similarity) — cheap, but they just check word/embedding overlap, so a correct paraphrase can score low and a fluent wrong answer can score high.
- **LLM-as-a-judge** — prompt an LLM with the question, response, and context, and have it score against a rubric. Both RAGAS and TruLens lean on this for anything that needs real understanding, and it's what I'm using for the judge agents here.

## Hallucination detection

The common approach is **claim decomposition** — split the response into individual factual claims and check each one against the retrieved context, instead of judging the whole response as one blob. You get a supported/unsupported fraction *and* a list of exactly which claims failed, which is a lot more useful than a single pass/fail number. That's why the Hallucination agent in this project is scoped to return both a score and the flagged claims.

## RAG basics

RAG has two halves and they fail independently, so it makes sense to evaluate them separately:
- **Retriever** — documents get chunked, each chunk embedded, and stored in a vector index. At query time the question gets embedded the same way and the nearest chunks come back by similarity. Chunk size/overlap and embedding model choice matter a lot here — too big and you pull in noise, too small and you lose context.
- **Generator** — the LLM writes an answer using the retrieved chunks. If this part fails you get low faithfulness/groundedness even when retrieval was fine.

Because of this split, most RAG eval frameworks report retrieval metrics and generation metrics separately, not one blended number — otherwise you can't tell which half is broken.

## RAGAS

Reference-free evaluation library built for RAG. Four main metrics, and they map cleanly onto retriever vs. generator:

| Metric | What it checks | Half |
|---|---|---|
| Faithfulness | fraction of answer claims supported by retrieved context | generator |
| Answer relevancy | how close the answer is to the question, semantically | generator |
| Context precision | are the relevant chunks ranked above irrelevant ones | retriever |
| Context recall | how much of the ground-truth answer is covered by retrieved context | retriever |

Faithfulness uses claim decomposition, same idea as above.

## TruLens — RAG Triad

Three LLM-judged checks meant to certify a RAG app end to end:
- **Context relevance** — is what got retrieved actually relevant to the query
- **Groundedness** — is the answer supported by the retrieved context (their name for faithfulness)
- **Answer relevance** — does the answer address the original question

The useful idea from TruLens isn't the three metrics individually, it's that they cover every *edge* of the pipeline (query→context, context→answer, query→answer) instead of just judging the final answer. That's basically the reasoning behind having separate agents here — Relevance and Accuracy cover query↔answer, Hallucination covers context↔answer, and the retrieval checks in M1.4 cover query↔context.

## What I took from this for the design

RAGAS and TruLens land on the same underlying pattern: LLM-as-a-judge, claim-level faithfulness, and a handful of mostly independent dimensions rather than one aggregate score. The four-agent setup here (Relevance, Accuracy, Hallucination, Completeness + a Verdict agent to combine them) follows that pattern, with Completeness added since it's called out explicitly in the assignment and isn't one of RAGAS/TruLens' core metrics.

## Benchmark datasets for the reference KB

- **TruthfulQA** — short question + "best" answer pairs, no long passage. Useful as ground truth for factuality checks, not for testing retrieval over long context.
- **SQuAD** — question, answer span, and a full supporting paragraph. This is what I'm using to actually test chunking and retrieval quality, since it has real context to search over and a known-correct answer to check against.

## Where this leaves the M1 build

- Score everything on a 0–1 scale (not 1–5) so aggregation in the Verdict agent doesn't need a rescale step later
- Each dimension is a separate, independent judge — easier to debug and improve one at a time in M2
- Hallucination agent returns flagged claims, not just a score
- Knowledge base is seeded from both SQuAD (long-context retrieval) and TruthfulQA (short factual QA) since they stress different parts of the pipeline
- Kept M1 fully local/free (SQLite, ChromaDB, sentence-transformers) — no API key needed yet. The Claude API is reserved for the actual judge agents in M2, where an LLM judge is required.
