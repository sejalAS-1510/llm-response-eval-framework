# Tech stack

## Milestone 1 (built)

| Layer | Tool | Why |
|---|---|---|
| API | FastAPI | quick to set up, built-in request validation |
| Input validation | Pydantic | pairs naturally with FastAPI, catches bad input early |
| Submission storage | SQLite | no separate server needed, fine for local dev |
| Chunking | LangChain text splitter | standard, handles overlap/chunk size out of the box |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | runs locally, no API key needed |
| Vector store | ChromaDB (embedded) | simplest local vector DB, no separate service |
| Benchmark data | TruthfulQA, SQuAD (HuggingFace `datasets`) | public, well known, cover both short-QA and long-context cases |

## Milestone 2 (planned, not built yet)

| Layer | Tool | Why |
|---|---|---|
| LLM-as-a-judge | Claude API (Anthropic) | needed for the actual relevance/accuracy/hallucination/completeness scoring |
| Orchestration | plain Python (async) to start | keep it simple until there's a reason for a framework |
| Results | JSON, later a small dashboard | structured output first, UI later |

Everything in M1 runs locally with no paid API keys required — that was a deliberate choice so the retrieval/knowledge-base part could be built and tested independently before wiring in the Claude API for the judge agents.
