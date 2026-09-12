"""
Evaluation Orchestrator for Milestone 2.
Coordinates context retrieval (RAG fallback) and runs Relevance, Accuracy, and Hallucination
judge agents concurrently.
"""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Optional

from .schemas import EvaluationResult
from .relevance_agent import RelevanceAgent
from .accuracy_agent import AccuracyAgent
from .hallucination_agent import HallucinationAgent
from ..knowledge_base.vector_store import retrieve

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """
    Orchestrates the evaluation of an AI response across all three judge agents.
    """
    def __init__(
        self,
        relevance_agent: Optional[RelevanceAgent] = None,
        accuracy_agent: Optional[AccuracyAgent] = None,
        hallucination_agent: Optional[HallucinationAgent] = None,
    ):
        self.relevance_agent = relevance_agent or RelevanceAgent()
        self.accuracy_agent = accuracy_agent or AccuracyAgent()
        self.hallucination_agent = hallucination_agent or HallucinationAgent()

    def resolve_context(
        self,
        question: str,
        reference_answer: Optional[str] = None,
        source_document: Optional[str] = None,
        top_k: int = 3,
    ) -> str:
        """
        Determines the ground truth context. Uses explicit reference/source documents
        if provided; otherwise queries the Reference Knowledge Base (ChromaDB) via RAG.
        """
        context_parts = []

        if reference_answer and reference_answer.strip():
            context_parts.append(f"Direct Reference Answer:\n{reference_answer.strip()}")

        if source_document and source_document.strip():
            context_parts.append(f"Supplied Source Document:\n{source_document.strip()}")

        # If no reference or source was supplied, use RAG retrieval
        if not context_parts:
            try:
                hits = retrieve(question, top_k=top_k)
                if hits:
                    rag_snippets = []
                    for h in hits:
                        source = h.get("metadata", {}).get("source", "unknown")
                        text = h.get("text", "")
                        rag_snippets.append(f"[Source: {source}]\n{text}")
                    context_parts.append("Retrieved Reference Knowledge Base Chunks:\n" + "\n\n".join(rag_snippets))
                else:
                    context_parts.append("No reference answer provided and no relevant knowledge base chunks found.")
            except Exception as e:
                logger.warning(f"RAG retrieval failed or vector store not initialized: {e}")
                context_parts.append("Retrieval unavailable. Evaluate with zero-shot reference context.")

        return "\n\n".join(context_parts)

    async def evaluate(
        self,
        question: str,
        ai_response: str,
        reference_answer: Optional[str] = None,
        source_document: Optional[str] = None,
        submission_id: Optional[int] = None,
        top_k: int = 3,
    ) -> EvaluationResult:
        """
        Runs Relevance, Accuracy, and Hallucination agents concurrently.
        """
        # 1. Resolve context
        context_str = self.resolve_context(
            question=question,
            reference_answer=reference_answer,
            source_document=source_document,
            top_k=top_k,
        )

        # 2. Execute agents sequentially to respect API rate limits smoothly
        relevance_res = await self.relevance_agent.evaluate(
            question=question,
            ai_response=ai_response,
        )
        accuracy_res = await self.accuracy_agent.evaluate(
            question=question,
            ai_response=ai_response,
            context=context_str,
        )
        hallucination_res = await self.hallucination_agent.evaluate(
            ai_response=ai_response,
            context=context_str,
            question=question,
        )

        return EvaluationResult(
            submission_id=submission_id,
            question=question,
            ai_response=ai_response,
            context_used=context_str,
            relevance=relevance_res,
            accuracy=accuracy_res,
            hallucination=hallucination_res,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )

    def evaluate_sync(
        self,
        question: str,
        ai_response: str,
        reference_answer: Optional[str] = None,
        source_document: Optional[str] = None,
        submission_id: Optional[int] = None,
        top_k: int = 3,
    ) -> EvaluationResult:
        """
        Synchronous wrapper for evaluate.
        """
        return asyncio.run(
            self.evaluate(
                question=question,
                ai_response=ai_response,
                reference_answer=reference_answer,
                source_document=source_document,
                submission_id=submission_id,
                top_k=top_k,
            )
        )
