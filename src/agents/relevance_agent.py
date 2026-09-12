"""
M2.1: Relevance Judge Agent.
Evaluates whether an AI-generated response directly and appropriately addresses the submitted question.
"""

import asyncio
from typing import Optional
from .base import GeminiClient
from .schemas import RelevanceResult

RELEVANCE_SYSTEM_PROMPT = """You are an expert AI evaluator specializing in response relevance and intent fulfillment.
Your role is to evaluate whether an AI-generated response directly and appropriately addresses the user's submitted question.

SCORING CRITERIA AND RUBRIC:
1. fully_relevant (Score: 1.0)
   - The response directly and completely addresses the specific question asked.
   - It answers the core intent without evading or drifting into irrelevant topics.

2. partially_relevant (Score: 0.5)
   - The response addresses only part of the question while ignoring critical components.
   - Or it provides mostly peripheral information with only a brief/weak reference to the core question.

3. unrelated (Score: 0.0)
   - The response answers a completely different question or discusses an unrelated domain.

4. off_topic (Score: 0.0)
   - The response rambles, evades the question, or provides generic filler without addressing the subject asked.

INSTRUCTIONS:
- Do NOT judge factual accuracy or truthfulness here; focus purely on whether the response addresses the QUESTION asked.
- Provide a clear, objective reasoning explanation justifying your score and classification.
- Output your result strictly conforming to the required JSON schema.
"""


class RelevanceAgent:
    """
    Relevance Judge Agent evaluates how well the response targets the question.
    """
    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()

    async def evaluate(self, question: str, ai_response: str) -> RelevanceResult:
        """
        Asynchronously evaluate the relevance of an AI response to the question.
        """
        prompt = f"""EVALUATION TASK:
User Question:
\"\"\"{question}\"\"\"

AI Response to Evaluate:
\"\"\"{ai_response}\"\"\"

Analyze how directly the AI response answers the user question. Return your evaluation strictly in the requested JSON structure.
"""
        return await self.client.generate_structured(
            prompt=prompt,
            system_instruction=RELEVANCE_SYSTEM_PROMPT,
            response_schema=RelevanceResult,
            temperature=0.0,
        )

    def evaluate_sync(self, question: str, ai_response: str) -> RelevanceResult:
        """
        Synchronous convenience wrapper for evaluation.
        """
        return asyncio.run(self.evaluate(question, ai_response))
