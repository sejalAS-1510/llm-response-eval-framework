"""
M3.1: Completeness Judge Agent.
Evaluates whether an AI-generated response sufficiently addresses all relevant aspects
and sub-questions contained within the prompt, identifying specific omissions with reasoning.
"""

import asyncio
from typing import Optional
from .base import GeminiClient
from .schemas import CompletenessResult

COMPLETENESS_SYSTEM_PROMPT = """You are an expert AI evaluator specializing in answer completeness, coverage analysis, and omission detection.
Your task is to evaluate whether an AI-generated response sufficiently addresses all relevant requirements, sub-questions, and expected aspects contained within the submitted question and reference context.

SCORING CRITERIA AND RUBRIC:
1. fully_complete (Score: 1.0)
   - Every requirement, explicit question, and implicit core need in the user query is thoroughly answered.
   - All necessary explanations, context, or expected details are provided without notable omissions.

2. mostly_complete (Score: 0.75 - 0.9)
   - The primary questions and major requirements are clearly addressed.
   - Only a minor detail, secondary sub-question, or non-critical nuance is omitted.

3. partially_complete (Score: 0.4 - 0.6)
   - Addresses some parts of the question, but leaves at least one major sub-question or key requirement unanswered.
   - Explanations are shallow, incomplete, or prematurely cut off.

4. incomplete (Score: 0.0 - 0.3)
   - Fails to answer the core question, omits the primary expected information, or provides only a trivial/evasive statement.

EVALUATION GUIDELINES:
- Decompose the user question (and reference answer/retrieved context if provided) into discrete requirements or sub-questions (`identified_requirements`).
- Compare the AI response against each requirement:
  - List fully covered requirements in `addressed_aspects`.
  - List requirements that are vaguely, superficially, or partially covered in `partially_addressed_aspects`.
  - List unaddressed, skipped, or omitted requirements in `missing_aspects`.
- When a reference answer is provided, use it to benchmark the expected breadth and depth of coverage.
- When no reference answer is provided, use the question's natural requirements and any retrieved knowledge base context.
- Provide comprehensive, constructive reasoning explaining the completeness score.
- Output MUST conform strictly to the required JSON schema.
"""


class CompletenessAgent:
    """
    Completeness Judge Agent assesses question coverage and omission depth.
    """
    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()

    async def evaluate(
        self,
        question: str,
        ai_response: str,
        context: Optional[str] = None,
    ) -> CompletenessResult:
        """
        Asynchronously evaluate the completeness of an AI response against the question and context.
        """
        context_block = f"""
Reference / Ground Truth Context:
\"\"\"{context}\"\"\"
""" if context else ""

        prompt = f"""EVALUATION TASK:
User Question:
\"\"\"{question}\"\"\"
{context_block}
AI Response to Evaluate:
\"\"\"{ai_response}\"\"\"


Assess the completeness of the AI response. Identify all expected requirements, catalog which are addressed, partially addressed, or missing, assign a score between 0.0 and 1.0, and explain your reasoning.
Return your evaluation strictly conforming to the requested JSON schema.
"""
        return await self.client.generate_structured(
            prompt=prompt,
            system_instruction=COMPLETENESS_SYSTEM_PROMPT,
            response_schema=CompletenessResult,
            temperature=0.0,
        )

    def evaluate_sync(
        self,
        question: str,
        ai_response: str,
        context: Optional[str] = None,
    ) -> CompletenessResult:
        """
        Synchronous convenience wrapper for completeness evaluation.
        """
        return asyncio.run(self.evaluate(question, ai_response, context))
