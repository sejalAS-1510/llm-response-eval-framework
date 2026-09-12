"""
M2.2: Accuracy Judge Agent.
Evaluates the factual correctness of an AI-generated response against a reference answer
or retrieved source chunks from the Reference Knowledge Base.
"""

import asyncio
from typing import Optional
from .base import GeminiClient
from .schemas import AccuracyResult

ACCURACY_SYSTEM_PROMPT = """You are an expert AI evaluator specializing in factual accuracy and ground-truth verification.
Your task is to evaluate the factual correctness of an AI-generated response using the provided reference answer or retrieved source context.

SCORING CRITERIA AND RUBRIC:
1. correct (Score: 1.0)
   - Every factual statement in the AI response is verified as true according to the reference context.
   - There are no factual errors, contradictions, or unsubstantiated falsehoods.

2. partially_correct (Score: 0.5)
   - The response contains some accurate facts, but also includes inaccuracies, omissions that distort the truth, or unverified claims.

3. incorrect (Score: 0.0)
   - The response provides factually false information or fails to provide the correct facts stated in the reference.

4. contradictory (Score: 0.0)
   - The response directly asserts statements that are explicitly contradicted by the reference context.

EVALUATION GUIDELINES:
- Ground your assessment strictly on the provided Reference / Ground Truth Context.
- Quote specific supporting evidence from the reference/context in the `supporting_evidence` list.
- Provide clear reasoning detailing which parts of the response are verified or contradicted.
- Return the output strictly conforming to the required JSON schema.
"""


class AccuracyAgent:
    """
    Accuracy Judge Agent verifies factual alignment with ground truth.
    """
    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()

    async def evaluate(
        self,
        question: str,
        ai_response: str,
        context: str,
    ) -> AccuracyResult:
        """
        Asynchronously evaluate the accuracy of an AI response against reference context.
        """
        prompt = f"""EVALUATION TASK:
User Question:
\"\"\"{question}\"\"\"

Reference / Ground Truth Context:
\"\"\"{context}\"\"\"

AI Response to Evaluate:
\"\"\"{ai_response}\"\"\"

Evaluate the factual accuracy of the AI response compared to the reference context. Extract supporting evidence and explain your reasoning.
Return your evaluation strictly in the requested JSON structure.
"""
        return await self.client.generate_structured(
            prompt=prompt,
            system_instruction=ACCURACY_SYSTEM_PROMPT,
            response_schema=AccuracyResult,
            temperature=0.0,
        )

    def evaluate_sync(
        self,
        question: str,
        ai_response: str,
        context: str,
    ) -> AccuracyResult:
        """
        Synchronous convenience wrapper for evaluation.
        """
        return asyncio.run(self.evaluate(question, ai_response, context))
