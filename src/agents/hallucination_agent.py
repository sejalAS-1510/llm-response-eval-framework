"""
M2.3: Hallucination Detection Agent.
Decomposes AI responses into atomic factual claims, cross-references each claim
against retrieved source chunks from the Reference Knowledge Base, and flags specific
unsupported or fabricated statements.
"""

import asyncio
from typing import Optional
from .base import GeminiClient
from .schemas import HallucinationResult

HALLUCINATION_SYSTEM_PROMPT = """You are an expert AI evaluator specializing in hallucination detection and claim-level verification (faithfulness / groundedness).

Your task is to detect whether an AI-generated response contains unsupported, fabricated, or contradicted statements when compared to the provided retrieved source context.

WORKFLOW:
1. Decompose the AI Response into individual, discrete atomic factual claims.
2. For each atomic claim:
   - Check if the claim is directly supported by the retrieved context.
   - Assign status:
     * "supported": The claim is explicitly backed by facts in the retrieved context.
     * "unsupported": The claim introduces names, numbers, dates, or assertions NOT mentioned or verifiable in the retrieved context.
     * "contradicted": The claim makes a statement that is in direct opposition to the retrieved context.
   - If supported or contradicted, quote the relevant context in `evidence`.
   - Provide a concise `explanation` for your determination.
3. Classify `is_hallucinated`: True if any claim is "unsupported" or "contradicted", False otherwise.
4. Calculate `hallucination_score`: (number of unsupported + contradicted claims) / (total claims). If total claims is 0, score is 0.0.
5. Populate `flagged_claims`: list containing ONLY the claims that are "unsupported" or "contradicted".
6. Populate `all_claims`: list of ALL decomposed claims.
7. Provide overall `reasoning` summarizing the groundedness of the response.

Return your evaluation strictly in the requested JSON structure.
"""


class HallucinationAgent:
    """
    Hallucination Detection Agent evaluates claim-level grounding against retrieved context.
    """
    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()

    async def evaluate(
        self,
        ai_response: str,
        context: str,
        question: Optional[str] = None,
    ) -> HallucinationResult:
        """
        Asynchronously perform claim decomposition and hallucination detection against context.
        """
        q_part = f"User Question:\n\"\"\"{question}\"\"\"\n\n" if question else ""
        prompt = f"""EVALUATION TASK:
{q_part}Retrieved Source Context:
\"\"\"{context}\"\"\"

AI Response to Verify:
\"\"\"{ai_response}\"\"\"

Perform claim decomposition on the AI response, cross-reference every claim against the retrieved source context, flag unsupported or contradicted assertions, and compute the hallucination metrics.
Return your evaluation strictly in the requested JSON structure.
"""
        return await self.client.generate_structured(
            prompt=prompt,
            system_instruction=HALLUCINATION_SYSTEM_PROMPT,
            response_schema=HallucinationResult,
            temperature=0.0,
        )

    def evaluate_sync(
        self,
        ai_response: str,
        context: str,
        question: Optional[str] = None,
    ) -> HallucinationResult:
        """
        Synchronous convenience wrapper for evaluation.
        """
        return asyncio.run(self.evaluate(ai_response, context, question))
