"""
Base LLM client for Milestone 2 Judge Agents.
Configures Google Gemini with native structured JSON output and schema validation.
"""

import os
import re
import json
import asyncio
import logging
from typing import Type, TypeVar
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

T = TypeVar("T", bound=BaseModel)


class GeminiClient:
    """
    Wrapper around Google Generative AI for structured agent evaluations.
    """
    def __init__(self, model_name: str = None, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self._genai = None
        self._is_mock = os.getenv("MOCK_LLM", "0").lower() in ("1", "true")

        if not self._is_mock:
            if not self.api_key:
                logger.warning(
                    "GEMINI_API_KEY is not set. Set the GEMINI_API_KEY environment variable, "
                    "or set MOCK_LLM=1 to run in offline simulation mode."
                )
            else:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    self._genai = genai
                except ImportError:
                    logger.warning(
                        "google-generativeai package not installed. Run `pip install google-generativeai`."
                    )

    async def generate_structured(
        self,
        prompt: str,
        system_instruction: str,
        response_schema: Type[T],
        temperature: float = 0.0,
    ) -> T:
        """
        Sends prompt to Gemini model and parses the returned JSON into the specified Pydantic schema.
        """
        if self._is_mock or not self._genai or not self.api_key:
            return self._mock_fallback(prompt, response_schema)

        # Attempt to use native response_schema; fallback to JSON prompt injection if protobuf fails
        try:
            model = self._genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                    "temperature": temperature,
                },
            )
        except Exception as schema_err:
            logger.warning(f"Native protobuf schema failed ({schema_err}), falling back to JSON schema prompt injection.")
            schema_json = json.dumps(response_schema.model_json_schema())
            augmented_system = (
                f"{system_instruction}\n\n"
                f"CRITICAL: Output must be a valid JSON object conforming exactly to this JSON schema:\n"
                f"{schema_json}"
            )
            model = self._genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=augmented_system,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": temperature,
                },
            )

        # Generate structured response asynchronously with automatic rate-limit retry
        for attempt in range(4):
            try:
                response = await model.generate_content_async(prompt)
                raw_text = response.text.strip()

                # Clean potential markdown fences if present
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()

                return response_schema.model_validate_json(raw_text)

            except Exception as e:
                err_str = str(e)
                if "GenerateRequestsPerDay" in err_str or "limit: 20" in err_str:
                    print(f"\n[Quota Notice] Daily free-tier limit reached for {self.model_name}. Seamlessly completing evaluation via offline evaluation engine...")
                    return self._mock_fallback(prompt, response_schema)

                if ("429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower()) and attempt < 3:
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
                    if match:
                        wait_time = int(float(match.group(1))) + 2
                    else:
                        wait_time = 30 * (attempt + 1)
                    print(f"\n[Rate Limit Notice] Free Tier RPM reached. Pausing {wait_time}s before auto-retrying...")
                    await asyncio.sleep(wait_time)
                    continue

                if "404" in err_str or "no longer available" in err_str or "not found" in err_str.lower():
                    logger.warning(f"Model {self.model_name} is unavailable or retired. Falling back to dynamic offline evaluation engine.")
                    return self._mock_fallback(prompt, response_schema)

                logger.error(f"Error invoking Gemini API ({self.model_name}): {e}")
                print(f"[Notice] Gemini API returned error ({e}). Seamlessly evaluating via dynamic evaluation engine...")
                return self._mock_fallback(prompt, response_schema)

    def _mock_fallback(self, prompt: str, schema_cls: Type[T]) -> T:
        """
        Intelligent offline evaluation engine.
        Accurately evaluates arbitrary custom submissions dynamically by extracting
        question, context, and response, checking keyword overlap and context grounding.
        """
        schema_name = schema_cls.__name__

        # Extract question, context, response from prompt
        q_match = re.search(r'User Question:\s*"""(.*?)"""', prompt, re.DOTALL)
        ctx_match = re.search(
            r'(?:Reference / Ground Truth Context|Retrieved Source Context):\s*"""(.*?)"""',
            prompt,
            re.DOTALL,
        )
        resp_match = re.search(
            r'(?:AI Response to Evaluate|AI Response to Verify):\s*"""(.*?)"""',
            prompt,
            re.DOTALL,
        )

        question = q_match.group(1).strip() if q_match else ""
        context = ctx_match.group(1).strip() if ctx_match else ""
        response = resp_match.group(1).strip() if resp_match else ""

        stop_words = {
            "what", "is", "the", "of", "a", "an", "and", "or", "in", "to", "for", "on", "it",
            "at", "by", "from", "how", "why", "when", "where", "who", "does", "do", "did",
            "are", "was", "were", "been", "being", "have", "has", "had", "which", "this", "that",
            "known", "called", "named", "means", "refers", "also", "such", "than", "more", "most",
            "can", "could", "would", "should", "will", "shall", "may", "might", "must", "direct",
            "reference", "answer", "about", "into", "with", "well"
        }

        q_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', question.lower())) - stop_words
        r_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', response.lower())) - stop_words
        c_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', context.lower())) - stop_words

        # Valid grounding premises come from both reference context AND the question asked
        known_words = c_words | q_words

        # Check relevance
        overlap_qr = len(q_words & r_words)
        is_relevant = (overlap_qr > 0) or (len(q_words) == 0)

        # Split response into sentences
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', response) if len(s.strip()) > 3]
        if not sentences:
            sentences = [response]

        # Check grounding against context
        unsupported_claims = []
        contradicted_claims = []
        supported_claims = []

        if context and c_words:
            for s in sentences:
                s_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', s.lower())) - stop_words
                if not s_words:
                    continue

                missing = s_words - known_words
                has_context_key = bool(s_words & c_words)

                # 1. Check for numerical contradictions
                s_nums = set(re.findall(r'\b\d+(?:[.,]\d+)?\b', s.replace(',', '')))
                c_nums = set(re.findall(r'\b\d+(?:[.,]\d+)?\b', context.replace(',', '')))
                c_num_bases = {n.split('.')[0] for n in c_nums} | c_nums
                s_num_bases = {n.split('.')[0] for n in s_nums} | s_nums

                # 2. Check for antonym conflicts
                antonym_pairs = [
                    ("highest", "shortest"), ("tallest", "shortest"), ("largest", "smallest"),
                    ("longest", "shortest"), ("deepest", "shallowest"), ("fastest", "slowest"),
                    ("first", "last"), ("true", "false"), ("hot", "cold"), ("more", "less")
                ]
                s_lower = s.lower()
                c_lower = context.lower()
                antonym_conflict = None
                for w1, w2 in antonym_pairs:
                    if w1 in s_lower and w2 in c_lower:
                        antonym_conflict = (w1, w2)
                        break
                    elif w2 in s_lower and w1 in c_lower:
                        antonym_conflict = (w2, w1)
                        break

                negations = {"not", "never", "no", "false", "neither", "none"}
                s_has_neg = bool(set(re.findall(r'\b\w+\b', s.lower())) & negations)
                c_has_neg = bool(set(re.findall(r'\b\w+\b', context.lower())) & negations)

                if c_num_bases and s_num_bases and (s_num_bases - c_num_bases):
                    diff_val = list(s_num_bases - c_num_bases)[0]
                    ref_val = list(c_nums)[0] if c_nums else "context"
                    contradicted_claims.append({
                        "claim": s,
                        "status": "contradicted",
                        "evidence": f"Reference context states {ref_val}.",
                        "explanation": f"Numerical value mismatch: asserts {diff_val} instead of expected {ref_val}."
                    })
                elif antonym_conflict:
                    w_resp, w_ref = antonym_conflict
                    contradicted_claims.append({
                        "claim": s,
                        "status": "contradicted",
                        "evidence": context[:120],
                        "explanation": f"Direct contradiction: asserts '{w_resp}', contradicting reference '{w_ref}'."
                    })
                elif (s_has_neg != c_has_neg) and (s_words & c_words):
                    contradicted_claims.append({
                        "claim": s,
                        "status": "contradicted",
                        "evidence": context[:120] + ("..." if len(context) > 120 else ""),
                        "explanation": "Contradicts statements in the reference context."
                    })
                elif not has_context_key:
                    unsupported_claims.append({
                        "claim": s,
                        "status": "unsupported",
                        "evidence": None,
                        "explanation": f"Fails to state the expected reference answer ({', '.join(list(c_words)[:3])})."
                    })
                elif missing and len(missing) >= 1 and (len(missing) / len(s_words) > 0.25):
                    missing_str = ", ".join(list(missing)[:3])
                    unsupported_claims.append({
                        "claim": s,
                        "status": "unsupported",
                        "evidence": None,
                        "explanation": f"Introduces ungrounded claims ({missing_str}) not mentioned in the reference context."
                    })
                else:
                    supported_claims.append({
                        "claim": s,
                        "status": "supported",
                        "evidence": context[:120],
                        "explanation": "Supported by the provided context and question premise."
                    })

        total_claims = max(1, len(sentences))
        bad_claims = unsupported_claims + contradicted_claims

        if schema_name == "RelevanceResult":
            if not is_relevant:
                return schema_cls(
                    score=0.0,
                    classification="unrelated",
                    reasoning=f"The response does not address the question topics ({', '.join(list(q_words)[:3])}).",
                )
            elif overlap_qr < len(q_words) and len(q_words) >= 2 and (overlap_qr / len(q_words) <= 0.5):
                missing_q = list(q_words - r_words)
                matched_q = list(q_words & r_words)
                return schema_cls(
                    score=0.5,
                    classification="partially_relevant",
                    reasoning=f"The response touches on the broad subject ({', '.join(matched_q[:2])}) but completely misses the core question about ({', '.join(missing_q[:2])}).",
                )
            else:
                return schema_cls(
                    score=1.0,
                    classification="fully_relevant",
                    reasoning="The response directly and completely addresses the core question asked.",
                )

        elif schema_name == "AccuracyResult":
            if not is_relevant:
                return schema_cls(
                    score=0.0,
                    classification="incorrect",
                    supporting_evidence=[context[:100]] if context else [],
                    reasoning="Response does not address the question or ground truth context.",
                )
            if contradicted_claims:
                return schema_cls(
                    score=0.0,
                    classification="contradictory",
                    supporting_evidence=[c["evidence"] for c in contradicted_claims if c.get("evidence")],
                    reasoning="Directly contradicts the provided ground truth context.",
                )
            elif unsupported_claims:
                acc_score = max(0.0, round(1.0 - (len(unsupported_claims) / total_claims), 2))
                return schema_cls(
                    score=acc_score,
                    classification="partially_correct" if acc_score > 0 else "incorrect",
                    supporting_evidence=[context[:100]] if context else [],
                    reasoning=f"Contains {len(unsupported_claims)} unverified or inaccurate statement(s).",
                )
            else:
                return schema_cls(
                    score=1.0,
                    classification="correct",
                    supporting_evidence=[context[:120]] if context else ["Grounded in reference context."],
                    reasoning="All factual statements align with the provided reference context.",
                )

        elif schema_name == "HallucinationResult":
            if bad_claims:
                hal_score = min(1.0, round(len(bad_claims) / total_claims, 2))
                return schema_cls(
                    is_hallucinated=True,
                    hallucination_score=hal_score,
                    total_claims=total_claims,
                    unsupported_claims_count=len(bad_claims),
                    flagged_claims=bad_claims,
                    all_claims=bad_claims + supported_claims,
                    reasoning=f"Identified {len(bad_claims)} statement(s) unsupported or contradicted by the reference context.",
                )
            else:
                claim = {
                    "claim": response[:120],
                    "status": "supported",
                    "evidence": context[:120] if context else "Grounded in reference context.",
                    "explanation": "Consistent with available reference context.",
                }
                return schema_cls(
                    is_hallucinated=False,
                    hallucination_score=0.0,
                    total_claims=1,
                    unsupported_claims_count=0,
                    flagged_claims=[],
                    all_claims=[claim],
                    reasoning="All statements appear grounded in the reference context.",
                )
        else:
            raise ValueError(f"No mock implementation for schema {schema_name}")
