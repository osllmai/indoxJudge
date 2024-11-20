from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from loguru import logger
import json
import sys
from .factualConsistencyTemplate import FactualConsistencyTemplate


class FactualClaim(BaseModel):
    claim: str
    source_evidence: Optional[str]
    consistency_score: float
    error_type: Optional[str] = None
    explanation: str


class CategoryVerdict(BaseModel):
    category: str
    score: float
    consistent_claims: List[str]
    inconsistent_claims: List[Dict[str, str]]  # claim -> error explanation
    reason: str = Field(default=None)


class ConsistencyScores(BaseModel):
    scores: List[CategoryVerdict]


class FactualConsistency:
    def __init__(
        self,
        summary: str,
        source_text: str,
        category_weights: Dict[str, float] = None,
        consistency_threshold: float = 0.8,
    ):
        self.summary = summary
        self.source_text = source_text
        self.consistency_threshold = consistency_threshold
        self.category_weights = category_weights or {
            "numerical_claims": 0.25,
            "entity_claims": 0.25,
            "causal_claims": 0.20,
            "descriptive_claims": 0.15,
            "comparative_claims": 0.15,
        }
        self.model = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.error_types = {
            "contradiction": "Direct conflict with source",
            "exaggeration": "Overstatement of facts",
            "misattribution": "Incorrect source or attribution",
            "unsupported": "Claim without source evidence",
            "oversimplification": "Excessive simplification losing accuracy",
        }

    def set_model(self, model):
        self.model = model

    def measure(self) -> float:
        self.summary_claims = self._extract_claims(self.summary)
        self.verified_claims = self._verify_claims()
        self.category_scores = self._generate_category_verdict()
        self.score = self._calculate_weighted_score()
        self.consistency_stats = self._calculate_consistency_statistics()

        logger.info(
            f"Token Usage Summary:\n Total Input: {self.total_input_tokens} | "
            f"Total Output: {self.total_output_tokens} | "
            f"Total: {self.total_input_tokens + self.total_output_tokens}"
        )
        return {
            "score": round(self.score, 3),
            "summary_claims": self.summary_claims,
            "verified_claims": self.verified_claims,
            "category_scores": self.category_scores,
            "consistency_stats": self.consistency_stats,
        }

    def _extract_claims(self, text: str) -> List[Dict]:
        prompt = FactualConsistencyTemplate.extract_claims(text=text)
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["claims"]

    def _verify_claims(self) -> List[FactualClaim]:
        prompt = FactualConsistencyTemplate.verify_claims(
            summary_claims=self.summary_claims, source_text=self.source_text
        )
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return [FactualClaim(**claim) for claim in data["verified_claims"]]

    def _generate_category_verdict(self) -> List[CategoryVerdict]:
        claims_dict = [claim.dict() for claim in self.verified_claims]
        prompt = FactualConsistencyTemplate.generate_category_verdict(
            verified_claims=claims_dict
        )
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return [CategoryVerdict(**score) for score in data["scores"]]

    def _calculate_weighted_score(self) -> float:
        total_score = 0.0
        for verdict in self.category_scores:
            category = verdict.category.lower().replace(" ", "_")
            weight = self.category_weights.get(category, 0.2)
            total_score += verdict.score * weight
        return total_score

    def _calculate_consistency_statistics(self) -> Dict:
        stats = {
            "total_claims": len(self.verified_claims),
            "consistent_claims": len(
                [
                    c
                    for c in self.verified_claims
                    if c.consistency_score >= self.consistency_threshold
                ]
            ),
            "error_distribution": self._calculate_error_distribution(),
            "category_stats": self._calculate_category_stats(),
            "severity_distribution": {
                "high": len(
                    [c for c in self.verified_claims if c.consistency_score < 0.5]
                ),
                "medium": len(
                    [
                        c
                        for c in self.verified_claims
                        if 0.5 <= c.consistency_score < self.consistency_threshold
                    ]
                ),
                "low": len(
                    [
                        c
                        for c in self.verified_claims
                        if c.consistency_score >= self.consistency_threshold
                    ]
                ),
            },
        }
        return stats

    def _calculate_error_distribution(self) -> Dict[str, int]:
        distribution = {error_type: 0 for error_type in self.error_types}
        for claim in self.verified_claims:
            if claim.error_type:
                distribution[claim.error_type] = (
                    distribution.get(claim.error_type, 0) + 1
                )
        return distribution

    def _calculate_category_stats(self) -> Dict[str, Dict]:
        stats = {}
        for verdict in self.category_scores:
            category = verdict.category
            stats[category] = {
                "total_claims": len(verdict.consistent_claims)
                + len(verdict.inconsistent_claims),
                "consistent_claims": len(verdict.consistent_claims),
                "inconsistent_claims": len(verdict.inconsistent_claims),
                "score": verdict.score,
            }
        return stats

    def get_error_examples(self) -> Dict[str, List[str]]:
        """Return examples of each type of error found in the summary."""
        examples = {error_type: [] for error_type in self.error_types}
        for claim in self.verified_claims:
            if claim.error_type:
                examples[claim.error_type].append(
                    {"claim": claim.claim, "explanation": claim.explanation}
                )
        return examples

    def _clean_json_response(self, response: str) -> str:
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response

    def _call_language_model(self, prompt: str) -> str:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        input_token_count = len(enc.encode(prompt))
        response = self.model.generate_evaluation_response(prompt=prompt)
        self.total_input_tokens += input_token_count

        if not response:
            raise ValueError("Received an empty response from the model.")

        clean_response = self._clean_json_response(response=response)
        output_token_count = len(enc.encode(response))
        self.total_output_tokens += output_token_count
        logger.info(
            f"Token Counts - Input: {input_token_count} | Output: {output_token_count}"
        )

        return clean_response
