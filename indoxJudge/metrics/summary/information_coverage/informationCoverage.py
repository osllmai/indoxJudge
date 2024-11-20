from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from loguru import logger
import json
import sys
from .informationCoverageTemplate import CoverageTemplate


class InformationElement(BaseModel):
    category: str
    content: str
    importance: float
    covered: bool = False
    coverage_quality: Optional[float] = None


class CoverageVerdict(BaseModel):
    category: str
    score: float
    elements_covered: List[str]
    elements_missed: List[str]
    reason: str = Field(default=None)


class CoverageScores(BaseModel):
    scores: List[CoverageVerdict]


class InformationCoverage:
    def __init__(
        self,
        summary: str,
        source_text: str,
        category_weights: Dict[str, float] = None,
        importance_threshold: float = 0.7,
    ):
        self.summary = summary
        self.source_text = source_text
        self.importance_threshold = importance_threshold
        self.category_weights = category_weights or {
            "core_facts": 0.35,
            "supporting_details": 0.25,
            "context": 0.15,
            "relationships": 0.15,
            "conclusions": 0.10,
        }
        self.model = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def set_model(self, model):
        self.model = model

    def measure(self) -> float:
        self.information_elements = self._extract_information_elements()
        self.coverage_scores = self._evaluate_coverage()
        self.score = self._calculate_weighted_score()
        self.coverage_stats = self._calculate_coverage_statistics()
        self.verdict = self._generate_final_verdict()

        logger.info(
            f"Token Usage Summary:\n Total Input: {self.total_input_tokens} | "
            f"Total Output: {self.total_output_tokens} | "
            f"Total: {self.total_input_tokens + self.total_output_tokens}"
        )
        return {
            "score": round(self.score, 3),
            "information_elements": self.information_elements,
            "coverage_scores": self.coverage_scores,
            "coverage_stats": self.coverage_stats,
            "verdicts": self.verdict,
        }

    def _extract_information_elements(self) -> List[InformationElement]:
        prompt = CoverageTemplate.extract_information_elements(text=self.source_text)
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return [InformationElement(**element) for element in data["elements"]]

    def _evaluate_coverage(self) -> List[CoverageVerdict]:
        elements_dict = [element.dict() for element in self.information_elements]
        prompt = CoverageTemplate.evaluate_coverage(
            summary=self.summary, elements=elements_dict
        )
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return [CoverageVerdict(**score) for score in data["scores"]]

    def _calculate_coverage_statistics(self) -> Dict:
        stats = {
            "total_elements": len(self.information_elements),
            "critical_elements": len(
                [
                    e
                    for e in self.information_elements
                    if e.importance >= self.importance_threshold
                ]
            ),
            "coverage_by_importance": {
                "high": self._calculate_importance_coverage(0.8, 1.0),
                "medium": self._calculate_importance_coverage(0.5, 0.8),
                "low": self._calculate_importance_coverage(0.0, 0.5),
            },
            "coverage_by_category": {
                score.category: {
                    "covered_count": len(score.elements_covered),
                    "total_count": len(score.elements_covered)
                    + len(score.elements_missed),
                    "coverage_rate": score.score,
                }
                for score in self.coverage_scores
            },
        }
        return stats

    def _calculate_importance_coverage(
        self, min_importance: float, max_importance: float
    ) -> Dict:
        elements = [
            e
            for e in self.information_elements
            if min_importance <= e.importance < max_importance
        ]
        if not elements:
            return {"coverage_rate": 0, "element_count": 0}

        covered = [e for e in elements if e.covered]
        return {
            "coverage_rate": len(covered) / len(elements),
            "element_count": len(elements),
        }

    def _calculate_weighted_score(self) -> float:
        total_score = 0.0
        for verdict in self.coverage_scores:
            category = verdict.category.lower().replace(" ", "_")
            weight = self.category_weights.get(category, 0.2)
            total_score += verdict.score * weight
        return total_score

    def _generate_final_verdict(self) -> str:
        scores_dict = [score.dict() for score in self.coverage_scores]
        prompt = CoverageTemplate.generate_final_verdict(
            scores=scores_dict,
            weighted_score=self.score,
            coverage_stats=self.coverage_stats,
        )
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["verdict"]

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
