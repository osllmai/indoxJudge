import json
from typing import List
from pydantic import BaseModel, Field
from .template import FairnessTemplate


class FairnessVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)
    score: float = Field(default=0.0)


class Reason(BaseModel):
    reason: str


class Verdicts(BaseModel):
    verdicts: List[FairnessVerdict]


class Fairness:
    def __init__(self, input_sentence: str):
        self.model = None
        self.template = FairnessTemplate()
        self.input_sentence = input_sentence
        self.fairness_score = 0

    def set_model(self, model):
        self.model = model

    def get_unfairness(self) -> List[str]:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            if data["score"] < 1:
                return [data["reason"]]
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return []

    def get_reason(self) -> Reason:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return Reason(reason=data["reason"])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return Reason(reason="Error in generating reason.")

    def get_verdict(self) -> FairnessVerdict:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            # Updated logic for verdict based on score
            verdict = "no" if data["score"] > 0 else "yes"
            return FairnessVerdict(
                verdict=verdict,
                reason=data.get("reason", "No reason provided"),
                score=data["score"]
            )
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return FairnessVerdict(verdict="error", reason="Error in generating verdict.", score=0.0)

    def calculate_fairness_score(self) -> float:
        verdict = self.get_verdict()
        self.fairness_score = verdict.score
        return self.fairness_score

    def _call_language_model(self, prompt: str) -> str:
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response
