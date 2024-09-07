import json
from typing import List
from pydantic import BaseModel, Field
from .template import RobustnessTemplate


class RobustnessVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)
    score: float = Field(default=0.0)


class Reason(BaseModel):
    reason: str


class Verdicts(BaseModel):
    verdicts: List[RobustnessVerdict]


class AdversarialRobustness:
    def __init__(self, input_sentence: str):
        self.model = None
        self.template = RobustnessTemplate()
        self.input_sentence = input_sentence
        self.robustness_score = 0

    def set_model(self, model):
        self.model = model

    def get_robustness(self) -> List[str]:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            if data["score"] > 0:
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

    def get_verdict(self) -> RobustnessVerdict:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return RobustnessVerdict(
                verdict="yes" if data["score"] > 0.2 else "no",
                reason=data.get("reason", "No reason provided"),
                score=data["score"]
            )
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return RobustnessVerdict(verdict="error", reason="Error in generating verdict.", score=0.0)

    def calculate_robustness_score(self) -> float:
        verdict = self.get_verdict()
        self.robustness_score = verdict.score
        return self.robustness_score

    def _call_language_model(self, prompt: str) -> str:
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response