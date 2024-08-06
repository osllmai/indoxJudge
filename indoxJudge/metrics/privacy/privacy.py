import json
from typing import List
from pydantic import BaseModel, Field
from .template import PrivacyTemplate


class PrivacyVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)
    score: float = Field(default=0.0)


class PrivacyReason(BaseModel):
    reason: str


class PrivacyVerdicts(BaseModel):
    verdicts: List[PrivacyVerdict]


class Privacy:
    def __init__(self, input_sentence: str):
        self.model = None
        self.template = PrivacyTemplate()  
        self.input_sentence = input_sentence
        self.privacy_score = 0

    def set_model(self, model):
        self.model = model

    def get_privacy(self) -> List[str]:
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

    def get_reason(self) -> PrivacyReason:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return PrivacyReason(reason=data["reason"])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return PrivacyReason(reason="Error in generating reason.")

    def get_verdict(self) -> PrivacyVerdict:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return PrivacyVerdict(
                verdict="yes" if data["score"] > 0 else "no",
                reason=data.get("reason", "No reason provided"),
                score=data["score"]
            )
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return PrivacyVerdict(verdict="error", reason="Error in generating verdict.", score=0.0)

    def calculate_privacy_score(self) -> float:
        verdict = self.get_verdict()
        self.privacy_score = verdict.score
        return self.privacy_score

    def _call_language_model(self, prompt: str) -> str:
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response
