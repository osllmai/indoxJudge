import json
from typing import List
from pydantic import BaseModel, Field
from .template import MisinformationTemplate


class MisinformationVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)
    score: float = Field(default=0.0)


class MisinformationReason(BaseModel):
    reason: str


class MisinformationVerdicts(BaseModel):
    verdicts: List[MisinformationVerdict]


class Misinformation:
    def __init__(self, input_sentence: str):
        self.model = None
        self.template = MisinformationTemplate()  
        self.input_sentence = input_sentence
        self.misinformation_score = 0

    def set_model(self, model):
        self.model = model

    def get_misinformation(self) -> List[str]:
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

    def get_reason(self) -> MisinformationReason:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return MisinformationReason(reason=data["reason"])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return MisinformationReason(reason="Error in generating reason.")

    def get_verdict(self) -> MisinformationVerdict:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return MisinformationVerdict(
                verdict="yes" if data["score"] > 0 else "no",
                reason=data.get("reason", "No reason provided"),
                score=data["score"]
            )
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return MisinformationVerdict(verdict="error", reason="Error in generating verdict.", score=0.0)

    def calculate_misinformation_score(self) -> float:
        verdict = self.get_verdict()
        self.misinformation_score = verdict.score
        return self.misinformation_score

    def _call_language_model(self, prompt: str) -> str:
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response
