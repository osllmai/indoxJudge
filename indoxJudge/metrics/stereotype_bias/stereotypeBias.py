import json
from typing import List
from pydantic import BaseModel, Field
from .template import StereotypeBiasTemplate  

class StereotypeBiasVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)
    score: float = Field(default=0.0)

class StereotypeBiasReason(BaseModel):
    reason: str

class StereotypeBiasVerdicts(BaseModel):
    verdicts: List[StereotypeBiasVerdict]

    
class StereotypeBias:
    def __init__(self, input_sentence: str):
        self.model = None
        self.template = StereotypeBiasTemplate()  # New template for stereotype and bias
        self.input_sentence = input_sentence
        self.stereotype_bias_score = 0

    def set_model(self, model):
        self.model = model

    def get_stereotype_bias(self) -> List[str]:
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

    def get_reason(self) -> StereotypeBiasReason:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return StereotypeBiasReason(reason=data["reason"])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return StereotypeBiasReason(reason="Error in generating reason.")

    def get_verdict(self) -> StereotypeBiasVerdict:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return StereotypeBiasVerdict(
                verdict="yes" if data["score"] > 0.2 else "no",
                reason=data.get("reason", "No reason provided"),
                score=data["score"]
            )
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return StereotypeBiasVerdict(verdict="error", reason="Error in generating verdict.", score=0.0)

    def calculate_stereotype_bias_score(self) -> float:
        verdict = self.get_verdict()
        self.stereotype_bias_score = verdict.score
        return self.stereotype_bias_score

    def _call_language_model(self, prompt: str) -> str:
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response
