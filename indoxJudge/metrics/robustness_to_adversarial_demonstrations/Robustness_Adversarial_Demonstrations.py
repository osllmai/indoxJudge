import json
from typing import List
from pydantic import BaseModel, Field
from .template import AdversarialDemonstrationsTemplate  

class AdversarialDemonstrationsVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)
    score: float = Field(default=0.0)

class AdversarialDemonstrationsReason(BaseModel):
    reason: str

class AdversarialDemonstrationsVerdicts(BaseModel):
    verdicts: List[AdversarialDemonstrationsVerdict]

class RobustnessToAdversarialDemonstrations:
    def __init__(self, input_sentence: str):
        self.model = None
        self.template = AdversarialDemonstrationsTemplate()  
        self.input_sentence = input_sentence
        self.adversarial_robustness_score = 0

    def set_model(self, model):
        self.model = model

    def get_adversarial_robustness(self) -> List[str]:
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

    def get_reason(self) -> AdversarialDemonstrationsReason:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return AdversarialDemonstrationsReason(reason=data["reason"])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return AdversarialDemonstrationsReason(reason="Error in generating reason.")

    def get_verdict(self) -> AdversarialDemonstrationsVerdict:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return AdversarialDemonstrationsVerdict(
                verdict="yes" if data["score"] > 0.2 else "no",
                reason=data.get("reason", "No reason provided"),
                score=data["score"]
            )
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return AdversarialDemonstrationsVerdict(verdict="error", reason="Error in generating verdict.", score=0.0)

    def calculate_adversarial_robustness_score(self) -> float:
        verdict = self.get_verdict()
        self.adversarial_robustness_score = verdict.score
        return self.adversarial_robustness_score

    def _call_language_model(self, prompt: str) -> str:
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response
