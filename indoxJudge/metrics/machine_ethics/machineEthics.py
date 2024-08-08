import json
from typing import List
from pydantic import BaseModel, Field
from .template import MachineEthicsTemplate

class EthicsVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)
    score: float = Field(default=0.0)

class EthicsReason(BaseModel):
    reason: str

class EthicsVerdicts(BaseModel):
    verdicts: List[EthicsVerdict]

class MachineEthics:
    def __init__(self, input_sentence: str):
        self.model = None
        self.template = MachineEthicsTemplate()  
        self.input_sentence = input_sentence
        self.ethics_score = 0

    def set_model(self, model):
        self.model = model

    def get_ethics(self) -> List[str]:
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

    def get_reason(self) -> EthicsReason:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return EthicsReason(reason=data["reason"])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return EthicsReason(reason="Error in generating reason.")

    def get_verdict(self) -> EthicsVerdict:
        prompt = self.template.generate_reason(self.input_sentence)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return EthicsVerdict(
                verdict="yes" if data["score"] > 0.0 else "no",
                reason=data.get("reason", "No reason provided"),
                score=data["score"]
            )
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return EthicsVerdict(verdict="error", reason="Error in generating verdict.", score=0.0)

    def calculate_ethics_score(self) -> float:
        verdict = self.get_verdict()
        self.ethics_score = verdict.score
        return self.ethics_score

    def _call_language_model(self, prompt: str) -> str:
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response
