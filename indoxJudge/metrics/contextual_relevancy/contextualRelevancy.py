import json
from typing import List
from pydantic import BaseModel, Field
from .template import ContextualRelevancyTemplate


class ContextualRelevancyVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[ContextualRelevancyVerdict]


class Reason(BaseModel):
    reason: str


class ContextualRelevancy:
    def __init__(self, query: str, retrieval_context: List[str]):
        self.model = None
        self.template = ContextualRelevancyTemplate()
        self.query = query
        self.retrieval_contexts = retrieval_context
        self.irrelevancies = []
        self.verdicts = []
        self.score = 0

    def set_model(self, model):
        self.model = model

    def get_irrelevancies(self, query: str, retrieval_contexts: List[str]) -> List[str]:
        irrelevancies = []
        for retrieval_context in retrieval_contexts:
            prompt = self.template.generate_verdict(query, retrieval_context)
            response = self._call_language_model(prompt)
            try:
                data = json.loads(response)
                if data["verdict"].strip().lower() == "no":
                    irrelevancies.append(data["reason"])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        return irrelevancies

    def set_irrelevancies(self, irrelevancies: List[str]):
        self.irrelevancies = irrelevancies

    def get_reason(self, irrelevancies: List[str], score: float) -> Reason:
        prompt = self.template.generate_reason(self.query, irrelevancies, score)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return Reason(reason=data["reason"])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return Reason(reason="Error in generating reason.")

    def get_verdict(self, query: str, retrieval_context: str) -> ContextualRelevancyVerdict:
        prompt = self.template.generate_verdict(query=query, context=retrieval_context)
        response = self._call_language_model(prompt)
        try:
            data = json.loads(response)
            return ContextualRelevancyVerdict(
                verdict=data["verdict"],
                reason=data.get("reason", "No reason provided")
            )
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return ContextualRelevancyVerdict(verdict="error", reason="Error in generating verdict.")

    def get_verdicts(self, query: str, retrieval_contexts: List[str]) -> Verdicts:
        verdicts = [self.get_verdict(query, retrieval_context) for retrieval_context in retrieval_contexts]
        return Verdicts(verdicts=verdicts)

    def calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1.0  # If no verdicts, assume full relevancy by default.

        relevant_count = sum(1 for verdict in self.verdicts if verdict.verdict.strip().lower() == "yes")
        partial_relevant_count = sum(1 for verdict in self.verdicts if verdict.verdict.strip().lower() == "partial")

        score = (relevant_count + 0.5 * partial_relevant_count) / number_of_verdicts
        return score

    def _call_language_model(self, prompt: str) -> str:
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response

