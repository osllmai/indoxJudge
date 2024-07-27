from typing import List
from pydantic import BaseModel, Field
import json

from .template import FaithfulnessTemplate


class FaithfulnessVerdict(BaseModel):
    """
    Model representing a verdict on the faithfulness of a claim,
    including the verdict itself and the reasoning behind it.
    """
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    """
    Model representing a list of FaithfulnessVerdict instances.
    """
    verdicts: List[FaithfulnessVerdict]


class Truths(BaseModel):
    """
    Model representing a list of truths extracted from the LLM response.
    """
    truths: List[str]


class Claims(BaseModel):
    """
    Model representing a list of claims extracted from the LLM response.
    """
    claims: List[str]


class Reason(BaseModel):
    """
    Model representing the reason provided for a given score or set of contradictions.
    """
    reason: str


class Faithfulness:
    """
    Class for evaluating the faithfulness of language model outputs by analyzing
    claims, truths, verdicts, and reasons using a specified language model.
    """
    def __init__(self, llm_response, retrieval_context):
        """
        Initializes the Faithfulness class with the LLM response and retrieval context.

        :param llm_response: The response generated by the language model.
        :param retrieval_context: The context used for retrieval during evaluation.
        """
        self.model = None
        self.llm_response = llm_response
        self.retrieval_context = retrieval_context

    def set_model(self, model):
        """
        Sets the language model to be used for evaluation.

        :param model: The language model to use.
        """
        self.model = model

    def evaluate_claims(self) -> Claims:
        """
        Evaluates and extracts claims from the LLM response.

        :return: A Claims object containing the list of claims.
        """
        prompt = FaithfulnessTemplate.generate_claims(self.llm_response)
        response = self._call_language_model(prompt)
        claims = json.loads(response).get('claims', [])
        return Claims(claims=claims)

    def evaluate_truths(self) -> Truths:
        """
        Evaluates and extracts truths from the LLM response.

        :return: A Truths object containing the list of truths.
        """
        prompt = FaithfulnessTemplate.generate_truths(self.llm_response)
        response = self._call_language_model(prompt)
        truths = json.loads(response).get('truths', [])
        return Truths(truths=truths)

    def evaluate_verdicts(self, claims: List[str]) -> Verdicts:
        """
        Evaluates the verdicts on the faithfulness of the given claims.

        :param claims: List of claims to evaluate.
        :return: A Verdicts object containing the list of verdicts.
        """
        prompt = FaithfulnessTemplate.generate_verdicts(claims, self.retrieval_context)
        response = self._call_language_model(prompt)
        verdicts = json.loads(response).get('verdicts', [])
        return Verdicts(verdicts=[FaithfulnessVerdict(**verdict) for verdict in verdicts])

    def evaluate_reason(self, score: float, contradictions: List[str]) -> Reason:
        """
        Evaluates the reason behind a given score or set of contradictions.

        :param score: The score assigned to the evaluation.
        :param contradictions: A list of contradictions identified.
        :return: A Reason object containing the reasoning.
        """
        prompt = FaithfulnessTemplate.generate_reason(score, contradictions)
        response = self._call_language_model(prompt)
        reason = json.loads(response).get('reason', '')
        return Reason(reason=reason)

    def _call_language_model(self, prompt: str) -> str:
        """
        Calls the language model with the given prompt and returns the response.

        :param prompt: The prompt to provide to the language model.
        :return: The response from the language model.
        """
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response