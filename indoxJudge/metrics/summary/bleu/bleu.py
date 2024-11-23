from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from loguru import logger
import json
import re
from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import nltk
import math
from .bleuTemplate import BleuTemplate


class ModelConfig:
    arbitrary_types_allowed = True


class BleuScore(BaseModel):
    score: float
    precisions: List[float]
    brevity_penalty: float
    details: Dict[str, Any] = Field(default_factory=dict)

    class Config(ModelConfig):
        pass


class BleuVerdict(BaseModel):
    score: BleuScore
    reason: Optional[str] = None

    class Config(ModelConfig):
        pass


class Bleu:
    def __init__(
        self,
        summary: str,
        source: str,
        weights: List[float] = None,
        include_reason: bool = True,
    ):
        self.generated = summary
        self.reference = source
        self.weights = weights or [0.4, 0.3, 0.2, 0.1]
        self.include_reason = include_reason
        self.model = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def set_model(self, model):
        self.model = model

    def measure(self) -> Dict:
        """Calculate BLEU score and return detailed results."""
        nltk.download("punkt")
        score, precisions, bp, details = self._calculate_bleu_score()

        bleu_score = BleuScore(
            score=score, precisions=precisions, brevity_penalty=bp, details=details
        )

        if self.include_reason:
            verdict = self._generate_final_verdict(bleu_score.dict(), score)
        else:
            verdict = None

        return {
            "overall_score": round(score, 3),
            "verdict": verdict,
            "detailed_scores": bleu_score.dict(),
        }

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BLEU calculation."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r"[^a-z0-9\s]", "", text)
        # Tokenize
        return word_tokenize(text)

    def _count_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Count n-grams in the token list."""
        return Counter(tuple(gram) for gram in ngrams(tokens, n))

    def _modified_precision(
        self, generated_tokens: List[str], reference_tokens: List[str], n: int
    ) -> Tuple[float, Dict]:
        """Calculate modified precision"""
        generated_ngrams = self._count_ngrams(generated_tokens, n)
        reference_ngrams = self._count_ngrams(reference_tokens, n)

        # Add smoothing for higher order n-grams
        smoothing = 1.0 if n <= 2 else (0.5 ** (n - 2))

        clipped_counts = Counter()
        for ngram, count in generated_ngrams.items():
            clipped_counts[ngram] = min(count, reference_ngrams[ngram]) + smoothing

        denominator = max(sum(generated_ngrams.values()), 1)
        numerator = sum(clipped_counts.values())

        precision = numerator / (denominator + smoothing)

        details = {
            f"matching_{n}grams": numerator,
            f"total_generated_{n}grams": sum(generated_ngrams.values()),
            f"total_reference_{n}grams": sum(reference_ngrams.values()),
        }

        return precision, details

    def _brevity_penalty(self, generated_len: int, reference_len: int) -> float:
        """Calculate brevity penalty"""
        if generated_len > reference_len:
            return 1.0
        elif generated_len == 0:
            return 0.0
        else:
            return math.exp(1 - (reference_len / generated_len))

    def _calculate_bleu_score(self) -> Tuple[float, List[float], float, Dict]:
        """Calculate BLEU score with detailed metrics."""
        generated_tokens = self._preprocess_text(self.generated)
        reference_tokens = self._preprocess_text(self.reference)

        # Calculate modified precisions for n-grams
        precisions = []
        details = {}
        for n in range(1, 5):  # BLEU-1 to BLEU-4
            precision, detail = self._modified_precision(
                generated_tokens, reference_tokens, n
            )
            precisions.append(precision)
            details.update(detail)

        # Calculate brevity penalty
        bp = self._brevity_penalty(len(generated_tokens), len(reference_tokens))

        # Calculate final score
        if min(precisions) > 0:
            log_precisions = [w * math.log(p) for w, p in zip(self.weights, precisions)]
            score = bp * math.exp(sum(log_precisions))
        else:
            score = 0.0

        details.update(
            {
                "generated_length": len(generated_tokens),
                "reference_length": len(reference_tokens),
                "brevity_penalty": bp,
            }
        )

        return score, precisions, bp, details

    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from language model."""
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response

    def _call_language_model(self, prompt: str) -> str:
        """Call language model for generating verdict."""
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

    def _generate_final_verdict(self, score: Dict, final_score: float) -> str:
        """Generate final verdict based on BLEU scores."""
        prompt = BleuTemplate.generate_final_verdict(
            score=score, final_score=final_score
        )
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["verdict"]
