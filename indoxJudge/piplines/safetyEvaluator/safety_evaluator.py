import os
from typing import List
import json
import sys
from loguru import logger

from indoxJudge.metrics import (Fairness, Harmfulness, Privacy, Misinformation, MachineEthics, StereotypeBias)

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")
logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class SafetyEvaluator:
    def __init__(self, model, metrics: List):
        self.model = model
        self.metrics = metrics
        logger.info("Evaluator initialized with model and metrics.")
        self.set_model_for_metrics()
        self.evaluation_score = 0
        self.metrics_score = {}
        self.metrics_reasons = {}

    def set_model_for_metrics(self):
        for metric in self.metrics:
            if hasattr(metric, 'set_model'):
                metric.set_model(self.model)
        logger.info("Model set for all metrics.")

    def judge(self):
        results = {}
        for metric in self.metrics:
            metric_name = metric.__class__.__name__

            logger.info(f"Evaluating metric: {metric_name}")

            if isinstance(metric, Fairness):
                score = metric.calculate_fairness_score()
                verdict = metric.get_verdict()
                reason = metric.get_reason()
                results['Fairness'] = {
                    'score': score,
                    'verdict': verdict.verdict,
                    'reason': reason.reason
                }
                self.evaluation_score += score
                self.metrics_score["Fairness"] = score
                self.metrics_reasons["Fairness"] = reason.reason

            elif isinstance(metric, Harmfulness):
                score = metric.calculate_harmfulness_score()
                verdict = metric.get_verdict()
                reason = metric.get_reason()
                results['Harmfulness'] = {
                    'score': score,
                    'verdict': verdict.verdict,
                    'reason': reason.reason
                }
                self.evaluation_score += score
                self.metrics_score["Harmfulness"] = score
                self.metrics_reasons["Harmfulness"] = reason.reason

            elif isinstance(metric, Privacy):
                score = metric.calculate_privacy_score()
                verdict = metric.get_verdict()
                reason = metric.get_reason()
                results['Privacy'] = {
                    'score': score,
                    'verdict': verdict.verdict,
                    'reason': reason.reason
                }
                self.evaluation_score += score
                self.metrics_score["Privacy"] = score
                self.metrics_reasons["Privacy"] = reason.reason

            elif isinstance(metric, Misinformation):
                score = metric.calculate_misinformation_score()
                verdict = metric.get_verdict()
                reason = metric.get_reason()
                results['Misinformation'] = {
                    'score': score,
                    'verdict': verdict.verdict,
                    'reason': reason.reason
                }
                self.evaluation_score += score
                self.metrics_score["Misinformation"] = score
                self.metrics_reasons["Misinformation"] = reason.reason

            elif isinstance(metric, MachineEthics):
                score = metric.calculate_ethics_score()
                verdict = metric.get_verdict()
                reason = metric.get_reason()
                results['MachineEthics'] = {
                    'score': score,
                    'verdict': verdict.verdict,
                    'reason': reason.reason
                }
                self.evaluation_score += score
                self.metrics_score["MachineEthics"] = score
                self.metrics_reasons["MachineEthics"] = reason.reason


        return results


class UniversalSafetyEvaluator(SafetyEvaluator):
    def __init__(self, model, llm_response):
        metrics = [
            Fairness(input_sentence=llm_response),
            Harmfulness(input_sentence=llm_response),
            Privacy(input_sentence=llm_response),
            Misinformation(input_sentence=llm_response),
            MachineEthics(input_sentence=llm_response),
            StereotypeBias(input_sentence=llm_response)
        ]
        super().__init__(model, metrics)
