import os
from dotenv import load_dotenv
import requests
import urllib3
from typing import List
from pydantic import BaseModel, Field
from loguru import logger
import json
import sys
from .metrics import Fairness, Harmfulness, Privacy, Misinformation

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
            try:
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

                elif isinstance(metric, Jailbreak):
                    score = metric.calculate_jailbreak_score()
                    verdict = metric.get_verdict()
                    reason = metric.get_reason()
                    results['Jailbreak'] = {
                        'score': score,
                        'verdict': verdict.verdict,
                        'reason': reason.reason
                    }
                    self.evaluation_score += score
                    self.metrics_score["Jailbreak"] = score
                    self.metrics_reasons["Jailbreak"] = reason.reason
                logger.info(f"Completed evaluation for metric: {metric_name}")

            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {str(e)}")
        return results

class UniversalSafetyEvaluator(SafetyEvaluator):
    def __init__(self, model, llm_response):
        metrics = [
            Fairness(input_sentence=llm_response),
            Harmfulness(input_sentence=llm_response),
            Privacy(input_sentence=llm_response),
            Misinformation(input_sentence=llm_response),
        ]
        super().__init__(model, metrics)

    def plot_results(self):
        from Safety_Plot import Visualization 
        visualization = Visualization(self.metrics_score, self.metrics_reasons)
        
        visualization.bar_plot()
        visualization.radar_plot()
        for metric in self.metrics_score:
            visualization.speedometer_plot(metric)

        metric_visualizer = MetricVisualizer(self.metrics_score)
        
        metric_visualizer.show_all_plots()

model = CustomModel()
evaluator = UniversalSafetyEvaluator(model, llm_response)
results = evaluator.judge()    
