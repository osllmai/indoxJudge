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

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

logger.remove()  
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

                elif isinstance(metric, Misinformation):
                    score = metric.calculate_mininformation_score()
                    verdict = metric.get_verdict()
                    reason = metric.get_reason()
                    results['Mininformation'] = {
                        'score': score,
                        'verdict': verdict.verdict,
                        'reason': reason.reason
                    }
                    self.evaluation_score += score
                    self.metrics_score["Mininformation"] = score

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
            Misinformation(input_sentence=llm_response)
        ]
        super().__init__(model, metrics)
        self.metrics_score = {}


def get_openai_response(prompt):
    """
    Get a response from the OpenAI API for the given prompt.
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    api_url = 'http://5.78.55.161/api/chat_completion/generate/'  # Ensure this URL is correct
    params = {
        "model": "gpt-3.5-turbo-0125",
        "messages": [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": False
    }

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    response = requests.post(api_url, headers=headers, json=params, verify=False)
    response.raise_for_status() 
    return response.json()

class CustomModel:
    def generate_evaluation_response(self, prompt: str) -> str:
        response = get_openai_response(prompt)
        content = response.get('text_message', '{}')
        return content
    
llm_response =  "Most people believe that teamwork is essential for successful project completion, although some individuals may work better independently."

model = CustomModel()
evaluator = UniversalSafetyEvaluator(model, llm_response)
results = evaluator.judge()    
