from typing import List
import json
import logging

from Harmfulness_evaluation.harmfulness import  HarmfulnessAnalysis, HarmfulnessEvaluateResponse
from Fairness_evaluation.fairness import FairnessAnalysis, FairnessEvaluateResponse
from Privacy_evaluation.privacy import PrivacyAnalysis,PrivacyEvaluateResponse
from Jailbreak_evaluation.jailbreak import  JailbreakAnalysis, JailbreakEvaluateResponse
from Robustness_evaluation.robustness import  RobustnessAnalysis, RobustnessEvaluateResponse

import plotly.express as px
import plotly.graph_objects as go
from visualization import VisualizeAnalysis

logger = logging.getLogger(__name__)

class Evaluator:
    """
    The Evaluator class evaluates text using specified analysis and evaluation classes.
    """

    def __init__(self, model, llm_response: str, metrics: List[str]):
        """
        Initializes the Evaluator with the model, response to be analyzed, and a list of metrics.

        Args:
            model: The language model to be evaluated.
            llm_response (str): The response to be analyzed.
            metrics (List[str]): A list of metrics to evaluate the response.
        """
        self.model = model
        self.llm_response = llm_response
        self.metrics = metrics
        self.results = {}
        self.score = {}

        self.metric_mapping = {
            'fairness': (FairnessAnalysis, FairnessEvaluateResponse),
            'privacy': (PrivacyAnalysis, PrivacyEvaluateResponse),
            'jailbreak': (JailbreakAnalysis, JailbreakEvaluateResponse),
            'robustness': (RobustnessAnalysis, RobustnessEvaluateResponse),
            'harmfulness': (HarmfulnessAnalysis, HarmfulnessEvaluateResponse)
        }
        logger.info("Evaluator initialized with model, response, and metrics.")

    def judge(self):
        """
        Evaluates the response using the provided metrics and returns the results.

        Returns:
            dict: A dictionary containing the evaluation results for each metric.
        """
        for metric in self.metrics:
            if metric in self.metric_mapping:
                analysis_class, evaluate_class = self.metric_mapping[metric]
                analysis = analysis_class()
                result = analysis.analyze(self.llm_response)
                result_json = json.loads(result)
                evaluator = evaluate_class(result)
                self.results[metric] = {
                    'result_json': result_json,
                    'score': evaluator.score(),
                    'feedback': evaluator.feedback(),
                    'verdict': evaluator.verdict()
                }
                self.score[metric] = self.results[metric]['score']
            else:
                logger.error(f"Metric '{metric}' is not recognized.")
        return self.results

    def visualize(self):
        """
        Generates visualizations of the evaluation results and saves them as HTML files.
        """
        scores = {metric: result['score'] for metric, result in self.results.items()}
        visualizer = VisualizeAnalysis(scores, self.model)

        visualizer.bar_chart('Analysis_Scores_Plotly.html')
        visualizer.radar_chart('Radar_Analysis_Scores_Plotly.html')

class EvaluationProcess:
    def __init__(self, model, llm_response: str, metrics: List[str]):
        self.evaluator = Evaluator(model, llm_response, metrics)

    def run(self):
        """
        Runs the evaluation process, saves results, and generates visualizations.
        """
        results = self.evaluator.judge()

        for metric, result in results.items():
            print(f"{metric.capitalize()} Analysis:")
            print(f"Score: {result['score']}")
            print(f"Explanation: {result['feedback']}")
            print(f"Verdict: {result['verdict']}")
            print()

        output_filename = "Analysis_Results.json"
        with open(output_filename, 'w') as json_file:
            json.dump(results, json_file, indent=4)

        print(f"Results saved to {output_filename}")

        self.evaluator.visualize()

if __name__ == "__main__":
    llm_response = "The development team encountered complex ethical challenges while designing the AI system, particularly in terms of balancing user privacy with the need for comprehensive data collection. They were tasked with ensuring that the system’s capabilities did not compromise ethical standards, which required careful consideration of how to handle and protect sensitive information."

    all_metrics = ["fairness", "privacy", "jailbreak", "robustness", "harmfulness"]
    selected_metrics = all_metrics  
    model_name = "gpt-3.5-turbo-0125"

    evaluation_process = EvaluationProcess(model_name, llm_response, selected_metrics)
    evaluation_process.run()
