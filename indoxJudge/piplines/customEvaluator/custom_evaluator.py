import json
from typing import List
from loguru import logger
import sys
from indoxJudge.metrics import Faithfulness, Privacy, Misinformation, MachineEthics, StereotypeBias, Fairness, \
    Harmfulness, AnswerRelevancy, KnowledgeRetention, Hallucination, Toxicity, Bias, BertScore, BLEU, \
    ContextualRelevancy, GEval, METEOR, Gruen

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")
logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class CustomEvaluator:
    """
    The Evaluator class is designed to evaluate various aspects of language model outputs using specified metrics.

    It supports metrics such as Faithfulness, Answer Relevancy, Bias, Contextual Relevancy, GEval, Hallucination,
    Knowledge Retention, Toxicity, BertScore, BLEU, Rouge, and METEOR.
    """

    def __init__(self, model, metrics: List):
        """
        Initializes the Evaluator with a language model and a list of metrics.

        Args:
            model: The language model to be evaluated.
            metrics (List): A list of metric instances to evaluate the model.
        """
        self.model = model
        self.metrics = metrics
        logger.info("Evaluator initialized with model and metrics.")
        self.set_model_for_metrics()
        self.evaluation_score = 0
        self.metrics_score = {}

    def set_model_for_metrics(self):
        """
        Sets the language model for each metric that requires it.
        """
        for metric in self.metrics:
            if hasattr(metric, 'set_model'):
                metric.set_model(self.model)
        logger.info("Model set for all metrics.")

    def judge(self):
        """
        Evaluates the language model using the provided metrics and returns the results.

        Returns:
            dict: A dictionary containing the evaluation results for each metric.
        """
        results = {}
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            try:
                logger.info(f"Evaluating metric: {metric_name}")
                if isinstance(metric, Faithfulness):
                    claims = metric.evaluate_claims()
                    truths = metric.evaluate_truths()
                    verdicts = metric.evaluate_verdicts(claims.claims)
                    reason = metric.evaluate_reason(verdicts, truths.truths)
                    score = metric.calculate_faithfulness_score()
                    results['Faithfulness'] = {
                        'claims': claims.claims,
                        'truths': truths.truths,
                        'verdicts': [verdict.__dict__ for verdict in verdicts.verdicts],
                        'score': score,
                        'reason': reason.reason
                    }
                    self.evaluation_score += score
                    self.metrics_score["Faithfulness"] = score
                elif isinstance(metric, AnswerRelevancy):
                    score = metric.measure()
                    results['AnswerRelevancy'] = {
                        'score': score,
                        'reason': metric.reason,
                        'statements': metric.statements,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts]
                    }
                    self.evaluation_score += score
                    self.metrics_score["AnswerRelevancy"] = score

                # elif isinstance(metric, ContextualRelevancy):
                #     irrelevancies = metric.get_irrelevancies(metric.query, metric.retrieval_contexts)
                #     metric.set_irrelevancies(irrelevancies)
                #     verdicts = metric.get_verdicts(metric.query, metric.retrieval_contexts)
                #     metric.verdicts = verdicts.verdicts  # Ensure verdicts are stored in the metric object
                #
                #     # Determine the score, e.g., based on the number of relevant contexts
                #     score = metric.calculate_score()
                #     reason = metric.get_reason(irrelevancies, score)
                #     results = {
                #         'ContextualRelevancy': {
                #             'verdicts': [verdict.dict() for verdict in verdicts.verdicts],
                #             'reason': reason.dict(),
                #             'score': score
                #         }
                #     }
                #     self.evaluation_score += score
                #
                #     self.metrics_score["ContextualRelevancy"] = score
                # elif isinstance(metric, GEval):
                #     geval_result = metric.g_eval()
                #     results['GEVal'] = geval_result.replace("\n", " ")
                #     geval_data = json.loads(results["GEVal"])
                #     score = geval_data["score"]
                #     self.evaluation_score += int(score) / 8
                #
                #     self.metrics_score["GEVal"] = int(score) / 8

                elif isinstance(metric, KnowledgeRetention):
                    score = metric.measure()
                    results['KnowledgeRetention'] = {
                        'score': score,
                        'reason': metric.reason,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts],
                        'knowledges': [knowledge.data for knowledge in metric.knowledges]
                    }
                    self.evaluation_score += score

                    self.metrics_score["KnowledgeRetention"] = score
                elif isinstance(metric, Hallucination):
                    score = metric.measure()
                    results['Hallucination'] = {
                        'score': score,
                        'reason': metric.reason,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts]
                    }
                    self.evaluation_score += score

                    self.metrics_score["Hallucination"] = score
                elif isinstance(metric, Toxicity):
                    score = metric.measure()
                    results['Toxicity'] = {
                        'score': score,
                        'reason': metric.reason,
                        'opinions': metric.opinions,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts]
                    }
                    self.evaluation_score += score

                    self.metrics_score["Toxicity"] = score

                elif isinstance(metric, Bias):
                    score = metric.measure()
                    results['Bias'] = {
                        'score': score,
                        'reason': metric.reason,
                        'opinions': metric.opinions,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts]
                    }
                    self.evaluation_score += score
                    self.metrics_score["Bias"] = score

                elif isinstance(metric, BertScore):
                    score = metric.measure()
                    results['BertScore'] = {
                        'precision': score['Precision'],
                        'recall': score['Recall'],
                        'f1_score': score['F1-score']
                    }
                    # self.evaluation_score += score

                    # self.metrics_score["BertScore"] = score
                    self.metrics_score['precision'] = score['Precision']
                    self.evaluation_score += score['Precision']

                    self.metrics_score['recall'] = score['Recall']
                    self.evaluation_score += score['Recall']

                    self.metrics_score['f1_score'] = score['F1-score']
                    self.evaluation_score += score['F1-score']

                elif isinstance(metric, BLEU):
                    score = metric.measure()
                    results['BLEU'] = {
                        'score': score
                    }
                    self.evaluation_score += score
                    self.metrics_score["BLEU"] = score
                    # elif isinstance(metric, Rouge):
                    #     score = metric.measure()
                    #     results['rouge'] = {
                    #         'precision': score['Precision'],
                    #         'recall': score['Recall'],
                    #         'f1_score': score['F1-score']
                    #     }
                    #
                    #     self.metrics_score["Rouge"] = score
                    #     self.metrics_score["Rouge"] = score
                elif isinstance(metric, Gruen):
                    score = metric.measure()
                    results['gruen'] = {
                        'score': score[0]
                    }
                    self.evaluation_score += score[0]
                    self.metrics_score["gruen"] = score[0]
                elif isinstance(metric, AnswerRelevancy):
                    score = metric.measure()
                    results['AnswerRelevancy'] = {
                        'score': score,
                        'reason': metric.reason,
                        'statements': metric.statements,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts]
                    }
                    self.evaluation_score += score
                    self.metrics_score["AnswerRelevancy"] = score
                elif isinstance(metric, ContextualRelevancy):
                    irrelevancies = metric.get_irrelevancies(metric.query, metric.retrieval_contexts)
                    metric.set_irrelevancies(irrelevancies)
                    verdicts = metric.get_verdicts(metric.query, metric.retrieval_contexts)
                    score = 1.0 if not irrelevancies else max(0,
                                                              1.0 - len(irrelevancies) / len(metric.retrieval_contexts))
                    reason = metric.get_reason(irrelevancies, score)
                    results['ContextualRelevancy'] = {
                        'verdicts': [verdict.dict() for verdict in verdicts.verdicts],
                        'reason': reason.dict(),
                        'score': score
                    }
                    self.evaluation_score += score
                    self.metrics_score["ContextualRelevancy"] = score
                elif isinstance(metric, GEval):
                    geval_result = metric.g_eval()
                    results['GEval'] = geval_result.replace("\n", " ")
                    geval_data = json.loads(results["GEval"])
                    score = int(geval_data["score"]) / 8
                    self.evaluation_score += score
                    self.metrics_score["GEval"] = score
                elif isinstance(metric, Hallucination):
                    score = metric.measure()
                    results['Hallucination'] = {
                        'score': score,
                        'reason': metric.reason,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts]
                    }
                    self.evaluation_score += 1 - score
                    self.metrics_score["Hallucination"] = 1 - score
                elif isinstance(metric, KnowledgeRetention):
                    score = metric.measure()
                    results['KnowledgeRetention'] = {
                        'score': score,
                        'reason': metric.reason,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts],
                        'knowledges': [knowledge.data for knowledge in metric.knowledges]
                    }
                    self.evaluation_score += score
                    self.metrics_score["KnowledgeRetention"] = score

                elif isinstance(metric, METEOR):
                    score = metric.measure()
                    results["METEOR"] = {"score": score}
                    self.evaluation_score += score
                    self.metrics_score["METEOR"] = score
                elif isinstance(metric, Fairness):
                    score = metric.calculate_fairness_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["Fairness"] = score
                    results["Fairness"] = reason.reason

                elif isinstance(metric, Harmfulness):
                    score = metric.calculate_harmfulness_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["Harmfulness"] = score
                    results["Harmfulness"] = reason.reason

                elif isinstance(metric, Privacy):
                    score = metric.calculate_privacy_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["Privacy"] = score
                    results["Privacy"] = reason.reason

                elif isinstance(metric, Misinformation):
                    score = metric.calculate_misinformation_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["Misinformation"] = score
                    results["Misinformation"] = reason.reason

                elif isinstance(metric, MachineEthics):
                    score = metric.calculate_ethics_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["MachineEthics"] = score
                    results["MachineEthics"] = reason.reason

                elif isinstance(metric, StereotypeBias):
                    score = metric.calculate_stereotype_bias_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["StereotypeBias"] = score
                    results["StereotypeBias"] = reason.reason
                logger.info(f"Completed evaluation for metric: {metric_name}")

            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {str(e)}")
        return results

    def plot(self, mode="external"):
        from indoxJudge.graph import Visualization
        from indoxJudge.utils import create_model_dict
        graph_input = create_model_dict(name="LLM Evaluator", metrics=self.metrics_score,
                                        score=self.evaluation_score / len(self.metrics))
        visualizer = Visualization(data=graph_input, mode="llm")
        return visualizer.plot(mode=mode)
