from typing import List
from loguru import logger
import sys
from .metrics import Faithfulness
from .metrics import AnswerRelevancy
from .metrics import Bias
from .metrics import ContextualRelevancy
from .metrics import GEval
from .metrics import Hallucination
from .metrics import KnowledgeRetention
from .metrics import Toxicity
from .metrics import BertScore
from .metrics import BLEU
from .metrics import Rouge
from .metrics import METEOR
from .metrics import Gruen
# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")
logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class Evaluator:
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
        self.score = {}

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
                    results['faithfulness'] = {
                        'claims': claims.claims,
                        'truths': truths.truths,
                        'verdicts': [verdict.__dict__ for verdict in verdicts.verdicts],
                        'score': score,
                        'reason': reason.reason
                    }
                    self.score["faithfulness"] = score
                elif isinstance(metric, AnswerRelevancy):
                    score = metric.measure()
                    results['answer_relevancy'] = {
                        'score': score,
                        'reason': metric.reason,
                        'statements': metric.statements,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts]
                    }
                    self.score["answer_relevancy"] = score
                elif isinstance(metric, Bias):
                    score = metric.measure()
                    results['bias'] = {
                        'score': score,
                        'reason': metric.reason,
                        'opinions': metric.opinions,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts]
                    }
                    self.score["bias"] = 1 - score
                elif isinstance(metric, ContextualRelevancy):
                    # Set the language model if not already set
                    irrelevancies = metric.get_irrelevancies(metric.query, metric.retrieval_contexts)
                    metric.set_irrelevancies(irrelevancies)
                    verdicts = metric.get_verdicts(metric.query, metric.retrieval_contexts)
                    # Determine the score, e.g., based on the number of relevant contexts
                    score = 1.0 if not irrelevancies else max(0,
                                                              1.0 - len(irrelevancies) / len(metric.retrieval_contexts))
                    reason = metric.get_reason(irrelevancies, score)
                    results['contextual_relevancy'] = {
                        'verdicts': [verdict.dict() for verdict in verdicts.verdicts],
                        'reason': reason.dict()
                    }
                    self.score["contextual_relevancy"] = score
                elif isinstance(metric, GEval):

                    geval_result = metric.g_eval()
                    results['geval'] = geval_result.replace("\n", " ")
                    geval_data = json.loads(results["geval"])
                    score = geval_data["score"]
                    self.score["geval"] = int(score) / 8
                elif isinstance(metric, Hallucination):
                    score = metric.measure()
                    results['hallucination'] = {
                        'score': score,
                        'reason': metric.reason,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts]
                    }
                    self.score["hallucination"] = 1 - score
                elif isinstance(metric, KnowledgeRetention):
                    score = metric.measure()
                    results['knowledge_retention'] = {
                        'score': score,
                        'reason': metric.reason,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts],
                        'knowledges': [knowledge.data for knowledge in metric.knowledges]
                    }
                    self.score["knowledge_retention"] = score
                elif isinstance(metric, Toxicity):
                    score = metric.measure()
                    results['toxicity'] = {
                        'score': score,
                        'reason': metric.reason,
                        'opinions': metric.opinions,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts]
                    }
                    self.score["toxicity"] = 1 - score
                elif isinstance(metric, BertScore):
                    score = metric.measure()
                    # results['BertScore'] = {
                    #     'precision': score['Precision'],
                    #     'recall': score['Recall'],
                    #     'f1_score': score['F1-score']
                    # }
                    results["precision"] = score['Precision']
                    results["recall"] = score['Recall']
                    results["f1_score"] = score['F1-score']

                    self.score["precision"] = score['Precision']
                    self.score["recall"] = score['Recall']
                    self.score["f1_score"] = score['F1-score']

                elif isinstance(metric, BLEU):
                    score = metric.measure()
                    results['BLEU'] = {
                        'score': score
                    }
                    self.score["BLEU"] = score
                elif isinstance(metric, Rouge):
                    score = metric.measure()
                    results['rouge'] = {
                        'precision': score['Precision'],
                        'recall': score['Recall'],
                        'f1_score': score['F1-score']
                    }
                    self.score["Rouge"] = score
                elif isinstance(metric, METEOR):
                    score = metric.measure()
                    results['Meteor'] = {
                        'score': score
                    }
                elif isinstance(metric, Gruen):
                    score = metric.measure()
                    results['gruen'] = {
                        'score': score
                    }

                logger.info(f"Completed evaluation for metric: {metric_name}")
            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {str(e)}")
        return results


# class UniversalEvaluator(Evaluator):
#     """
#     The UniversalEvaluator class evaluates language model outputs using all available metrics.
#     """
#
#     def __init__(self, model, llm_response, retrieval_context, query):
#         metrics = [
#             Faithfulness(llm_response=llm_response, retrieval_context=retrieval_context),
#             AnswerRelevancy(query=query, llm_response=llm_response),
#             Bias(llm_response=llm_response),
#             ContextualRelevancy(query=query, retrieval_context=retrieval_context),
#             GEval(parameters="Rag Pipeline", llm_response=llm_response, query=query,
#                   retrieval_context=retrieval_context),
#             Hallucination(llm_response=llm_response, retrieval_context=retrieval_context),
#             KnowledgeRetention(messages=[{"query": query, "llm_response": llm_response}]),
#             Toxicity(messages=[{"query": query, "llm_response": llm_response}]),
#         ]
#         self.weighted_score = self._calculate_weighted_score(self.score)
#         self.weights = {
#             'faithfulness': 0.20,
#             'answer_relevancy': 0.15,
#             'bias': 0.10,
#             'contextual_relevancy': 0.20,
#             'geval': 0.10,
#             'hallucination': 0.05,
#             'knowledge_retention': 0.15,
#             'toxicity': 0.05
#         }
#
#         super().__init__(model, metrics)
#         self.score = {}
#
#     def _calculate_weighted_score(self, scores):
#         weighted_scores = {}
#         weighted_sum = 0
#         for metric, weight in self.weights.items():
#             if metric in scores:
#                 weighted_score = scores[metric] * weight
#                 weighted_scores[metric] = weighted_score
#                 weighted_sum += weighted_score
#         return weighted_score, weighted_sum
#
#     def plot_metrics(self):
#         """
#         Visualizes the evaluation results using a radar chart.
#         """
#         from .graph.plots import MetricVisualizer
#         scores_dic = self.score
#         visualizer = MetricVisualizer(scores_dic)
#         return visualizer.show_all_plots()

class UniversalEvaluator(Evaluator):
    """
    The UniversalEvaluator class evaluates language model outputs using all available metrics.
    """

    def __init__(self, model, llm_response, retrieval_context, query):
        metrics = [
            Faithfulness(llm_response=llm_response, retrieval_context=retrieval_context),
            AnswerRelevancy(query=query, llm_response=llm_response),
            Bias(llm_response=llm_response),
            ContextualRelevancy(query=query, retrieval_context=retrieval_context),
            GEval(parameters="Rag Pipeline", llm_response=llm_response, query=query,
                  retrieval_context=retrieval_context),
            Hallucination(llm_response=llm_response, retrieval_context=retrieval_context),
            KnowledgeRetention(messages=[{"query": query, "llm_response": llm_response}]),
            Toxicity(messages=[{"query": query, "llm_response": llm_response}]),

            BertScore(llm_response=llm_response, retrieval_context=retrieval_context),
            BLEU(llm_response=llm_response, retrieval_context=retrieval_context),
            Rouge(llm_response=llm_response, retrieval_context=retrieval_context),
            METEOR(llm_response=llm_response, retrieval_context=retrieval_context),
            Gruen(candidates=llm_response)

        ]
        self.weights = {
            'faithfulness': 0.20,
            'answer_relevancy': 0.15,
            'bias': 0.10,
            'contextual_relevancy': 0.20,
            'geval': 0.10,
            'hallucination': 0.05,
            'knowledge_retention': 0.15,
            'toxicity': 0.05
        }

        super().__init__(model, metrics)
        self.score = {}
        self.weighted_score,self.weighted_sum = self._calculate_weighted_score(self.score)

    def _calculate_weighted_score(self, scores):
        weighted_scores = {}
        weighted_sum = 0
        for metric, weight in self.weights.items():
            if metric in scores:
                weighted_score = scores[metric] * weight
                weighted_scores[metric] = weighted_score
                weighted_sum += weighted_score
        return weighted_scores, weighted_sum

    def plot_metrics(self):
        """
        Visualizes the evaluation results using a radar chart.
        """
        weighted_scores, _ = self._calculate_weighted_score(self.score)
        from .graph.plots import MetricVisualizer
        visualizer = MetricVisualizer(weighted_scores)
        return visualizer.show_all_plots()
