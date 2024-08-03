from typing import List
from loguru import logger
import sys
import json
from indoxJudge.metrics import (Faithfulness, AnswerRelevancy, Bias, ContextualRelevancy, Gruen, GEval, Rouge,
                                KnowledgeRetention, BLEU, Hallucination, Toxicity, METEOR, BertScore)

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")
logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class LlmEvaluation:
    """
    The Evaluator class is designed to evaluate various aspects of language model outputs using specified metrics.

    It supports metrics such as Faithfulness, Answer Relevancy, Bias, Contextual Relevancy, GEval, Hallucination,
    Knowledge Retention, Toxicity, BertScore, BLEU, Rouge, and METEOR.
    """

    def __init__(self, llm_as_judge,llm_response, retrieval_context, query):
        """
        Initializes the Evaluator with a language model and a list of metrics.

        Args:
            llm_as_judge: The language model .
        """
        self.model = llm_as_judge
        self.metrics = [
            Faithfulness(llm_response=llm_response, retrieval_context=retrieval_context),
            AnswerRelevancy(query=query, llm_response=llm_response),
            Bias(llm_response=llm_response),
            Hallucination(llm_response=llm_response, retrieval_context=retrieval_context),
            KnowledgeRetention(messages=[{"query": query, "llm_response": llm_response}]),
            Toxicity(messages=[{"query": query, "llm_response": llm_response}]),
            BertScore(llm_response=llm_response, retrieval_context=retrieval_context),
            BLEU(llm_response=llm_response, retrieval_context=retrieval_context),
            Rouge(llm_response=llm_response, retrieval_context=retrieval_context),
            Gruen(candidates=llm_response)
        ]
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
                elif isinstance(metric, BertScore):
                    precision, recall, f1 = metric.measure()
                    results['BertScore'] = {
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }
                    self.evaluation_score += score

                    self.metrics_score["BertScore"] = score
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
                elif isinstance(metric, BLEU):
                    score = metric.measure()
                    results['BLEU'] = {
                        'score': score
                    }

                    self.metrics_score["BLEU"] = score
                elif isinstance(metric, Rouge):
                    precision, recall, f1 = metric.measure()
                    results['rouge'] = {
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }

                    self.metrics_score["Rouge"] = score
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
                self._calculate_weighted_score()

            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {str(e)}")
        return results


class UniversalEvaluator(Evaluator):
    """
    The UniversalEvaluator class evaluates language model outputs using all available metrics.
    """

    def __init__(self, model, llm_response, retrieval_context, query):
        # if weights is None:
        #     weights = default_weights
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
            # BertScore(llm_response=llm_response, retrieval_context=retrieval_context),
            # BLEU(llm_response=llm_response, retrieval_context=retrieval_context),
            # Rouge(llm_response=llm_response, retrieval_context=retrieval_context),
            # METEOR(llm_response=llm_response, retrieval_context=retrieval_context),
            # Gruen(candidates=llm_response)
        ]
        # self.weights = weights
        super().__init__(model, metrics)
        self.metrics_score = {}
        # self.metrics_weighted_score = {}
        # self.evaluation_weighted_score = 0

    # def plot_metrics_weighted(self):
    #     """
    #     Visualizes the evaluation results using a radar chart.
    #     """
    #     from .graph.plots import MetricVisualizer
    #     visualizer = MetricVisualizer(self.metrics_weighted_score)
    #     return visualizer.show_all_plots()

    def plot_metrics(self):
        from .graph.plots import MetricVisualizer
        visualizer = MetricVisualizer(self.metrics_score)
        return visualizer.show_all_plots()
