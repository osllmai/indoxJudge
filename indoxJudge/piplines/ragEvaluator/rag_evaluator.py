from loguru import logger
import sys
import json
from indoxJudge.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextualRelevancy,
    GEval,
    KnowledgeRetention,
    Hallucination,
    BertScore,
    METEOR,
)
from .graph.ragplot import RagVisualizer
import warnings

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)
logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*`resume_download` is deprecated.*")
class RagEvaluator:
    """
    The RagEvaluator class is designed to evaluate various aspects of language model outputs using specified metrics.

    It supports metrics such as Faithfulness, Answer Relevancy, Contextual Relevancy, GEval, Hallucination,
    Knowledge Retention, BertScore, and METEOR.
    """

    def __init__(self, llm_as_judge, llm_response, retrieval_context, query):
        """
        Initializes the RagEvaluator with a language model and a list of metrics.

        Args:
            llm_as_judge: The language model.
            llm_response: The response from the language model.
            retrieval_context: The context retrieved for the query.
            query: The input query.
        """
        self.model = llm_as_judge
        self.metrics = [
            Faithfulness(llm_response=llm_response, retrieval_context=retrieval_context),
            AnswerRelevancy(query=query, llm_response=llm_response),
            ContextualRelevancy(query=query, retrieval_context=retrieval_context),
            GEval(parameters="Rag Pipeline", llm_response=llm_response, query=query, retrieval_context=retrieval_context),
            Hallucination(llm_response=llm_response, retrieval_context=retrieval_context),
            KnowledgeRetention(messages=[{"query": query, "llm_response": llm_response}]),
            BertScore(llm_response=llm_response, retrieval_context=retrieval_context),
            METEOR(llm_response=llm_response, retrieval_context=retrieval_context),
        ]
        logger.info("RagEvaluator initialized with model and metrics.")
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
                elif isinstance(metric, ContextualRelevancy):
                    irrelevancies = metric.get_irrelevancies(metric.query, metric.retrieval_contexts)
                    metric.set_irrelevancies(irrelevancies)
                    verdicts = metric.get_verdicts(metric.query, metric.retrieval_contexts)
                    score = 1.0 if not irrelevancies else max(0, 1.0 - len(irrelevancies) / len(metric.retrieval_contexts))
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
                elif isinstance(metric, BertScore):
                    score = metric.measure()
                    results['BertScore'] = {
                        'precision': score['Precision'],
                        'recall': score['Recall'],
                        'f1_score': score['F1-score']
                    }
                    self.metrics_score['precision'] = score['Precision']
                    self.evaluation_score += score['Precision']
                    self.metrics_score['recall'] = score['Recall']
                    self.evaluation_score += score['Recall']
                    self.metrics_score['f1_score'] = score['F1-score']
                    self.evaluation_score += score['F1-score']
                elif isinstance(metric, METEOR):
                    score = metric.measure()
                    results["METEOR"] = {"score": score}
                    self.evaluation_score += score
                    self.metrics_score["METEOR"] = score

                logger.info(f"Completed evaluation for metric: {metric_name}")
            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {str(e)}")
        return results

    def plot(self):
        visualizer = RagVisualizer(scores=self.metrics_score)#, score=self.evaluation_score)
        return visualizer.get_dashboard_html()