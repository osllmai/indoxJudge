from loguru import logger
import sys
from indoxJudge.metrics import (Faithfulness, AnswerRelevancy, Bias, Gruen, Rouge,
                                KnowledgeRetention, BLEU, Hallucination, Toxicity, BertScore)

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")
logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class LLMEvaluator:
    """
    The Evaluator class is designed to evaluate various aspects of language model outputs using specified metrics.

    It supports metrics such as Faithfulness, Answer Relevancy, Bias, Contextual Relevancy, GEval, Hallucination,
    Knowledge Retention, Toxicity, BertScore, BLEU, and Gruen.
    """

    def __init__(self, llm_as_judge, llm_response, retrieval_context, query):
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

            # Rouge(llm_response=llm_response, retrieval_context=retrieval_context),
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
                logger.info(f"Completed evaluation for metric: {metric_name}")

            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {str(e)}")
        return results

    def plot(self, mode="external"):
        from indoxJudge.graph import Visualization
        from indoxJudge.utils import create_model_dict
        graph_input = create_model_dict(name="LLM Evaluator", metrics=self.metrics_score,
                                        score=self.evaluation_score / 9)
        visualizer = Visualization(data=graph_input, mode="llm")
        return visualizer.plot(mode=mode)
