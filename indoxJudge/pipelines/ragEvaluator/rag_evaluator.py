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

    Args:
        llm_as_judge: The language model to be used for evaluation.
        entries (dict): A dictionary of entries to evaluate. Each entry must be a dictionary with the following keys:
            - 'llm_response': The response from the language model.
            - 'retrieval_context': A list of strings representing the context retrieved for the query.
            - 'query': The input query.
            - 'ground_truth': The expected answer (ground truth).
            - 'context': The context or domain of the query.
        llm_response (str): The response from the language model for single evaluation.
        retrieval_context (list): The context retrieved for the query in a single evaluation.
        query (str): The input query for single evaluation.
        ground_truth (str): The expected answer (ground truth) for single evaluation.
        context (str): The context or domain of the query for single evaluation.

    Example:
        Sample entries for evaluation:

        entries = {
            "entry1": {
                "llm_response": "The Eiffel Tower is in Paris, France.",
                "retrieval_context": [
                    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."],
                "query": "Where is the Eiffel Tower located?",
                "ground_truth": "Paris, France",
                "context": "Geography"
            },
            "entry2": {
                "llm_response": "The Great Wall of China is in India.",
                "retrieval_context": [
                    "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the northern borders of China."],
                "query": "Where is the Great Wall of China located?",
                "ground_truth": "China",
                "context": "History"
            },
            "entry3": {
                "llm_response": "Python is a programming language that was created by Guido van Rossum and first released in 1991.",
                "retrieval_context": [
                    "Python is a high-level, interpreted programming language known for its readability and versatility.",
                    "It was created by Guido van Rossum and first released in 1991."],
                "query": "Who created Python, and when was it first released?",
                "ground_truth": "Guido van Rossum, 1991",
                "context": "Computer Science"
            }
        }

    Raises:
        ValueError: If both entries and individual evaluation parameters are provided or if none are provided.
    """

    def __init__(self, llm_as_judge, entries=None, llm_response=None, retrieval_context=None, query=None,
                 ground_truth=None, context=None):
        """
        Initializes the RagEvaluator with a language model and a list of metrics.
        """
        self.model = llm_as_judge
        self.entries = entries
        self.metrics_score = {}
        self.results = {}

        # Ensure only entries or individual parameters are provided
        if entries and (llm_response or retrieval_context or query or ground_truth or context):
            raise ValueError("Provide either 'entries' or individual parameters for evaluation, not both.")
        if not entries and not (llm_response and retrieval_context and query and ground_truth and context):
            raise ValueError("Either 'entries' or all individual parameters for a single evaluation must be provided.")

        # If entries are provided, we will evaluate them individually
        if entries:
            self.entries_results = {}
        else:
            # For single entry evaluation
            self._initialize_metrics(llm_response, retrieval_context, query, ground_truth, context)

    def _initialize_metrics(self, llm_response, retrieval_context, query, ground_truth, context):
        """
        Initializes the metrics with the provided data for evaluation.
        """
        retrieval_context_join = "\n".join(retrieval_context)

        self.metrics = [
            Faithfulness(llm_response=llm_response, retrieval_context=retrieval_context_join),
            AnswerRelevancy(query=query, llm_response=llm_response),
            ContextualRelevancy(query=query, retrieval_context=retrieval_context),
            GEval(parameters="Rag Pipeline", llm_response=llm_response, query=query,
                  retrieval_context=retrieval_context, ground_truth=ground_truth, context=context),
            Hallucination(llm_response=llm_response, retrieval_context=retrieval_context),
            KnowledgeRetention(messages=[{"query": query, "llm_response": llm_response}]),
            BertScore(llm_response=llm_response, retrieval_context=retrieval_context),
            METEOR(llm_response=llm_response, retrieval_context=retrieval_context),
        ]
        self._set_model_for_metrics()

    def _set_model_for_metrics(self):
        """
        Sets the language model for each metric that requires it.
        """
        for metric in self.metrics:
            if hasattr(metric, 'set_model'):
                metric.set_model(self.model)

    def _evaluate(self):
        """
        Evaluates the language model using the provided metrics and returns the results.

        Returns:
            dict: A dictionary containing the evaluation results for each metric.
        """
        results = {}
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            try:
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
                    self._calculate_metric_score(parameter="Faithfulness", score=score)
                elif isinstance(metric, AnswerRelevancy):
                    score = metric.measure()
                    results['AnswerRelevancy'] = {
                        'score': score,
                        'reason': metric.reason,
                        'statements': metric.statements,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts]
                    }
                    self._calculate_metric_score(parameter="AnswerRelevancy", score=score)
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
                    self._calculate_metric_score(parameter="ContextualRelevancy", score=score)
                elif isinstance(metric, GEval):
                    geval_result = metric.g_eval()
                    results['GEval'] = geval_result.replace("\n", " ")
                    geval_data = json.loads(results["GEval"])
                    score = int(geval_data["score"]) / 10
                    self._calculate_metric_score(parameter="GEval", score=score)
                elif isinstance(metric, Hallucination):
                    score = metric.measure()
                    results['Hallucination'] = {
                        'score': score,
                        'reason': metric.reason,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts]
                    }
                    self._calculate_metric_score(parameter="Hallucination", score=score)
                elif isinstance(metric, KnowledgeRetention):
                    score = metric.measure()
                    results['KnowledgeRetention'] = {
                        'score': score,
                        'reason': metric.reason,
                        'verdicts': [verdict.dict() for verdict in metric.verdicts],
                        'knowledges': [knowledge.data for knowledge in metric.knowledges]
                    }
                    self._calculate_metric_score(parameter="KnowledgeRetention", score=score)
                elif isinstance(metric, BertScore):
                    score = metric.measure()
                    results['BertScore'] = {
                        'precision': score['Precision'],
                        'recall': score['Recall'],
                        'f1_score': score['F1-score']
                    }
                    self._calculate_metric_score(parameter="precision", score=score['Precision'])
                    self._calculate_metric_score(parameter="recall", score=score['Recall'])
                    self._calculate_metric_score(parameter="f1_score", score=score['F1-score'])
                elif isinstance(metric, METEOR):
                    score = metric.measure()
                    results["METEOR"] = {"score": score}
                    self._calculate_metric_score(parameter="METEOR", score=score)
            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {str(e)}")

        return results

    def _calculate_metric_score(self, parameter, score):
        """
        Accumulates the scores for each metric.
        If the metric is not initialized, it sets the initial score.
        If the metric is already initialized, it adds the score to the existing score.

        Args:
            parameter (str): The name of the metric.
            score (float): The score to be added.
        """
        if parameter not in self.metrics_score:
            self.metrics_score[parameter] = 0  # Initialize the metric score to 0 if it doesn't exist

        self.metrics_score[parameter] += score  # Add the score to the existing score

    def _finalize_metric_scores(self):
        """
        Finalizes the metric scores by dividing each score by the number of entries
        and rounding to two decimal places.
        """
        num_entries = len(self.entries) if self.entries else 1  # Ensure no division by zero

        for parameter in self.metrics_score:
            self.metrics_score[parameter] = round(self.metrics_score[parameter] / num_entries, 2)

    def judge(self):
        """
        Evaluates the entries or single response using the initialized metrics.

        Returns:
            dict: A dictionary with the evaluation results for each entry or single response.
        """
        results = {}
        logger.info("Model set for all metrics.")
        logger.info("RagEvaluator initialized with model and metrics.")
        logger.info(f"Evaluation Began...")

        if not self.entries:
            single_result = self._evaluate()
            self.results = single_result
            self._finalize_metric_scores()  # Finalize scores after evaluation
            evaluation_score = self._evaluation_score_rag_mcda()
            self.metrics_score["evaluation_score"] = evaluation_score
            self.results['evaluation_score'] = evaluation_score
            return self.results
        elif self.entries:
            for entry_id, entry_data in self.entries.items():
                logger.info(f"Evaluating entry: {entry_id}")
                llm_response = entry_data['llm_response']
                retrieval_context = entry_data['retrieval_context']
                query = entry_data['query']
                ground_truth = entry_data.get('ground_truth')
                context = entry_data.get('context')

                # Initialize metrics for this entry
                self._initialize_metrics(llm_response, retrieval_context, query, ground_truth, context)

                # Evaluate the entry and store results
                entry_result = self._evaluate()
                results[entry_id] = entry_result  # Store result under unique entry key
                logger.info(f"Completed evaluation for entry: {entry_id}")

            self.results = results
            self._finalize_metric_scores()  # Finalize scores after all evaluations
            logger.info(f"Evaluation Completed, Check out the results")
            return self.results

    def _evaluation_score_rag_mcda(self):
        """
        Computes the final evaluation score using Multiple Criteria Decision Analysis (MCDA).

        Returns:
            float: The final evaluation score.
        """
        from skcriteria import mkdm
        from skcriteria.madm import simple

        evaluation_metrics = self.metrics_score.copy()
        if "evaluation_score" in evaluation_metrics:
            del evaluation_metrics['evaluation_score']
        # Transform the values for Hallucination (lower is better)
        evaluation_metrics['Hallucination'] = 1 - evaluation_metrics['Hallucination']

        # Weights for each metric (adjusted for RAG evaluation)
        weights = {
            'Faithfulness': 0.2,
            'AnswerRelevancy': 0.15,
            'ContextualRelevancy': 0.15,
            'GEval': 0.1,
            'Hallucination': 0.15,
            'KnowledgeRetention': 0.1,
            'precision': 0.05,
            'recall': 0.05,
            'f1_score': 0.05,
            'METEOR': 0.05,
        }

        # Convert metrics and weights to lists
        metric_values = list(evaluation_metrics.values())
        weight_values = list(weights.values())

        # Create decision matrix
        dm = mkdm(
            matrix=[metric_values],
            objectives=[max] * len(metric_values),  # All are maximization since we adjusted the values
            weights=weight_values,
            criteria=list(evaluation_metrics.keys())
        )

        # Apply Simple Additive Weighting (SAW) method
        saw = simple.WeightedSumModel()
        rank = saw.evaluate(dm)
        final_score_array = rank.e_['score']

        # Return the rounded final score
        return round(final_score_array.item(), 2)

    def plot(self, mode="external"):
        """
        Plots the evaluation results.

        Args:
            mode (str): The mode for plotting. Default is "external".
        """
        from indoxJudge.graph import Visualization
        from indoxJudge.utils import create_model_dict
        metrics = self.metrics_score.copy()
        del metrics['evaluation_score']
        score = self.metrics_score['evaluation_score']
        graph_input = create_model_dict(name="RAG Evaluator", metrics=metrics,
                                        score=score)
        visualizer = Visualization(data=graph_input, mode="rag")
        return visualizer.plot(mode=mode)

    def format_for_analyzer(self, name):
        """
        Formats the results for analysis.

        Args:
            name (str): The name of the analysis.
        """
        from indoxJudge.utils import create_model_dict
        metrics = self.metrics_score.copy()
        del metrics['evaluation_score']
        score = self.metrics_score['evaluation_score']
        analyzer_input = create_model_dict(name=name, score=score, metrics=metrics)
        return analyzer_input
