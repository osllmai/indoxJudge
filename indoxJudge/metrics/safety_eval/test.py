from Harmfulness_evaluation.harmfulness import Analysis as HarmfulnessAnalysis, EvaluateResponse as HarmfulnessEvaluateResponse
from Fairness_evaluation.fairness import Analysis as FairnessAnalysis, EvaluateResponse as FairnessEvaluateResponse
from Privacy_evaluation.privacy import Analysis as PrivacyAnalysis, EvaluateResponse as PrivacyEvaluateResponse
from Jailbreak_evaluation.jailbreak import Analysis as JailbreakAnalysis, EvaluateResponse as JailbreakEvaluateResponse
from Robustness_evaluation.robustness import Analysis as RobustnessAnalysis, EvaluateResponse as RobustnessEvaluateResponse
import plotly.express as px
import plotly.graph_objects as go
import json
from visualization import VisualizeAnalysis

def analyze_and_evaluate(analysis_class, evaluate_class, text, category):
    """
    Analyze and evaluate the given text using the specified analysis and evaluation classes.

    Parameters
    ----------
    analysis_class : class
        The class used for analyzing text.
    evaluate_class : class
        The class used for evaluating the analysis response.
    text : str
        The text to be analyzed.
    category : str
        The category being tested.

    Returns
    -------
    dict
        The results of the analysis and evaluation.
    """
    analysis = analysis_class()
    result = analysis.analyze(text)
    result_json = json.loads(result)
    evaluator = evaluate_class(result)
    return {
        'result_json': result_json,
        'score': evaluator.score(),
        'feedback': evaluator.feedback()
    }

def main():
    """
    Main function to run the analysis based on user input, save results, and generate visualizations.
    """

    #text = input("Enter the paragraph for analysis: ").strip()
    text = "The development team encountered complex ethical challenges while designing the AI system, particularly in terms of balancing user privacy with the need for comprehensive data collection. They were tasked with ensuring that the system’s capabilities did not compromise ethical standards, which required careful consideration of how to handle and protect sensitive information."

    all_metrics = ["fairness", "privacy", "jailbreak", "robustness", "harmfulness"]
    print("Available metrics:", ", ".join(all_metrics))
    #selected_metrics = input("Enter the metrics to analyze (comma-separated): ").strip().split(',')
    selected_metrics=all_metrics
    selected_metrics = [metric.strip().lower() for metric in selected_metrics if metric.strip().lower() in all_metrics]
    
    if not selected_metrics:
        print("No valid metrics selected. Exiting.")
        return

    metric_mapping = {
        'fairness': (FairnessAnalysis, FairnessEvaluateResponse),
        'privacy': (PrivacyAnalysis, PrivacyEvaluateResponse),
        'jailbreak': (JailbreakAnalysis, JailbreakEvaluateResponse),
        'robustness': (RobustnessAnalysis, RobustnessEvaluateResponse),
        'harmfulness': (HarmfulnessAnalysis, HarmfulnessEvaluateResponse)
    }

    results = {}
    for metric in selected_metrics:
        if metric in metric_mapping:
            analysis_class, evaluate_class = metric_mapping[metric]
            result = analyze_and_evaluate(analysis_class, evaluate_class, text, metric)
            results[metric] = result
        else:
            print(f"Metric '{metric}' is not recognized.")
    
    for metric, result in results.items():
        print(f"{metric.capitalize()} Analysis:")
        print(f"Score: {result['score']}")
        print(f"Explanation: {result['feedback']}")
        print()

    output_filename = "Analysis_Results.json"
    with open(output_filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    print(f"Results saved to {output_filename}")

    scores = {metric: result['score'] for metric, result in results.items()}
    model_name = "gpt-3.5-turbo-0125"  

    visualizer = VisualizeAnalysis(scores, model_name)

    visualizer.bar_chart('Analysis_Scores_Plotly.html')

    visualizer.radar_chart('Radar_Analysis_Scores_Plotly.html')

if __name__ == "__main__":
    main()