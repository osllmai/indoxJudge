class EvaluationAnalyzer:
    def __init__(self, models):
        self.models = models

    def plot(self, mode="external"):
        from indoxJudge.graph import Visualization
        visualization = Visualization(data=self.models, mode="llm")
        return visualization.plot(mode=mode)


