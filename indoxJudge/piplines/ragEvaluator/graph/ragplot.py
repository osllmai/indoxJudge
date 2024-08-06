import dash
from dash import html
import plotly.graph_objects as go
import statistics
import plotly.io as pio


class RagVisualizer:
    """
    A class for visualizing RAG (Retrieval-Augmented Generation) metric scores using various chart types.

    This class takes a dictionary of metric scores and provides methods to create
    bar charts and gauge charts to visualize the data. It also generates HTML content
    for displaying these charts.

    Attributes:
        scores (dict): A dictionary of metric names and their corresponding scores.
        metrics (list): The names of the metrics.
        values (list): The scores of the metrics.
        overall_score (float): The mean score across all metrics.
    """

    def __init__(self, scores):
        """
        Initialize the RagVisualizer with a dictionary of scores.

        Args:
            scores (dict): A dictionary where keys are metric names and values are scores.
        """
        self.scores = scores
        self.metrics = list(self.scores.keys())
        self.values = list(self.scores.values())
        self.overall_score = statistics.mean(self.values)

    def create_bar_chart(self):
        """
        Creates a bar chart of the metric scores.

        The bar chart displays metrics in descending order of their scores,
        with each bar colored differently using a gradient scale.

        Returns:
            plotly.graph_objs._figure.Figure: A Plotly Figure object representing the bar chart.
        """
        # Sort metrics and values in descending order
        sorted_data = sorted(zip(self.metrics, self.values), key=lambda x: x[1], reverse=True)
        sorted_metrics, sorted_values = zip(*sorted_data)

        # Generate a color scale
        colors = [f'rgb({255 - int(230 * i / len(sorted_metrics))}, {int(230 * i / len(sorted_metrics))}, 255)' for i in
                  range(len(sorted_metrics))]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=sorted_metrics,
                    y=sorted_values,
                    text=[f"{v:.2f}" for v in sorted_values],
                    textposition="auto",
                    marker_color=colors,
                    width=0.5,
                )
            ]
        )

        fig.update_layout(
            title="Metrics Comparison",
            xaxis_title="Metrics",
            yaxis_title="Score",
            template="plotly_white",
            xaxis_tickangle=-45,
            yaxis=dict(range=[0, 1]),
            height=300,
            width=600,
            margin=dict(l=50, r=50, t=50, b=100),
            autosize=False,
            font=dict(size=10)
        )

        return fig

    def create_gauge_chart(self):
        """
        Creates a gauge chart of the overall metric score.

        The gauge chart represents the average score across all metrics,
        with color-coded sections indicating different score ranges.

        Returns:
            plotly.graph_objs._figure.Figure: A Plotly Figure object representing the gauge chart.
        """
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(self.overall_score, 2),
                title={"text": "Average Score"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 0.3], "color": "red"},
                        {"range": [0.3, 0.7], "color": "yellow"},
                        {"range": [0.7, 1], "color": "green"},
                    ],
                },
            )
        )

        fig.update_layout(
            height=300,
            width=400,
            margin=dict(l=50, r=50, t=50, b=50),
            autosize=False,
            font=dict(size=10)
        )
        return fig

    def figure_to_html(self, fig):
        """
        Converts a Plotly Figure object to HTML.

        Args:
            fig (plotly.graph_objs._figure.Figure): The Plotly Figure to convert.

        Returns:
            str: HTML representation of the Figure.
        """
        return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    def generate_html(self):
        """
        Generates HTML content for the dashboard including all charts.

        This method creates both the bar chart and gauge chart, converts them to HTML,
        and wraps them in a styled HTML document.

        Returns:
            str: HTML content for the complete dashboard.
        """
        bar_chart_html = self.figure_to_html(self.create_bar_chart())
        gauge_chart_html = self.figure_to_html(self.create_gauge_chart())

        html_content = f"""
        <html>
        <head>
            <title>Metrics Visualization</title>
            <style>
                .chart-container {{ 
                    margin-bottom: 30px; 
                    display: inline-block;
                    vertical-align: top;
                }}
                .bar-chart {{
                    width: 600px;
                    height: 300px;
                }}
                .gauge-chart {{
                    width: 400px;
                    height: 300px;
                }}
            </style>
        </head>
        <body>
            <h1>Metrics Visualization</h1>
            <div class="chart-container bar-chart">
                <h2>Bar Chart</h2>
                {bar_chart_html}
            </div>
            <div class="chart-container gauge-chart">
                <h2>Gauge Chart</h2>
                {gauge_chart_html}
            </div>
        </body>
        </html>
        """
        return html_content

    def get_dashboard_html(self):
        """
        Gets the complete HTML content for the dashboard.

        This is a convenience method that calls generate_html().

        Returns:
            str: HTML content for the complete dashboard.
        """
        return self.generate_html()