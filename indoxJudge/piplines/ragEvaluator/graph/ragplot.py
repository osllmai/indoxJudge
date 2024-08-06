import dash
from dash import  html #dcc
# from dash.dependencies import Input, Output
import plotly.graph_objects as go
import statistics
import plotly.io as pio
import openai

class RagVisualizer:
    """
    A class for visualizing metric scores using various chart types.

    This class takes a dictionary of metric scores and provides methods to create
    bar charts, radar charts, and gauge charts to visualize the data. It also
    generates HTML content for displaying these charts with explanatory notes.

    Attributes:
        scores (dict): A dictionary of metric names and their corresponding scores.
        metrics (list): The names of the metrics.
        values (list): The scores of the metrics.
        overall_score (float): The mean score across all metrics.
    """

    def __init__(self, scores):
        """
        Initialize the MetricVisualizer with a dictionary of scores.

        Args:
            scores (dict): A dictionary where keys are metric names and values are scores.
        """
        self.scores = scores
        self.metrics = list(self.scores.keys())
        self.values = list(self.scores.values())
        self.overall_score = statistics.mean(self.values)

    def create_note_box(self, text, width="30%", height="320px"):
        """
        Create a note box with the given text.

        Args:
            text (str): The text to display in the note box.
            width (str): The width of the note box. Default is '30%'.
            height (str): The height of the note box. Default is '320px'.

        Returns:
            dash.html.Div: A Div component containing the note box.
        """
        return html.Div(
            [
                html.P(
                    text,
                    style={
                        "margin": "0",
                        "padding": "10px",
                        "textAlign": "justify",
                        "textJustify": "inter-word",
                    },
                )
            ],
            style={
                "width": width,
                "height": height,
                "overflowY": "auto",
                "border": "1px solid black",
                "borderRadius": "5px",
                "backgroundColor": "white",
                "position": "absolute",
                "right": "50px",
                "top": "50px",
                "fontSize": "14px",
                "lineHeight": "1.4",
            },
        )

    def create_bar_chart(self):
        """
        Create a bar chart of the metric scores.

        Returns:
            plotly.graph_objs._figure.Figure: A Plotly Figure object representing the bar chart.
        """
        fig = go.Figure(
            data=[
                go.Bar(
                    x=self.metrics,
                    y=self.values,
                    text=[f"{v:.2f}" for v in self.values],
                    textposition="auto",
                    marker_color="skyblue",
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
            height=500,
            width=1000,
            margin=dict(l=50, r=50, t=50, b=100),
            autosize=False,
            xaxis=dict(domain=[0, 0.6]),
        )

        return fig

    def create_radar_chart(self):
        """
        Create a radar chart of the metric scores.

        Returns:
            plotly.graph_objs._figure.Figure: A Plotly Figure object representing the radar chart.
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=self.values + [self.values[0]],
                theta=self.metrics + [self.metrics[0]],
                fill="toself",
                marker_color="skyblue",
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                domain=dict(x=[0, 0.7], y=[0, 1]),
            ),
            showlegend=False,
            title="Radar Chart of Metrics",
            template="plotly_white",
            height=500,
            width=1000,
            margin=dict(l=50, r=50, t=50, b=50),
            autosize=False,
        )

        return fig

    def create_gauge_chart(self):
        """
        Create a gauge chart of the overall metric score.

        Returns:
            plotly.graph_objs._figure.Figure: A Plotly Figure object representing the gauge chart.
        """
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(self.overall_score, 2),
                title={"text": "Average Score"},
                domain={"x": [0, 0.5], "y": [0, 1]},
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
            height=500,
            width=1000,
            margin=dict(l=50, r=50, t=50, b=50),
            autosize=False,
        )
        return fig

    def figure_to_html(self, fig):
        """
        Convert a Plotly Figure object to HTML.

        Args:
            fig (plotly.graph_objs._figure.Figure): The Plotly Figure to convert.

        Returns:
            str: HTML representation of the Figure.
        """
        return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    
    def get_chart_description(self, chart_type, word_limit=50):
        """
        Get a description for a chart from ChatGPT.

        Args:
            chart_type (str): The type of chart (bar, radar, or gauge).
            word_limit (int): The maximum number of words for the description.

        Returns:
            str: A description of the chart.
        """
        openai.api_key = 'your-api-key-here'

        if chart_type == 'bar':
            prompt = f"Describe a bar chart comparing these metrics: {', '.join(self.metrics)}. Values range from 0 to 1. Limit the description to {word_limit} words."
        elif chart_type == 'radar':
            prompt = f"Describe a radar chart showing these metrics: {', '.join(self.metrics)}. Values range from 0 to 1. Limit the description to {word_limit} words."
        elif chart_type == 'gauge':
            prompt = f"Describe a gauge chart showing the overall score (average of {', '.join(self.metrics)}). The score is {self.overall_score:.2f}. Limit the description to {word_limit} words."
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that describes data visualizations."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()
    
    def generate_html(self):
        """
        Generate HTML content for the dashboard including all charts and notes.

        Returns:
            str: HTML content for the complete dashboard.
        """
        bar_chart_html = self.figure_to_html(self.create_bar_chart())
        radar_chart_html = self.figure_to_html(self.create_radar_chart())
        gauge_chart_html = self.figure_to_html(self.create_gauge_chart())

        bar_note = self.get_chart_description('bar')
        radar_note = self.get_chart_description('radar')
        gauge_note = self.get_chart_description('gauge')

        html_content = f"""
        <html>
        <head>
            <title>Metrics Visualization</title>
            <style>
                .chart-container {{ 
                    position: relative;
                    margin-bottom: 50px; 
                    height: 500px;
                }}
                .note-box {{
                    width: 30%;
                    height: 300px;
                    overflow-y: auto;
                    border: 1px solid black;
                    border-radius: 5px;
                    background-color: white;
                    position: absolute;
                    right: 70px;
                    top: 70px;
                    font-size: 14px;
                    line-height: 1.4;
                    padding: 10px;
                    text-align: justify;
                    text-justify: inter-word;
                    color: #000000; 
                }}
            </style>
        </head>
        <body>
            <h1>Metrics Visualization</h1>
            <div class="chart-container">
                <h2>Bar Chart</h2>
                {bar_chart_html}
                <div class="note-box">{bar_note}</div>
            </div>
            <div class="chart-container">
                <h2>Radar Chart</h2>
                {radar_chart_html}
                <div class="note-box">{radar_note}</div>
            </div>
            <div class="chart-container">
                <h2>Gauge Chart</h2>
                {gauge_chart_html}
                <div class="note-box">{gauge_note}</div>
            </div>
        </body>
        </html>
        """
        return html_content

    def get_dashboard_html(self):
        """
        Get the complete HTML content for the dashboard.

        Returns:
            str: HTML content for the complete dashboard.
        """
        return self.generate_html()
