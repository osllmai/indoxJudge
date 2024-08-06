import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import base64
from IPython.display import HTML, display
import ipywidgets as widgets

class Visualizer:
    """
    A class to visualize various evaluation metrics through bar charts and radar charts.

    Attributes:
    ----------
    scores : dict
        A dictionary containing metric scores.

    Methods:
    -------
    calculate_averages():
        Calculates the average of F1, Recall, and Precision for harmonic metrics.

    plot_to_html(fig):
        Converts a Matplotlib figure to an HTML image for display.

    combined_bar_chart():
        Generates an HTML representation of a bar chart comparing harmonic and classic metrics.

    radar_chart():
        Generates an HTML representation of a radar chart displaying various metrics.

    display_visualizations():
        Displays the bar chart and radar chart in a tabbed widget.
    """

    def __init__(self, scores):
        """
        Initializes the Visualizer with the given scores.

        Parameters:
        ----------
        scores : dict
            A dictionary containing metric scores.
        """
        self.scores = scores
        self.calculate_averages()

    def calculate_averages(self):
        """
        Calculates the average scores for F1, Recall, and Precision metrics.
        These are combined as harmonic metrics.
        """
        self.avg_f1 = (self.scores['F1'] + self.scores['Bert_f1']) / 2
        self.avg_recall = (self.scores['Recall'] + self.scores['Bert_recall']) / 2
        self.avg_precision = (self.scores['Precision'] + self.scores['Bert_precision']) / 2

    def plot_to_html(self, fig):
        """
        Converts a Matplotlib figure to an HTML image for embedding in an HTML page.

        Parameters:
        ----------
        fig : matplotlib.figure.Figure
            The Matplotlib figure to convert.

        Returns:
        -------
        str
            A string containing the HTML representation of the image.
        """
        canvas = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        plt.close(fig)  # Close the figure to prevent inline display
        return f"<img src='data:image/png;base64,{data}'/>"

    def combined_bar_chart(self):
        """
        Generates a bar chart comparing harmonic and classic metrics.

        Returns:
        -------
        str
            An HTML string containing the bar chart.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['F1', 'Recall', 'Precision', 'BLEU', 'METEOR', 'Gruen']
        harmonic_values = [self.avg_f1, self.avg_recall, self.avg_precision]
        classic_values = [self.scores['BLEU'], self.scores['METEOR'], self.scores['Gruen']]

        index = np.arange(len(metrics))
        bar_width = 0.35

        ax.bar(index - bar_width/2, harmonic_values + [0, 0, 0], bar_width, label='Harmonic Metrics', color='skyblue')

        ax.bar(index + bar_width/2, [0, 0, 0] + classic_values, bar_width, label='Classic Metrics', color='lightgreen')

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Metrics Comparison')
        ax.set_xticks(index)
        ax.set_xticklabels(metrics)
        ax.legend()

        ax.grid(axis='y', linestyle='--', alpha=0.7)

        for i, v in enumerate(harmonic_values + classic_values):
            if v != 0:
                ax.text(i - bar_width / 2, v + 0.02, f"{v:.2f}", ha='center')

        plt.tight_layout()
        return self.plot_to_html(fig)

    def radar_chart(self):
        """
        Generates a radar chart displaying various metrics.

        Returns:
        -------
        str
            An HTML string containing the radar chart.
        """
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
        metrics = ['F1', 'Recall', 'Precision', 'BLEU', 'METEOR', 'Gruen']
        values = [self.avg_f1, self.avg_recall, self.avg_precision,
                  self.scores['BLEU'], self.scores['METEOR'], self.scores['Gruen']]
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title('Chart of Metrics')
        plt.tight_layout()
        return self.plot_to_html(fig)

    def display_visualizations(self):
        """
        Displays the bar chart and radar chart in a tabbed widget for interactive exploration.
        """
        tab1_content = widgets.HTML(self.combined_bar_chart())
        tab2_content = widgets.HTML(self.radar_chart())

        tab = widgets.Tab()
        tab.children = [tab1_content, tab2_content]
        tab.set_title(0, 'Bar Chart')
        tab.set_title(1, 'Radar Chart')

        display(tab)