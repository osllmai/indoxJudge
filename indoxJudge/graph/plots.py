import urllib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from IPython.display import HTML
import ipywidgets as widgets
from IPython.display import display

# Set seaborn style for dark mode
sns.set_style("darkgrid")


class MetricVisualizer:
    def __init__(self, scores_dic):
        self.scores_dic = scores_dic
        self.metric_categories = {
            'accuracy_relevance': [
                'GEVal',
                'KnowledgeRetention',
                'Faithfulness',
                'ContextualRelevancy',
                'AnswerRelevancy'
            ],
            'safety_ethics': [
                'Toxicity',
                'Bias',
                'Hallucination'
            ],
        }
        self.colors = {
            'accuracy_relevance': 'royalblue',
            'safety_ethics': 'darkred',
        }
        self.hatches = {
            'safety_ethics': '//'
        }
        self.df = self.create_dataframe()

    def create_dataframe(self):
        data = [
            (metric, score, category)
            for category, metrics in self.metric_categories.items()
            for metric, score in self.scores_dic.items()
            if metric in metrics
        ]
        return pd.DataFrame(data, columns=['Metric', 'Score', 'Category'])

    def plot_to_html(self, fig):
        """Convert a Matplotlib figure to HTML image without displaying it."""
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read()).decode('utf-8')
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)
        plt.close(fig)
        return f'<img src="{uri}" style="height: 550px;">'

    def plot_bar_charts(self):
        fig, axes = plt.subplots(1, len(self.metric_categories), figsize=(10, 6), sharey=True)
        for i, (category, group) in enumerate(self.df.groupby('Category')):
            sns.barplot(x='Metric', y='Score', data=group, ax=axes[i], color=self.colors[category],
                        hatch=self.hatches.get(category, None))
            axes[i].set_title(category.replace('_', ' ').title(), color='white')
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=60, colors='white')
            axes[i].tick_params(axis='y', colors='white')
            axes[i].set_xlabel('', color='white')
        plt.suptitle('Bar Charts of Evaluation Metrics by Category', y=1.05, color='white')
        plt.subplots_adjust(top=0.8, hspace=0.3, wspace=0.3)
        fig.patch.set_facecolor('#2e2e2e')
        return self.plot_to_html(fig)

    def plot_radar_chart(self):
        labels = list(self.scores_dic.keys())
        values = list(self.scores_dic.values())
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='purple', alpha=0.25)
        ax.plot(angles, values, color='purple', linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10, color='teal', )
        plt.title('Evaluation Metrics Radar Chart', size=16, color='white')
        fig.patch.set_facecolor('#2e2e2e')
        return self.plot_to_html(fig)

    def plot_heatmap(self):
        metric_scores_df = pd.DataFrame.from_dict(self.scores_dic, orient='index', columns=['Score'])
        metric_scores_df = metric_scores_df.sort_values(by='Score', ascending=False)

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(metric_scores_df, annot=True, cmap='coolwarm', cbar_kws={'label': 'Metric Score'}, ax=ax,
                    annot_kws={"color": "white"})
        plt.title('Heatmap of Evaluation Metrics Scores', color='white')
        plt.yticks(rotation=45, color='white')
        plt.xticks(rotation=45, ha='right', color='white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        fig.patch.set_facecolor('#2e2e2e')
        return self.plot_to_html(fig)

    def show_all_plots(self):
        bar_chart_html = self.plot_bar_charts()
        radar_chart_html = self.plot_radar_chart()
        heatmap_html = self.plot_heatmap()

        # Define a consistent background color and text style for dark mode
        background_color = '#2e2e2e'
        text_color = 'white'

        # Explanations for each plot
        bar_chart_description = widgets.HTML(
            value="<div style='background-color: {bg_color}; color: {text_color}; height: 550px; padding: 10px;'>"
                  "<p>The bar charts display the scores of various evaluation metrics categorized into "
                  "'accuracy relevance' and 'safety ethics'. Each bar represents a metric, with the height of the bar "
                  "indicating the score value. This visualization allows for a quick comparison of metric scores across different categories.</p>"
                  "</div>".format(bg_color=background_color, text_color=text_color)
        )

        radar_chart_description = widgets.HTML(
            value="<div style='background-color: {bg_color}; color: {text_color}; height: 550px; padding: 10px;'>"
                  "<p>The radar chart provides a comprehensive view of the scores across multiple metrics. "
                  "Each axis represents a metric, and the distance from the center indicates the score value. "
                  "This chart helps in identifying strengths and areas for improvement by comparing scores across different metrics.</p>"
                  "</div>".format(bg_color=background_color, text_color=text_color)
        )

        heatmap_description = widgets.HTML(
            value="<div style='background-color: {bg_color}; color: {text_color}; height: 550px; padding: 10px;'>"
                  "<p>The heatmap shows the scores for various evaluation metrics. Each cell represents the score of a metric, "
                  "with the color indicating the score level. Warmer colors (closer to red) indicate higher scores, while cooler colors "
                  "(closer to blue) indicate lower scores. This visualization helps in identifying strengths and areas for improvement based on the evaluation metrics.</p>"
                  "</div>".format(bg_color=background_color, text_color=text_color)
        )

        # Create HBox for each tab with plot and description
        bar_chart_tab_content = widgets.HBox([widgets.HTML(value=bar_chart_html), bar_chart_description],
                                             layout=widgets.Layout(align_items='stretch', height='550px'))
        radar_chart_tab_content = widgets.HBox([widgets.HTML(value=radar_chart_html), radar_chart_description],
                                               layout=widgets.Layout(align_items='stretch', height='550px'))
        heatmap_tab_content = widgets.HBox([widgets.HTML(value=heatmap_html), heatmap_description],
                                           layout=widgets.Layout(align_items='stretch', height='550px', width='100%'))

        tab_contents = [
            bar_chart_tab_content,
            radar_chart_tab_content,
            heatmap_tab_content
        ]

        tab = widgets.Tab()
        tab.children = tab_contents
        tab.set_title(0, 'Bar Charts')
        tab.set_title(1, 'Radar Chart')
        tab.set_title(2, 'Heatmap')

        display(tab)

#
