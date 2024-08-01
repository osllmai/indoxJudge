import plotly.express as px
import plotly.graph_objects as go
import json
class VisualizeAnalysis:
    """
    Visualize the analysis scores using bar and radar charts.

    Parameters
    ----------
    categories : list of str
        The categories for which scores are visualized.
    scores : dict
        The scores for each category.

    Returns
    -------
    None
    """
    def __init__(self, scores, model_name):
        self.scores = scores
        self.model_name = model_name

    def bar_chart(self, filename):
        fig_bar = px.bar(
            x=list(self.scores.keys()),
            y=list(self.scores.values()),
            labels={'x': 'Category', 'y': 'Score'},
            title='Analysis Scores by Category'
        )
        fig_bar.update_layout(
            yaxis_range=[0, 1],
            legend_title_text=f"Model: {self.model_name}",
            legend=dict(x=0, y=-0.2)
        )
        fig_bar.write_html(filename)
        fig_bar.show()

    def radar_chart(self, filename):
        categories_radar = list(self.scores.keys())
        values_radar = list(self.scores.values())

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=values_radar + [values_radar[0]],
            theta=categories_radar + [categories_radar[0]],
            fill='toself',
            name='Scores'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            title='Radar Chart of Analysis Scores',
            legend=dict(
                title=f'Model: {self.model_name}',
                x=0,
                y=-0.2
            )
        )

        fig_radar.write_html(filename)
        fig_radar.show()