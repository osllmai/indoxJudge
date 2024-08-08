import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

class MetricsVisualizer:
    def __init__(self, metrics, score):
        self.metrics = metrics
        self.score = score
        self.template = "ggplot2"  # Choosing ggplot2 theme for all plots

    def create_radar_chart(self):
        filtered_metrics = {k: v for k, v in self.metrics.items() if v != 0}
        labels = list(filtered_metrics.keys())
        values = list(filtered_metrics.values())
        values += values[:1]
        labels += labels[:1]

        fig = go.Figure(
            data=[
                go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill='toself',
                    name='Metrics'
                )
            ],
            layout=go.Layout(
                title='Evaluation Metrics Radar Chart',
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                template=self.template
            )
        )
        return fig

    def create_bar_chart(self):
        fig = go.Figure(data=[
            go.Bar(
                x=list(self.metrics.keys()),
                y=list(self.metrics.values()),
                text=list(self.metrics.values()),
                textposition='auto',
                hoverinfo='x+y',
                marker_color='skyblue'
            )
        ])

        fig.update_layout(
            title='Evaluation Metrics Bar Chart',
            xaxis_title='Metrics',
            yaxis_title='Values',
            xaxis_tickangle=-45,
            template=self.template
        )
        return fig

    def create_gauge_chart(self):
        average_score = sum(self.metrics.values()) / len(self.metrics)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=self.score,
            title={'text': "Overall Evaluation Score", 'font': {'size': 24}},
            delta={'reference': 0.5, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 0.7, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.2], 'color': "red"},
                    {'range': [0.2, 0.4], 'color': "orange"},
                    {'range': [0.4, 0.6], 'color': "yellow"},
                    {'range': [0.6, 0.8], 'color': "lightgreen"},
                    {'range': [0.8, 1], 'color': "green"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': self.score}
            }
        ))

        fig.add_trace(go.Indicator(
            mode="number",
            value=average_score,
            title={"text": "Average Score", 'font': {'size': 20}},
            domain={'x': [0.75, 1], 'y': [0.75, 1]}
        ))

        fig.update_layout(
            title='Overall Evaluation Score',
            font={'color': "darkblue", 'family': "Arial"},
            paper_bgcolor="white",
            plot_bgcolor="white",
            template=self.template
        )
        return fig

    def create_scatter_plot(self):
        df = pd.DataFrame(list(self.metrics.items()), columns=['Metric', 'Value'])
        fig = px.scatter(df, x='Value', y='Metric', size='Value', color='Value', color_continuous_scale='Viridis')

        fig.update_layout(
            title='Evaluation Metrics Scatter Plot',
            xaxis_title='Value',
            yaxis_title='Metric',
            template=self.template
        )
        return fig

    def plot(self):
        radar_chart = self.create_radar_chart()
        bar_chart = self.create_bar_chart()
        scatter_plot = self.create_scatter_plot()
        gauge_chart = self.create_gauge_chart()

        tab1 = widgets.Output()
        tab2 = widgets.Output()
        tab3 = widgets.Output()
        tab4 = widgets.Output()

        with tab1:
            display(radar_chart)

        with tab2:
            display(bar_chart)

        with tab3:
            display(scatter_plot)

        with tab4:
            display(gauge_chart)

        tabs = widgets.Tab(children=[tab1, tab2, tab3, tab4])
        tabs.set_title(0, 'Radar Chart')
        tabs.set_title(1, 'Bar Chart')
        tabs.set_title(2, 'Scatter Plot')
        tabs.set_title(3, 'Gauge Chart')

        display(tabs)


