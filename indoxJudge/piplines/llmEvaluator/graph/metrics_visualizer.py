import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
import plotly.express as px
# import pandas as pd
# from plotly.subplots import make_subplots
# from dash import Dash, html, dcc
# import dash_bootstrap_components as dbc
# from dash.dependencies import Output, Input
#
# class MetricsVisualizer:
#     def __init__(self, metrics, score):
#         self.metrics = metrics
#         self.score = score
#         self.template = "ggplot2"  # Choosing ggplot2 theme for all plots
#
#     def create_radar_chart(self):
#         filtered_metrics = {k: v for k, v in self.metrics.items() if v != 0}
#         labels = list(filtered_metrics.keys())
#         values = list(filtered_metrics.values())
#         values += values[:1]
#         labels += labels[:1]
#
#         fig = go.Figure(
#             data=[
#                 go.Scatterpolar(
#                     r=values,
#                     theta=labels,
#                     fill='toself',
#                     name='Metrics'
#                 )
#             ],
#             layout=go.Layout(
#                 title='Evaluation Metrics Radar Chart',
#                 polar=dict(
#                     radialaxis=dict(
#                         visible=True,
#                         range=[0, 1]
#                     )
#                 ),
#                 showlegend=False,
#                 template=self.template
#             )
#         )
#         return fig
#
#     def create_bar_chart(self):
#         fig = go.Figure(data=[
#             go.Bar(
#                 x=list(self.metrics.keys()),
#                 y=list(self.metrics.values()),
#                 text=list(self.metrics.values()),
#                 textposition='auto',
#                 hoverinfo='x+y',
#                 marker_color='skyblue'
#             )
#         ])
#
#         fig.update_layout(
#             title='Evaluation Metrics Bar Chart',
#             xaxis_title='Metrics',
#             yaxis_title='Values',
#             xaxis_tickangle=-45,
#             template=self.template
#         )
#         return fig
#
#     def create_gauge_chart(self):
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number+delta",
#             value=self.score,
#             title={'text': "Overall Evaluation Score", 'font': {'size': 24}},
#             delta={'reference': 0.5, 'increasing': {'color': "green"}},
#             gauge={
#                 'axis': {'range': [0, 1], 'tickwidth': 0.7, 'tickcolor': "darkblue"},
#                 'bar': {'color': "darkblue"},
#                 'bgcolor': "white",
#                 'borderwidth': 2,
#                 'bordercolor': "gray",
#                 'steps': [
#                     {'range': [0, 0.2], 'color': "red"},
#                     {'range': [0.2, 0.4], 'color': "orange"},
#                     {'range': [0.4, 0.6], 'color': "yellow"},
#                     {'range': [0.6, 0.8], 'color': "lightgreen"},
#                     {'range': [0.8, 1], 'color': "green"}],
#                 'threshold': {
#                     'line': {'color': "black", 'width': 4},
#                     'thickness': 0.75,
#                     'value': self.score}
#             }
#         ))
#
#         fig.update_layout(
#             title='Overall Evaluation Score',
#             font={'color': "darkblue", 'family': "Arial"},
#             paper_bgcolor="white",  # Ensure the background is white
#             plot_bgcolor="white",  # Ensure the plot background is white
#             template=self.template
#         )
#         return fig
#
#     def create_scatter_plot(self):
#         df = pd.DataFrame(list(self.metrics.items()), columns=['Metric', 'Value'])
#         fig = px.scatter(df, x='Value', y='Metric', size='Value', color='Value', color_continuous_scale='Viridis')
#
#         fig.update_layout(
#             title='Evaluation Metrics Scatter Plot',
#             xaxis_title='Value',
#             yaxis_title='Metric',
#             template=self.template
#         )
#         return fig
#
#     def plot(self):
#         radar_chart = self.create_radar_chart()
#         bar_chart = self.create_bar_chart()
#         scatter_plot = self.create_scatter_plot()
#         gauge_chart = self.create_gauge_chart()
#
#         tab1 = widgets.Output()
#         tab2 = widgets.Output()
#         tab3 = widgets.Output()
#         tab4 = widgets.Output()
#
#         with tab1:
#             display(radar_chart)
#
#         with tab2:
#             display(bar_chart)
#
#         with tab3:
#             display(scatter_plot)
#
#         with tab4:
#             display(gauge_chart)
#
#         tabs = widgets.Tab(children=[tab1, tab2, tab3, tab4])
#         tabs.set_title(0, 'Radar Chart')
#         tabs.set_title(1, 'Bar Chart')
#         tabs.set_title(2, 'Scatter Plot')
#         tabs.set_title(3, 'Gauge Chart')
#
#         display(tabs)
#
#
#
#

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input


class MetricsVisualizer:
    def __init__(self, metrics, score):
        self.metrics = metrics
        self.score = score
        self.light_template = "plotly_white"  # Template for light mode
        self.dark_template = "plotly_dark"  # Template for dark mode
        self.current_template = self.light_template

    def set_theme(self, theme):
        """
        Sets the theme for the visualizations.

        Parameters:
        theme (str): 'light' or 'dark' to set the respective theme.
        """
        if theme == 'dark':
            self.current_template = self.dark_template
        else:
            self.current_template = self.light_template

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
                template=self.current_template
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
            template=self.current_template
        )
        return fig

    def create_gauge_chart(self):
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

        fig.update_layout(
            title='Overall Evaluation Score',
            font={'color': "darkblue", 'family': "Arial"},
            paper_bgcolor="white",  # Ensure the background is white
            plot_bgcolor="white",  # Ensure the plot background is white
            template=self.current_template
        )
        return fig

    def plot(self, mode="external"):
        app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
        app.layout = html.Div([
            html.Link(href='/assets/style.css', rel='stylesheet'),
            dcc.Location(id='url', refresh=False),
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.H2("IndoxJudge", className="text-center my-4 display-4 text-custom-primary", id="title-text"), width=10),
                    dbc.Col(dbc.Switch(id="dark-mode-switch", label="Dark Mode", className="my-4"), width=2)
                ], align="center"),
                dbc.Row([
                    dbc.Col([
                        dbc.Nav([
                            dbc.NavItem(dbc.NavLink("Radar Chart", href="#radar-chart", className="nav-link", external_link=True, id="nav-radar-chart")),
                            dbc.NavItem(dbc.NavLink("Bar Chart", href="#bar-chart", className="nav-link", external_link=True, id="nav-bar-chart")),
                            dbc.NavItem(dbc.NavLink("Gauge Chart", href="#gauge-chart", className="nav-link", external_link=True, id="nav-gauge-chart")),
                        ], pills=True, className="bg-light-custom p-3 stylish-nav justify-content-center", id="nav-container"),
                    ], width=12),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Radar Chart", id="radar-chart", className="card-header"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id="graph-radar-chart"), width=8),
                                    dbc.Col(html.P("This radar chart displays the comparison of different categories.", className="card-text p-3"), width=4)
                                ])
                            ])
                        ], className="mb-4", id="card-radar-chart"),
                        dbc.Card([
                            dbc.CardHeader("Bar Chart", id="bar-chart", className="card-header"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id="graph-bar-chart"), width=8),
                                    dbc.Col(html.P("This bar chart shows the distribution of values across different categories.", className="card-text p-3"), width=4)
                                ])
                            ])
                        ], className="mb-4", id="card-bar-chart"),
                        dbc.Card([
                            dbc.CardHeader("Gauge Chart", id="gauge-chart", className="card-header"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id="graph-gauge-chart"), width=8),
                                    dbc.Col(html.P("This gauge chart shows the current value relative to a scale.", className="card-text p-3"), width=4)
                                ])
                            ])
                        ], className="mb-4", id="card-gauge-chart"),
                    ], width=12, className="mb-4"),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Go to Top", className="btn-primary position-fixed bottom-0 end-0 m-4", href="#url")
                    ], width=12, className="d-flex justify-content-end")
                ])
            ], fluid=True, className="p-5", id='main-container')
        ])

        @app.callback(
            [Output('main-container', 'className'),
             Output('title-text', 'className'),
             Output('nav-container', 'className'),
             Output('nav-radar-chart', 'className'),
             Output('nav-bar-chart', 'className'),
             Output('nav-gauge-chart', 'className')],
            [Input('dark-mode-switch', 'value')]
        )
        def toggle_theme(dark_mode):
            if dark_mode:
                self.set_theme('dark')
                return [
                    'bg-dark-custom', 'text-custom-primary-dark', 'navbar-custom-dark stylish-nav',
                    'nav-link-dark', 'nav-link-dark', 'nav-link-dark'
                ]
            else:
                self.set_theme('light')
                return [
                    'bg-light-custom', 'text-custom-primary', 'navbar-custom stylish-nav',
                    'nav-link', 'nav-link', 'nav-link'
                ]

        @app.callback(
            [Output('graph-radar-chart', 'figure'),
             Output('graph-bar-chart', 'figure'),
             Output('graph-gauge-chart', 'figure')],
            [Input('dark-mode-switch', 'value')]
        )
        def update_charts(dark_mode):
            if dark_mode:
                self.set_theme('dark')
            else:
                self.set_theme('light')

            radar_chart = self.create_radar_chart()
            bar_chart = self.create_bar_chart()
            gauge_chart = self.create_gauge_chart()

            return radar_chart, bar_chart, gauge_chart

        app.run(jupyter_mode=mode)

        # app.run(jupyter_mode="tab")
        # app.run(jupyter_mode="external")



