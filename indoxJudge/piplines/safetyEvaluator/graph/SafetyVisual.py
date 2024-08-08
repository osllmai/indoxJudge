import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input


class SafetyVis:

    def __init__(self, models, interpretations):
        """
        Initialize the LLMVisualizer with the given models and interpretations.

        Parameters:
        models (list): A list of dictionaries, each containing 'name', 'score', and 'metrics'.
        interpretations (list): A list of dictionaries with chart names as keys and their interpretations as values.
        """
        self.models = models
        self.interpretations = interpretations
        self.light_template = "plotly_white"
        self.dark_template = "plotly_dark"
        self.current_template = self.light_template

    def set_theme(self, theme):
        if theme == 'dark':
            self.current_template = self.dark_template
        else:
            self.current_template = self.light_template

    def create_radar_chart(self):
        fig = go.Figure()
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            labels = list(metrics.keys())
            values = list(metrics.values())
            values += values[:1]
            labels += labels[:1]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                mode='lines+markers',
                line=dict(width=1),
                name=name
            ))

        fig.update_layout(
            title='Evaluation Metrics Radar Chart',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            template=self.current_template
        )
        return fig

    def create_bar_chart(self):
        data = []
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            for metric, value in metrics.items():
                data.append({"Model": name, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)

        fig = px.bar(df, x='Metric', y='Value', color='Model', barmode='group', text='Value',
                     template=self.current_template)
        fig.update_layout(
            title='Evaluation Metrics Bar Chart',
            xaxis_title='Metrics',
            yaxis_title='Values',
            xaxis_tickangle=-45
        )
        return fig

    def create_box_plot(self):
        data = []
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            for metric, value in metrics.items():
                data.append({"Model": name, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)

        fig = px.box(df, x='Metric', y='Value', color='Model',
                    template=self.current_template, title='Evaluation Metrics Box Plot')

        fig.update_layout(
            xaxis_title='Metrics',
            yaxis_title='Values'
        )
        return fig

    def create_bubble_plot(self):
        data = []
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            for metric, value in metrics.items():
                data.append({"Model": name, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)

        fig = px.scatter(df, x='Metric', y='Value', color='Model', size='Value',
                        template=self.current_template, title='Evaluation Metrics Bubble Plot')

        fig.update_layout(
            xaxis_title='Metrics',
            yaxis_title='Values'
        )
        return fig

    def create_gauge_chart(self):
        fig = make_subplots(rows=1, cols=len(self.models), specs=[[{'type': 'indicator'}] * len(self.models)],
                            subplot_titles=[model['name'] for model in self.models])

        for i, model in enumerate(self.models, 1):
            name = model['name']
            score = model['score']
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                delta={'reference': 0.5, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 1], 'tickwidth': 0.7, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.2], 'color': "green"},
                        {'range': [0.2, 0.4], 'color': "lightgreen"},
                        {'range': [0.4, 0.6], 'color': "yellow"},
                        {'range': [0.6, 0.8], 'color': "orange"},
                        {'range': [0.8, 1], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': score}
                }
            ), row=1, col=i)

        font = {'color': "white", 'family': "Arial"} if self.current_template == "plotly_dark" else {
            'color': "black", 'family': "Arial"}
        paper_bgcolor = "black" if self.current_template == "plotly_dark" else "white"
        plot_bgcolor = "white" if self.current_template == "plotly_dark" else "black"
        fig.update_layout(
            title='Overall Evaluation Scores',
            font=font,
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=plot_bgcolor,
            template=self.current_template
        )
        return fig

    def create_pie_chart(self):
        data = []
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            for metric, value in metrics.items():
                data.append({"Model": name, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)
        df_agg = df.groupby('Metric')['Value'].sum().reset_index()

        fig = px.pie(df_agg, names='Metric', values='Value',
                    template=self.current_template, title='Aggregated Evaluation Metrics Pie Chart')

        fig.update_layout(
            title='Aggregated Evaluation Metrics Pie Chart'
        )
        return fig

    def create_table(self):
        data = []
        for model in self.models:
            row = {'Model': model['name']}
            row.update(model['metrics'])
            data.append(row)

        df = pd.DataFrame(data)
        text_color = 'white' if self.current_template == self.dark_template else 'black'
        cells_color = 'black' if self.current_template == self.dark_template else 'white'
        fig = go.Figure(data=[go.Table(
            columnwidth=[150] * len(df.columns),
            header=dict(values=list(df.columns),
                        fill_color='paleturquoise',
                        align='left',
                        font=dict(color="black", size=12)),
            cells=dict(values=[df[col].tolist() for col in df.columns],
                       fill_color='lavender',
                       align='left',
                       font=dict(size=10, color="black"))
        )])

        fig.update_layout(
            title='Evaluation Metrics Table',
            autosize=True,
            margin=dict(l=0, r=0, t=30, b=0),
            template=self.current_template
        )
        return fig

    def plot(self):
        app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.themes.DARKLY])

        app.layout = html.Div([
            html.Link(href='/assets/style.css', rel='stylesheet'),
            dcc.Location(id='url', refresh=False),
            dbc.Container([
                dbc.Row([
                    dbc.Col(
                        html.H2("IndoxJudge", className="text-center my-4 display-4 text-custom-primary",
                                id="title-text"), width=10),
                    dbc.Col(dbc.Switch(id="dark-mode-switch", label="Dark Mode", className="my-4"), width=2)
                ], align="center"),
                dbc.Row([
                    dbc.Col([
                        dbc.Nav([
                            dbc.NavItem(dbc.NavLink("Radar Chart", href="#radar-chart", className="nav-link",
                                                    external_link=True, id="nav-radar-chart")),
                            dbc.NavItem(dbc.NavLink("Bar Chart", href="#bar-chart", className="nav-link",
                                                    external_link=True, id="nav-bar-chart")),
                            dbc.NavItem(
                                dbc.NavLink("Box Plot", href="#box-plot", className="nav-link", external_link=True,
                                            id="nav-box-plot")),
                            dbc.NavItem(dbc.NavLink("Bubble Plot", href="#bubble-plot", className="nav-link",
                                                    external_link=True, id="nav-bubble-plot")),
                            dbc.NavItem(dbc.NavLink("Gauge Chart", href="#gauge-chart", className="nav-link",
                                                    external_link=True, id="nav-gauge-chart")),
                            dbc.NavItem(dbc.NavLink("Pie Chart", href="#pie-chart", className="nav-link",
                                                    external_link=True, id="nav-pie-chart")),                                                    
                            dbc.NavItem(dbc.NavLink("Table", href="#table", className="nav-link", external_link=True,
                                                    id="nav-table")),
                        ], pills=True, className="bg-light-custom p-3 stylish-nav justify-content-center",
                            id="nav-container"),
                    ], width=12),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Radar Chart", id="radar-chart", className="card-header"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id="graph-radar-chart"), width=8),
                                    dbc.Col(dcc.Markdown(self.interpretations.get('Radar Chart', 'No interpretation provided'),
                                                         className="card-text p-3"), width=4)
                                ])
                            ])
                        ], className="mb-4", id="card-radar-chart"),
                        dbc.Card([
                            dbc.CardHeader("Bar Chart", id="bar-chart", className="card-header"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id="graph-bar-chart"), width=8),
                                    dbc.Col(dcc.Markdown(self.interpretations.get('Bar Chart', 'No interpretation provided'),
                                                         className="card-text p-3"), width=4)
                                ])
                            ])
                        ], className="mb-4", id="card-bar-chart"),
                        dbc.Card([
                            dbc.CardHeader("Box Plot", id="box-plot", className="card-header"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id="graph-box-plot"), width=8),
                                    dbc.Col(dcc.Markdown(self.interpretations.get('Box Plot', 'No interpretation provided'),
                                                         className="card-text p-3"), width=4)
                                ])
                            ])
                        ], className="mb-4", id="card-box-plot"),
                        dbc.Card([
                            dbc.CardHeader("Bubble Plot", id="bubble-plot", className="card-header"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id="graph-bubble-plot"), width=8),
                                    dbc.Col(dcc.Markdown(self.interpretations.get('Bubble Plot', 'No interpretation provided'),
                                                         className="card-text p-3"), width=4)
                                ])
                            ])
                        ], className="mb-4", id="card-bubble-plot"),
                        dbc.Card([
                            dbc.CardHeader("Gauge Chart", id="gauge-chart", className="card-header"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id="graph-gauge-chart"), width=8),
                                    dbc.Col(dcc.Markdown(self.interpretations.get('Gauge Chart', 'No interpretation provided'),
                                                         className="card-text p-3"), width=4)
                                ])
                            ])
                        ], className="mb-4", id="card-gauge-chart"),
                        dbc.Card([
                            dbc.CardHeader("Pie Chart", id="pie-chart", className="card-header"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id="graph-pie-chart"), width=8),
                                    dbc.Col(dcc.Markdown(self.interpretations.get('Pie Chart', 'No interpretation provided'),
                                                         className="card-text p-3"), width=4)
                                ])
                            ])
                        ], className="mb-4", id="card-pie-chart"),
                        dbc.Card([
                            dbc.CardHeader("Table", id="table", className="card-header"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id="graph-table"), width=8),
                                    dbc.Col(dcc.Markdown(self.interpretations.get('Table', 'No interpretation provided'),
                                                         className="card-text p-3"), width=4)
                                ])
                            ])
                        ], className="mb-4", id="card-table")
                    ], width=12, className="mb-4"),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Go to Top", className="btn-primary position-fixed bottom-0 end-0 m-4",
                                   href="#url")
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
             Output('nav-box-plot', 'className'),
             Output('nav-bubble-plot', 'className'),
             Output('nav-gauge-chart', 'className'),
             Output('nav-pie-chart', 'className'),
             Output('nav-table', 'className'),
             Output('card-radar-chart', 'className'),
             Output('card-bar-chart', 'className'),
             Output('card-box-plot', 'className'),
             Output('card-bubble-plot', 'className'),
             Output('card-gauge-chart', 'className'),
             Output('card-pie-chart', 'className')],
            [Input('dark-mode-switch', 'value')]
        )
        def toggle_theme(dark_mode):
            if dark_mode:
                self.set_theme('dark')
                return [
                    'bg-dark-custom', 'text-custom-primary-dark', 'navbar-custom-dark stylish-nav',
                    'nav-link-dark', 'nav-link-dark', 'nav-link-dark', 'nav-link-dark',
                    'nav-link-dark', 'nav-link-dark', 'nav-link-dark', 'nav-link-dark',
                    'card-header-dark', 'card-header-dark', 'card-header-dark', 'card-header-dark',
                    'card-header-dark', 'card-header-dark', 'card-header-dark'
                ]
            else:
                self.set_theme('light')
                return [
                    'bg-light-custom', 'text-custom-primary', 'navbar-custom stylish-nav',
                    'nav-link', 'nav-link', 'nav-link', 'nav-link', 'nav-link', 'nav-link',
                    'nav-link', 'nav-link', 'card-header', 'card-header', 'card-header',
                    'card-header', 'card-header', 'card-header', 'card-header'
                ]

        @app.callback(
            [Output('graph-radar-chart', 'figure'),
             Output('graph-bar-chart', 'figure'),
             Output('graph-box-plot', 'figure'),
             Output('graph-bubble-plot', 'figure'),
             Output('graph-gauge-chart', 'figure'),
             Output('graph-pie-chart', 'figure'),
             Output('graph-table', 'figure')],
            [Input('dark-mode-switch', 'value')]
        )
        def update_charts(dark_mode):
            if dark_mode:
                self.set_theme('dark')
            else:
                self.set_theme('light')

            radar_chart = self.create_radar_chart()
            bar_chart = self.create_bar_chart()
            box_plot = self.create_box_plot()
            bubble_plot = self.create_bubble_plot()
            gauge_chart = self.create_gauge_chart()
            pie_chart = self.create_pie_chart()
            table = self.create_table()

            return radar_chart, bar_chart, box_plot, bubble_plot, gauge_chart, pie_chart, table

        app.run(jupyter_mode="external")