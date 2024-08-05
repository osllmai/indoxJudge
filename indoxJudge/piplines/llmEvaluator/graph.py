import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input


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


class LLMVisualizer:
    def __init__(self, models):
        """
        Initialize the LLMVisualizer with the given models.

        Parameters:
        models (list): A list of dictionaries, each containing 'name', 'score', and 'metrics'.
        """
        self.models = models
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
        """
        Creates a radar chart comparing the metrics of multiple models.
        Returns the radar chart figure.
        """
        fig = go.Figure()
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            filtered_metrics = {k: v for k, v in metrics.items()}
            labels = list(filtered_metrics.keys())
            values = list(filtered_metrics.values())
            values += values[:1]
            labels += labels[:1]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                mode='lines+markers',
                line=dict(width=1),  # Thinner lines
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
        """
        Creates a bar chart comparing the metrics of multiple models.
        Returns the bar chart figure.
        """
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

    def create_gauge_chart(self):
        """
        Creates a gauge chart comparing the scores of multiple models.
        Returns the gauge chart figure.
        """
        fig = make_subplots(rows=1, cols=len(self.models), specs=[[{'type': 'indicator'}] * len(self.models)],
                            subplot_titles=[model['name'] for model in self.models])

        for i, model in enumerate(self.models, 1):
            name = model['name']
            score = model['score']
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                # title={'text': name, 'font': {'size': 24}},
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
                        'value': score}
                }
            ), row=1, col=i)
        font = {'color': "white", 'family': "Arial"} if self.current_template == "plotly_dark" else {'color': "black",
                                                                                                     'family': "Arial"}
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

    def create_scatter_plot(self):
        """
        Creates a scatter plot comparing the metrics of multiple models.
        Returns the scatter plot figure.
        """
        data = []
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            for metric, value in metrics.items():
                data.append({"Model": name, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)

        fig = px.scatter(df, x='Value', y='Metric', color='Model', size='Value', hover_name='Model',
                         template=self.current_template)
        fig.update_layout(
            title='Evaluation Metrics Scatter Plot',
            xaxis_title='Value',
            yaxis_title='Metric'
        )
        return fig

    def create_line_plot(self):
        """
        Creates a line plot comparing the metrics of multiple models.
        Returns the line plot figure.
        """
        data = []
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            for metric, value in metrics.items():
                data.append({"Model": name, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)

        fig = px.line(df, x='Metric', y='Value', color='Model', markers=True, template=self.current_template)
        fig.update_layout(
            title='Evaluation Metrics Line Plot',
            xaxis_title='Metrics',
            yaxis_title='Values'
        )
        return fig

    def create_heatmap(self):
        """
        Creates a heatmap comparing the metrics of multiple models.
        Returns the heatmap figure.
        """
        data = []
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            for metric, value in metrics.items():
                data.append({"Model": name, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)

        heatmap_data = df.pivot(index="Model", columns="Metric", values="Value")
        fig = px.imshow(heatmap_data, aspect="auto", color_continuous_scale="Viridis", template=self.current_template)

        fig.update_layout(
            title='Evaluation Metrics Heatmap',
            xaxis_title='Metrics',
            yaxis_title='Models'
        )
        return fig

    def create_violin_plot(self):
        """
        Creates a violin plot comparing the metrics of multiple models.
        Returns the violin plot figure.
        """
        data = []
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            for metric, value in metrics.items():
                data.append({"Model": name, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)

        fig = px.violin(df, x='Metric', y='Value', color='Model', box=True, points="all",
                        template=self.current_template)

        fig.update_layout(
            title='Evaluation Metrics Violin Plot',
            xaxis_title='Metrics',
            yaxis_title='Values'
        )
        return fig

    def create_table(self):
        """
        Creates a table comparing the metrics of multiple models.
        Returns the table figure.
        """
        data = []
        for model in self.models:
            row = {'Model': model['name']}
            row.update(model['metrics'])
            data.append(row)

        df = pd.DataFrame(data)
        text_color = 'white' if self.current_template == self.dark_template else 'black'
        cells_color = 'black' if self.current_template == self.dark_template else 'white'
        fig = go.Figure(data=[go.Table(
            columnwidth=[150] * len(df.columns),  # Increase column width
            header=dict(values=list(df.columns),
                        fill_color='paleturquoise',
                        align='left',
                        font=dict(color="black", size=12)),  # Increase header font size
            cells=dict(values=[df[col].tolist() for col in df.columns],
                       fill_color='lavender',
                       align='left',
                       font=dict(size=10, color="black"))  # Increase cell font size
        )])

        fig.update_layout(
            title='Evaluation Metrics Table',
            autosize=True,
            # width=900,
            margin=dict(l=0, r=0, t=30, b=0),
            template=self.current_template
        )
        return fig

    def plot(self):
        app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.themes.DARKLY])
        app.layout = html.Div([
            html.Link(href='/assets/style.css', rel='stylesheet'),
            dcc.Location(id='url', refresh=False),
            # dcc.Markdown(children=custom_css, dangerously_allow_html=True),
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.H2("IndoxJudge Dashboard", className="text-center my-4 display-4 text-custom-primary",
                                    id="title-text"), width=10),
                    dbc.Col(dbc.Switch(id="dark-mode-switch", label="Dark Mode", className="my-4"), width=2)
                ], align="center"),
                dbc.Row([
                    dbc.Col([
                        dbc.Nav([
                            dbc.NavLink("Radar Chart", href="#radar-chart", className="nav-link", external_link=True,
                                        id="nav-radar-chart"),
                            dbc.NavLink("Bar Chart", href="#bar-chart", className="nav-link", external_link=True,
                                        id="nav-bar-chart"),
                            dbc.NavLink("Scatter Plot", href="#scatter-plot", className="nav-link", external_link=True,
                                        id="nav-scatter-plot"),
                            dbc.NavLink("Line Plot", href="#line-plot", className="nav-link", external_link=True,
                                        id="nav-line-plot"),
                            dbc.NavLink("Heatmap", href="#heatmap", className="nav-link", external_link=True,
                                        id="nav-heatmap"),
                            dbc.NavLink("Violin Plot", href="#violin-plot", className="nav-link", external_link=True,
                                        id="nav-violin-plot"),
                            dbc.NavLink("Gauge Chart", href="#gauge-chart", className="nav-link", external_link=True,
                                        id="nav-gauge-chart"),
                            dbc.NavLink("Table", href="#table", className="nav-link", external_link=True,
                                        id="nav-table"),
                        ], pills=True, className="bg-light-custom p-3 stylish-nav justify-content-center",
                            id="nav-container"),
                    ], width=12),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Radar Chart", id="radar-chart", className="card-header"),
                            dbc.CardBody([
                                dcc.Graph(id="graph-radar-chart"),
                                html.P("This radar chart displays the comparison of different categories.",
                                       className="card-text p-3")
                            ])
                        ], className="mb-4", id="card-radar-chart"),
                        dbc.Card([
                            dbc.CardHeader("Bar Chart", id="bar-chart", className="card-header"),
                            dbc.CardBody([
                                dcc.Graph(id="graph-bar-chart"),
                                html.P("This bar chart shows the distribution of values across different categories.",
                                       className="card-text p-3")
                            ])
                        ], className="mb-4", id="card-bar-chart"),
                        dbc.Card([
                            dbc.CardHeader("Scatter Plot", id="scatter-plot", className="card-header"),
                            dbc.CardBody([
                                dcc.Graph(id="graph-scatter-plot"),
                                html.P("This scatter plot visualizes the relationship between two variables.",
                                       className="card-text p-3")
                            ])
                        ], className="mb-4", id="card-scatter-plot"),
                        dbc.Card([
                            dbc.CardHeader("Line Plot", id="line-plot", className="card-header"),
                            dbc.CardBody([
                                dcc.Graph(id="graph-line-plot"),
                                html.P("This line plot illustrates the trend over a period of time.",
                                       className="card-text p-3")
                            ])
                        ], className="mb-4", id="card-line-plot"),
                        dbc.Card([
                            dbc.CardHeader("Heatmap", id="heatmap", className="card-header"),
                            dbc.CardBody([
                                dcc.Graph(id="graph-heatmap"),
                                html.P("This heatmap represents data values in a matrix format.",
                                       className="card-text p-3")
                            ])
                        ], className="mb-4", id="card-heatmap"),
                        dbc.Card([
                            dbc.CardHeader("Violin Plot", id="violin-plot", className="card-header"),
                            dbc.CardBody([
                                dcc.Graph(id="graph-violin-plot"),
                                html.P(
                                    "This violin plot displays the distribution of the data across different categories.",
                                    className="card-text p-3")
                            ])
                        ], className="mb-4", id="card-violin-plot"),
                        dbc.Card([
                            dbc.CardHeader("Gauge Chart", id="gauge-chart", className="card-header"),
                            dbc.CardBody([
                                dcc.Graph(id="graph-gauge-chart"),
                                html.P("This gauge chart shows the current value relative to a scale.",
                                       className="card-text p-3")
                            ])
                        ], className="mb-4", id="card-gauge-chart"),
                        dbc.Card([
                            dbc.CardHeader("Table", id="table", className="card-header"),
                            dbc.CardBody([
                                dcc.Graph(id="graph-table"),
                                html.P("This table provides a tabular representation of the data.",
                                       className="card-text p-3")
                            ])
                        ], className="mb-4", id="card-table"),
                    ], width=12),
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
             Output('nav-scatter-plot', 'className'),
             Output('nav-line-plot', 'className'),
             Output('nav-heatmap', 'className'),
             Output('nav-violin-plot', 'className'),
             Output('nav-gauge-chart', 'className'),
             Output('nav-table', 'className'),
             Output('card-radar-chart', 'className'),
             Output('card-bar-chart', 'className'),
             Output('card-scatter-plot', 'className'),
             Output('card-line-plot', 'className'),
             Output('card-heatmap', 'className'),
             Output('card-violin-plot', 'className'),
             Output('card-gauge-chart', 'className'),
             Output('card-table', 'className')],
            [Input('dark-mode-switch', 'value')]
        )
        def toggle_theme(dark_mode):
            if dark_mode:
                self.set_theme('dark')
                return [
                    'bg-dark-custom', 'text-custom-primary-dark', 'bg-dark-custom',
                    'nav-link-dark', 'nav-link-dark', 'nav-link-dark', 'nav-link-dark',
                    'nav-link-dark', 'nav-link-dark', 'nav-link-dark', 'nav-link-dark',
                    'card-header-dark', 'card-header-dark', 'card-header-dark', 'card-header-dark',
                    'card-header-dark', 'card-header-dark', 'card-header-dark', 'card-header-dark'
                ]
            else:
                self.set_theme('light')
                return [
                    'bg-light-custom', 'text-custom-primary', 'bg-light-custom',
                    'nav-link', 'nav-link', 'nav-link', 'nav-link', 'nav-link', 'nav-link',
                    'nav-link', 'nav-link', 'card-header', 'card-header', 'card-header',
                    'card-header', 'card-header', 'card-header', 'card-header', 'card-header'
                ]

        @app.callback(
            Output('graph-radar-chart', 'figure'),
            Output('graph-bar-chart', 'figure'),
            Output('graph-scatter-plot', 'figure'),
            Output('graph-line-plot', 'figure'),
            Output('graph-heatmap', 'figure'),
            Output('graph-violin-plot', 'figure'),
            Output('graph-gauge-chart', 'figure'),
            Output('graph-table', 'figure'),
            [Input('dark-mode-switch', 'value')]
        )
        def update_charts(dark_mode):
            if dark_mode:
                self.set_theme('dark')
            else:
                self.set_theme('light')

            radar_chart = self.create_radar_chart()
            bar_chart = self.create_bar_chart()
            scatter_plot = self.create_scatter_plot()
            line_plot = self.create_line_plot()
            heatmap = self.create_heatmap()
            violin_plot = self.create_violin_plot()
            gauge_chart = self.create_gauge_chart()
            table = self.create_table()

            return radar_chart, bar_chart, scatter_plot, line_plot, heatmap, violin_plot, gauge_chart, table

        app.run(jupyter_mode="external")
