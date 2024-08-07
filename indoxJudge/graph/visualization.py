import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
import statistics
from typing import Union


class Visualization:
    """
    Visualization class to generate different types of charts for model evaluation.

    This class provides methods to create radar charts, scatter plots, and gauge charts
    for comparing the performance of multiple machine learning models. It supports both
    light and dark themes and adapts its visualization based on the specified mode.
    """

    def __init__(self, data: Union[list[dict], dict], mode: str = 'llm'):
        """
        Initializes the Visualization object with data and a mode.

        Parameters:
        -----------
        data : list of dict or dict
            The data to visualize. For 'llm' and 'safety' modes, this should be a list of dictionaries,
            each containing 'name' and 'score'. For 'rag' mode, this should be a dictionary with metrics
            as keys and their corresponding values.
        mode : str, optional
            The mode of visualization, which can be 'llm', 'safety', or 'rag'. Default is 'llm'.

        Raises:
        -------
        ValueError:
            If the mode is not one of 'llm', 'safety', or 'rag'.
        """
        self.mode = mode
        self.light_template = "plotly_white"
        self.dark_template = "plotly_dark"
        self.current_template = self.light_template

        if mode in ['llm', 'safety']:
            self.models = data
        elif mode == 'rag':
            self.rag_data = data
            self.metrics = list(self.rag_data.keys())
            self.values = list(self.rag_data.values())
            self.overall_score = statistics.mean(self.values)
        else:
            raise ValueError("Mode must be either 'llm', 'rag', or 'safety'")

    def set_theme(self, theme: str):
        """Sets the theme for the visualizations.

        Args:
            theme (str): The desired theme. Must be either 'light' or 'dark'.

        Raises:
            ValueError: If the provided theme is invalid.
        """

        if theme not in ('light', 'dark'):
            raise ValueError("Invalid theme. Must be 'light' or 'dark'.")

        self.current_template = self.dark_template if theme == 'dark' else self.light_template

    def create_radar_chart(self):
        """
        Creates a radar chart comparing the metrics of multiple models.

        The chart visualizes the performance of different models across various metrics.

        Returns:
            plotly.graph_objs._figure.Figure: A radar chart figure.

        Raises:
            ValueError: If the current mode is not 'llm' or 'safety'.
        """

        if self.mode not in ['llm', 'safety']:
            raise ValueError("Radar chart is only available in LLM and Safety modes")
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
        """
        Creates a bar chart visualizing evaluation metrics.

        The function generates a bar chart comparing the performance of different models
        across various metrics (LLM and Safety modes) or displaying RAG (Red, Amber,
        Green) metrics (RAG mode).

        Args:
            self: An object of the class containing the data.

        Returns:
            plotly.graph_objs._figure.Figure: A bar chart figure.

        Raises:
            ValueError: If the current mode is not 'llm', 'safety', or 'rag'.
        """
        if self.mode in ['llm', 'safety']:
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
        elif self.mode == 'rag':
            df = pd.DataFrame({'Metric': self.metrics, 'Value': self.values})
            df = df.sort_values('Value', ascending=False)

            fig = px.bar(df, x='Metric', y='Value', text='Value',
                         template=self.current_template,
                         color='Metric',
                         color_discrete_sequence=px.colors.qualitative.Plotly)  # Use a predefined color sequence

            fig.update_traces(texttemplate='%{text:.2f}', textposition='auto')

            fig.update_layout(
                title="RAG Metrics Comparison",
                xaxis_title="Metrics",
                yaxis_title="Score",
                xaxis_tickangle=-45,
                yaxis=dict(range=[0, 1]),
                height=500,
                margin=dict(l=50, r=50, t=50, b=100),
                showlegend=False
            )

        return fig

    def create_gauge_chart(self):
        """
        Creates a gauge chart visualizing evaluation scores.

        Generates a gauge chart comparing the scores of multiple models in LLM or Safety modes,
        or displays an overall evaluation score in RAG mode.

        Returns:
            plotly.graph_objs._figure.Figure: A gauge chart figure.
        """
        if self.mode == 'llm':

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
        elif self.mode == 'safety':

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
        elif self.mode == 'rag':
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(self.overall_score, 2),
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
                        'thickness': 0.75}
                }
            ))

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
        fig = px.imshow(heatmap_data, aspect="auto", color_continuous_scale="Viridis",
                        template=self.current_template)

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

        if self.mode == 'llm':
            app.layout = html.Div([
                html.Link(href='/assets/style.css', rel='stylesheet'),
                dcc.Location(id='url', refresh=False),
                dbc.Container([
                    dbc.Row([
                        dbc.Col(
                            html.H2("LLM Comparison", className="text-center my-4 display-4 text-custom-primary",
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
                                dbc.NavItem(dbc.NavLink("Scatter Plot", href="#scatter-plot", className="nav-link",
                                                        external_link=True, id="nav-scatter-plot")),
                                dbc.NavItem(dbc.NavLink("Line Plot", href="#line-plot", className="nav-link",
                                                        external_link=True, id="nav-line-plot")),
                                dbc.NavItem(dbc.NavLink("Heatmap", href="#heatmap", className="nav-link",
                                                        external_link=True, id="nav-heatmap")),
                                dbc.NavItem(dbc.NavLink("Violin Plot", href="#violin-plot", className="nav-link",
                                                        external_link=True, id="nav-violin-plot")),
                                dbc.NavItem(dbc.NavLink("Gauge Chart", href="#gauge-chart", className="nav-link",
                                                        external_link=True, id="nav-gauge-chart")),
                                dbc.NavItem(dbc.NavLink("Table", href="#table", className="nav-link",
                                                        external_link=True, id="nav-table")),
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
                                        dbc.Col(
                                            html.P(
                                                "This radar chart compares multiple metrics across different models.",
                                                className="card-text p-3"), width=4)
                                    ])
                                ])
                            ], className="mb-4", id="card-radar-chart"),
                            dbc.Card([
                                dbc.CardHeader("Bar Chart", id="bar-chart", className="card-header"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(id="graph-bar-chart"), width=8),
                                        dbc.Col(html.P(
                                            "This bar chart shows the distribution of metric values across different models.",
                                            className="card-text p-3"), width=4)
                                    ])
                                ])
                            ], className="mb-4", id="card-bar-chart"),
                            dbc.Card([
                                dbc.CardHeader("Scatter Plot", id="scatter-plot", className="card-header"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(id="graph-scatter-plot"), width=8),
                                        dbc.Col(
                                            html.P(
                                                "This scatter plot visualizes the relationship between metrics for different models.",
                                                className="card-text p-3"), width=4)
                                    ])
                                ])
                            ], className="mb-4", id="card-scatter-plot"),
                            dbc.Card([
                                dbc.CardHeader("Line Plot", id="line-plot", className="card-header"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(id="graph-line-plot"), width=8),
                                        dbc.Col(
                                            html.P("This line plot shows trends in metrics across different models.",
                                                   className="card-text p-3"), width=4)
                                    ])
                                ])
                            ], className="mb-4", id="card-line-plot"),
                            dbc.Card([
                                dbc.CardHeader("Heatmap", id="heatmap", className="card-header"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(id="graph-heatmap"), width=8),
                                        dbc.Col(html.P(
                                            "This heatmap represents metric values across models in a color-coded matrix.",
                                            className="card-text p-3"), width=4)
                                    ])
                                ])
                            ], className="mb-4", id="card-heatmap"),
                            dbc.Card([
                                dbc.CardHeader("Violin Plot", id="violin-plot", className="card-header"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(id="graph-violin-plot"), width=8),
                                        dbc.Col(html.P(
                                            "This violin plot displays the distribution of metric values across different models.",
                                            className="card-text p-3"), width=4)
                                    ])
                                ])
                            ], className="mb-4", id="card-violin-plot"),
                            dbc.Card([
                                dbc.CardHeader("Gauge Chart", id="gauge-chart", className="card-header"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(id="graph-gauge-chart"), width=8),
                                        dbc.Col(html.P("This gauge chart displays the overall scores for each model.",
                                                       className="card-text p-3"), width=4)
                                    ])
                                ])
                            ], className="mb-4", id="card-gauge-chart"),
                            dbc.Card([
                                dbc.CardHeader("Table", id="table", className="card-header"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(id="graph-table"), width=8),
                                        dbc.Col(
                                            html.P("This table provides a detailed view of all metrics for each model.",
                                                   className="card-text p-3"), width=4)
                                    ])
                                ])
                            ], className="mb-4", id="card-table"),
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
                 Output('graph-radar-chart', 'figure'),
                 Output('graph-bar-chart', 'figure'),
                 Output('graph-scatter-plot', 'figure'),
                 Output('graph-line-plot', 'figure'),
                 Output('graph-heatmap', 'figure'),
                 Output('graph-violin-plot', 'figure'),
                 Output('graph-gauge-chart', 'figure'),
                 Output('graph-table', 'figure')],
                [Input('dark-mode-switch', 'value')]
            )
            def update_llm_theme_and_graphs(dark_mode):
                if dark_mode:
                    self.set_theme('dark')
                    container_class = 'bg-dark text-white'
                    title_class = 'text-white'
                    nav_class = 'bg-dark'
                else:
                    self.set_theme('light')
                    container_class = 'bg-light'
                    title_class = 'text-dark'
                    nav_class = 'bg-light'

                radar_chart = self.create_radar_chart()
                bar_chart = self.create_bar_chart()
                scatter_plot = self.create_scatter_plot()
                line_plot = self.create_line_plot()
                heatmap = self.create_heatmap()
                violin_plot = self.create_violin_plot()
                gauge_chart = self.create_gauge_chart()
                table = self.create_table()

                return (container_class, title_class, nav_class, radar_chart, bar_chart, scatter_plot,
                        line_plot, heatmap, violin_plot, gauge_chart, table)

        if self.mode == 'safety':
            app.layout = html.Div([
                html.Link(href='/assets/style.css', rel='stylesheet'),
                dcc.Location(id='url', refresh=False),
                dbc.Container([
                    dbc.Row([
                        dbc.Col(
                            html.H2("Safety Comparison", className="text-center my-4 display-4 text-custom-primary",
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
                                dbc.NavItem(dbc.NavLink("Gauge Chart", href="#gauge-chart", className="nav-link",
                                                        external_link=True, id="nav-gauge-chart")),
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
                                        dbc.Col(
                                            html.P(
                                                "This radar chart compares multiple metrics across different models.",
                                                className="card-text p-3"), width=4)
                                    ])
                                ])
                            ], className="mb-4", id="card-radar-chart"),
                            dbc.Card([
                                dbc.CardHeader("Bar Chart", id="bar-chart", className="card-header"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(id="graph-bar-chart"), width=8),
                                        dbc.Col(html.P(
                                            "This bar chart shows the distribution of metric values across different models.",
                                            className="card-text p-3"), width=4)
                                    ])
                                ])
                            ], className="mb-4", id="card-bar-chart"),

                            dbc.CardHeader("Gauge Chart", id="gauge-chart", className="card-header"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id="graph-gauge-chart"), width=8),
                                    dbc.Col(html.P("This gauge chart displays the overall scores for each model.",
                                                   className="card-text p-3"), width=4)
                                ])
                            ])
                        ], className="mb-4", id="card-gauge-chart"),
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
                 Output('graph-radar-chart', 'figure'),
                 Output('graph-bar-chart', 'figure'),
                 Output('graph-gauge-chart', 'figure')],
                [Input('dark-mode-switch', 'value')]
            )
            def update_llm_theme_and_graphs(dark_mode):
                if dark_mode:
                    self.set_theme('dark')
                    container_class = 'bg-dark text-white'
                    title_class = 'text-white'
                    nav_class = 'bg-dark'
                else:
                    self.set_theme('light')
                    container_class = 'bg-light'
                    title_class = 'text-dark'
                    nav_class = 'bg-light'

                radar_chart = self.create_radar_chart()
                bar_chart = self.create_bar_chart()
                gauge_chart = self.create_gauge_chart()

                return (container_class, title_class, nav_class, radar_chart, bar_chart, gauge_chart)

        if self.mode == 'rag':
            app.layout = html.Div([
                html.Link(href='/assets/style.css', rel='stylesheet'),
                dcc.Location(id='url', refresh=False),
                dbc.Container([
                    dbc.Row([
                        dbc.Col(
                            html.H2("RAG Metrics", className="text-center my-4 display-4 text-custom-primary",
                                    id="title-text"), width=10),
                        dbc.Col(dbc.Switch(id="dark-mode-switch", label="Dark Mode", className="my-4"), width=2)
                    ], align="center"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Nav([
                                dbc.NavItem(dbc.NavLink("Bar Chart", href="#bar-chart", className="nav-link",
                                                        external_link=True, id="nav-bar-chart")),
                                dbc.NavItem(dbc.NavLink("Gauge Chart", href="#gauge-chart", className="nav-link",
                                                        external_link=True, id="nav-gauge-chart")),
                            ], pills=True, className="bg-light-custom p-3 stylish-nav justify-content-center",
                                id="nav-container"),
                        ], width=12),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Bar Chart", id="bar-chart", className="card-header"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(id="graph-bar-chart"), width=8),
                                        dbc.Col(html.P(
                                            "This bar chart shows the distribution of metric values across different models.",
                                            className="card-text p-3"), width=4)
                                    ])
                                ])
                            ], className="mb-4", id="card-bar-chart"),

                            dbc.CardHeader("Gauge Chart", id="gauge-chart", className="card-header"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id="graph-gauge-chart"), width=8),
                                    dbc.Col(html.P("This gauge chart displays the overall scores for each model.",
                                                   className="card-text p-3"), width=4)
                                ])
                            ])
                        ], className="mb-4", id="card-gauge-chart"),
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
                 Output('graph-bar-chart', 'figure'),
                 Output('graph-gauge-chart', 'figure')],
                [Input('dark-mode-switch', 'value')]
            )
            def update_llm_theme_and_graphs(dark_mode):
                if dark_mode:
                    self.set_theme('dark')
                    container_class = 'bg-dark text-white'
                    title_class = 'text-white'
                    nav_class = 'bg-dark'
                else:
                    self.set_theme('light')
                    container_class = 'bg-light'
                    title_class = 'text-dark'
                    nav_class = 'bg-light'

                bar_chart = self.create_bar_chart()
                gauge_chart = self.create_gauge_chart()

                return (container_class, title_class, nav_class, bar_chart, gauge_chart)

        app.run(jupyter_mode="external")
        # app.run(jupyter_mode="tab")
        # app.run()
