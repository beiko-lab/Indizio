from dash import html, dcc
import dash_bootstrap_components as dbc


def make_heatmap_layout(dm_metric_options, colorscales):
    """
    Creates the Dash Layout for the page displaying the Heatmap distance matrix plot.
    ====================
    Inputs:
      dm_metric_options: A list of dicts, where each dict has the entries "label" and "value".
                         Populated by app.py
      colorscales: A list of Plotly colorscales which will be accessible by the heatmap.
    Outputs:
      layout: The Dash layout of the Heatmap component.
    ====================
    """
    layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                id="heatmap-display",
                children=[
                    dbc.Col(
                        [
                            dcc.Loading(
                                dcc.Graph(id="heatmap-graph"),
                            ),
                        ],
                        className="col-9",
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label("Choose Metric"),
                                                    dcc.Dropdown(
                                                        id="dataset-select",
                                                        value=dm_metric_options[0][
                                                            "value"
                                                        ],
                                                        options=dm_metric_options,
                                                    ),
                                                ],
                                                className="pl-5 pr-5",
                                            ),
                                        ],
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    html.P("Color Scale"),
                                    dcc.Dropdown(
                                        id="colorscale",
                                        options=[
                                            {"value": x, "label": x}
                                            for x in colorscales
                                        ],
                                        value="inferno",
                                    ),
                                ]
                            ),
                            dbc.RadioItems(
                                options=[
                                    {"label": "Continuous", "value": 1},
                                    {"label": "Binned", "value": 2},
                                ],
                                value=1,
                                id="plot-mode-radio",
                                inline=True,
                            ),
                            dbc.Row(
                                [
                                    dbc.Button(
                                        html.Span(
                                            [
                                                html.I(
                                                    className="fas fa-minus-circle ml-2"
                                                )
                                            ]
                                        ),
                                        className="col col-1",
                                        id="minus-button",
                                    ),
                                    dbc.Button(
                                        html.Span(
                                            [
                                                html.I(
                                                    className="fas fa-plus-circle ml-2"
                                                )
                                            ]
                                        ),
                                        className="col col-1",
                                        id="plus-button",
                                    ),
                                    dbc.Col(
                                        id="slider-container",
                                        children=[dcc.RangeSlider(min=0, max=0)],
                                    ),
                                ]
                            ),
                            dbc.Button(
                                "Update Heatmap", id="heatmap-button", color="secondary"
                            ),
                        ]
                    ),
                ],
            )
        ],
    )
    return layout
