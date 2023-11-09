from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto


def make_network_layout(node_items, dm_labels, stylesheet):
    """
    Creates the Dash Layout for the page displaying the Heatmap distance matrix plot.
    ====================
    Inputs:
      node_items: List of node dicts produced by app.py
      dm_labels: The user-provided distance metric labels 
      stylesheet: The CSS stylesheet for the Dash application. 
    Outputs:
      layout: The Dash layout of the Network component.
    ====================
    """
    layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                [
                    dbc.Row(
                        [
                            html.H3(children="Network Visualization"),
                            dbc.Col(
                                children=[
                                    dbc.Row(
                                        children=[
                                            html.Div(
                                                [
                                                    dbc.Label("Change network Layout"),
                                                    dcc.Dropdown(
                                                        id="network-callbacks-1",
                                                        value="grid",
                                                        clearable=False,
                                                        options=[
                                                            {
                                                                "label": name.capitalize(),
                                                                "value": name,
                                                            }
                                                            for name in [
                                                                "random",
                                                                "grid",
                                                                "circle",
                                                                "concentric",
                                                                "breadthfirst",
                                                                "cose",
                                                                "cose-bilkent",
                                                                "cola",
                                                                "klay",
                                                                "spread",
                                                                "euler",
                                                            ]
                                                        ],
                                                        className="bg-light text-dark",
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label(
                                                                "Select a node of interest."
                                                            ),
                                                            dcc.Dropdown(
                                                                id="node-dropdown",
                                                                options=node_items,
                                                                value=[],
                                                                className="bg-light text-dark",
                                                                multi=True,
                                                            ),
                                                        ]
                                                    ),
                                                    dbc.Col(
                                                        make_network_form(
                                                            dm_labels
                                                        )
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "Update Network",
                                                        id="interactive-button",
                                                        color="success",
                                                        style={"margin-bottom": "1em"},
                                                    )
                                                ],
                                                className="d-grid gap-2",
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "Download as GraphML",
                                                        id="download-network-button",
                                                        color="success",
                                                        style={"margin-bottom": "1em"},
                                                    ),
                                                    dcc.Download(id="download-network"),
                                                ],
                                                className="d-grid gap-2",
                                            ),
                                        ]
                                    ),
                                    dbc.Row(
                                        children=dbc.Card(
                                            [
                                                dbc.CardHeader(
                                                    "Network Properties",
                                                    className="bg-primary text-white",
                                                ),
                                                dbc.CardBody(
                                                    html.P(
                                                        "Lorem Ipsum and all that.",
                                                        className="card-text text-dark",
                                                        id="node-selected",
                                                    )
                                                ),
                                            ]
                                        )
                                    ),
                                ]
                            ),
                            dbc.Col(
                                children=[
                                    # dbc.Col(dcc.Graph(id='interactive-graph')),  # Not including fig here because it will be generated with the callback
                                    dbc.Col(
                                        cyto.Cytoscape(
                                            id="network-plot",
                                            elements=[],
                                            stylesheet=stylesheet,
                                            style={"width": "100%", "height": "800px"},
                                            layout={"name": "grid"},
                                        ),
                                        className="bg-white",
                                    ),
                                ],
                                className="col col-xl-9 col-lg-8 col-md-6 ",
                            ),
                        ],
                        className="bg-secondary text-white",
                    )
                ]
            ),
        ],
    )
    return layout


def make_network_form(attributes):
    data = [
        dbc.Label("Select thresholding values"),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Degree (depth of neighborhood)"),
                dbc.Input(
                    id="degree",
                    placeholder="Degree (depth of neighborhood)",
                    type="number",
                    min=0,
                    step=1,
                    value=0,
                ),
            ]
        ),
    ]
    for i, attr in enumerate(attributes):
        div = dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("{} threshold".format(attr)),
                                dbc.Input(
                                    id={"role": "threshold", "index": i},
                                    placeholder="{} threshold".format(attr),
                                    type="number",
                                    value=0,
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.RadioItems(
                            options=[
                                {"label": "Lower Bound", "value": 1},
                                {"label": "Upper Bound", "value": 2},
                            ],
                            value=1,
                            id={"role": "bounds-select", "index": i},
                            inline=True,
                        ),
                    ]
                ),
            ],
            className="border",
        )
        data.append(div)
    return data
