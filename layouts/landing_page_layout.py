from dash import html
import dash_bootstrap_components as dbc

### Landing Page ###
# Adapted from https://getbootstrap.com/docs/4.0/examples/product/
landing_page_layout = [
    html.Div(
        className="position-relative overflow-hidden p-3 p-md-5 m-md-3 text-center bg-light",
        children=[
            html.Div(
                className="col-md-5 p-lg-5 mx-auto my-5",
                children=[
                    html.H1("Indizio", className="display-4 font-weight-normal"),
                    html.P(
                        "Interactively explore connected data.",
                        className="lead font-weight-normal",
                    ),
                    html.P(
                        "(Note: This page will be replaced by a data upload form later.)",
                        className="lead font-weight-normal",
                    ),
                    html.A(
                        "Get Started",
                        href="page-1",
                        className="btn btn-outline-secondary",
                    ),
                ],
            )
        ],
    ),
    html.Div(
        className="d-md-flex flex-md-equal w-100 my-md-3 pl-md-3",
        children=[
            html.Div(
                className="bg-dark mr-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center text-white overflow-hidden",
                children=[
                    html.Div(
                        className="my-3 py-3",
                        children=[
                            html.H2("Heatmap viewer.", className="display-5"),
                            html.P(
                                children=[
                                    "View connected data as heatmaps. Explroe heatmaps for each distance metric you uploaded. Click 'Matrices'.",
                                ],
                                className="lead",
                            ),
                        ],
                    )
                ],
            ),
            html.Div(
                className="bg-light mr-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center overflow-hidden",
                children=[
                    html.Div(
                        className="my-3 py-3",
                        children=[
                            html.H2(
                                "Explore Specific Subnetworks.", className="display-5"
                            ),
                            html.P(
                                'Visualize networks surrounding specific nodes. Choose your node or nodes, select filtering parameters, and explore. Click "Network Visualisation."',
                                className="lead",
                            ),
                        ],
                    )
                ],
            ),
        ],
    ),
    html.Div(
        className="d-md-flex flex-md-equal w-100 my-md-3 pl-md-3",
        children=[
            html.Div(
                className="bg-light mr-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center overflow-hidden",
                children=[
                    html.Div(
                        className="my-3 py-3",
                        children=[
                            html.H2(
                                "View Macro Level Statistics.", className="display-5"
                            ),
                            html.P(
                                "View at a macro level how LR and p value choices influence network properties. Click 'Network Statistics'.",
                                className="lead",
                            ),
                        ],
                    )
                ],
            ),
            html.Div(
                className="bg-primary mr-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center text-white overflow-hidden",
                children=[
                    html.Div(
                        className="my-3 py-3",
                        children=[
                            html.H2(
                                "(Not yet) View genome presence/absence cladogram.",
                                className="display-5",
                            ),
                            html.P(
                                "See the presence and absence of genes etc. in a per genome basis. (Not yet implemented).",
                                className="lead",
                            ),
                        ],
                    )
                ],
            ),
        ],
    ),
]
