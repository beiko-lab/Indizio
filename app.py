import itertools as it
import sys
from tempfile import NamedTemporaryFile
import io
import base64

import numpy as np
import pandas as pd

import dash
from dash.dependencies import Output, Input, State
from dash import dcc, html, ALL
from dash.exceptions import PreventUpdate
import dash_bio as dashbio

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

# Load extra layouts
cyto.load_extra_layouts()

import networkx as nx

import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from matplotlib.figure import Figure

from components import *
from utils import *
from layouts.landing_page_layout import landing_page_layout
from layouts.heatmap_layout import make_heatmap_layout
from layouts.network_layout import make_network_layout

if __name__ == "__main__":

    FONT_AWESOME = "https://use.fontawesome.com/releases/v5.7.2/css/all.css"
    external_stylesheets = [
        FONT_AWESOME,
        dbc.themes.JOURNAL,
    ]
    app = dash.Dash(
        __name__,
        external_stylesheets=external_stylesheets,
        suppress_callback_exceptions=False,
    )
    server = app.server
    colorscales = px.colors.named_colorscales()
    try:
        assert len(sys.argv) == 2
    except:
        raise ValueError(
            "app.py accepts exactly one argument. Please use the included sample sheet maker to create the required file."
        )
    # get the files
    print("Parsing sample sheet. . .")
    metas, dms, pa, tree = initialize_data(sys.argv[1])
    # make the network
    print("Initializing network. . . ")
    G = make_graph(metas, dms)
    node_items = [{"label": node, "value": node} for node in G.nodes]
    print("Done. Configuring dashboard. . .")
    dm_metric_options = []

    for i, tup in enumerate(dms):
        dm_metric_options.append({"label": tup[0], "value": tup[0]})
    # dms is a list of tuples
    dm_dict = {attr: frame for attr, frame in dms}
    # metas is either list of tuples or empty list
    meta_dict = {attr: frame for attr, frame in metas}
    default_stylesheet = [
        {
            "selector": "edge",
            "style": {
                #'width': 'mapData(lr, 50, 200, 0.75, 5)',
                "opacity": 0.4,
            },
        },
        {
            "selector": "node",
            "style": {
                #'color': '#317b75',
                "background-color": "#317b75",
                "content": "data(label)",
            },
        },
        {
            "selector": ".focal",
            "style": {
                #'color': '#E65340',
                "background-color": "#E65340",
                "content": "data(label)",
            },
        },
        {
            "selector": ".other",
            "style": {
                #'color': '#317b75',
                "background-color": "#317b75",
                "content": "data(label)",
            },
        },
    ]

    ################################################################################
    ### Page Layouts                                                             ###
    ################################################################################

    ### Entry Point. Also Serves as data dump for sharing between apps  ###
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            # Stores for data persistence.
            dcc.Store(id="node-data-store"),
            make_navbar(active=0),
            html.Div(id="page-content"),
        ]
    )

    ### Heat Map Viewer Layout ###
    page1_layout = make_heatmap_layout(dm_metric_options, colorscales)

    ### Network viz layout ###
    page2_layout = make_network_layout(
        node_items, list(dm_dict.keys()), default_stylesheet
    )

    page3_layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                id="historgram-display",
                children=[
                    dbc.Col(
                        [
                            dcc.Loading(
                                #html.Img(id="clustergram-graph"),
                                dcc.Graph(id='clustergram-graph', style={'width': '90vh', 'height': '90vh'}),
                            ),
                            dbc.Row(
                                [],
                            ),
                        ]
                    ),
                ],
            ),
        ],
    )

    ################################################################################
    ### Heatmap Callbacks                                                        ###
    ################################################################################
    @app.callback(
        Output("slider-container", "children"), [Input("dataset-select", "value")]
    )
    def update_colorscale_slider(metric):
        df = dm_dict[metric]
        maxval = np.nanmax(df.values)
        minval = np.nanmin(df.values)
        slider = dcc.RangeSlider(
            min=minval,
            max=maxval,
            step=(maxval - minval) / 100,
            marks={
                minval: {"label": "{:.2f}".format(minval)},
                maxval: {
                    "label": "{:.2f}".format(maxval),
                },
            },
            value=[minval, maxval],
            tooltip={"placement": "bottom", "always_visible": False},
            id={"role": "slider", "index": 0},
        )
        #print(minval, maxval)
        return slider

    @app.callback(
        Output({"role": "slider", "index": ALL}, "value"),
        [
            Input("minus-button", "n_clicks"),
            Input("plus-button", "n_clicks"),
            State({"role": "slider", "index": ALL}, "value"),
        ],
    )
    def update_marks(minus, plus, sliderstate):
        ctx = dash.callback_context
        slidervals = sliderstate[0]
        minval = slidervals[0]
        maxval = slidervals[-1]
        n_vals = len(slidervals)
        if not ctx.triggered:
            button_id = "noclick"
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "noclick":
            return dash.no_update
        elif button_id == "minus-button":
            if n_vals <= 2:
                return dash.no_update
            vals = list(np.linspace(minval, maxval, n_vals - 1))
            return [vals]
        else:
            vals = list(np.linspace(minval, maxval, n_vals + 1))
            return [vals]

    @app.callback(
        Output("heatmap-graph", "figure"),
        [
            Input("heatmap-button", "n_clicks"),
            State("dataset-select", "value"),
            State("colorscale", "value"),
            State("plot-mode-radio", "value"),
            State({"role": "slider", "index": ALL}, "value"),
        ],
    )
    def plot(click, dataset, scale, mode, slidervals):

        fig = go.Figure()
        # empty initially

        feature_df = dm_dict[dataset]
        meta_df = None

        if dataset in meta_dict.keys():
            meta_df = meta_dict[dataset]
        if len(slidervals) == 0:
            slidervals = [np.nanmin(feature_df.values), np.nanmax(feature_df.values)]
        else:
            slidervals = slidervals[0]
        slidervals = sorted(slidervals)
        if mode == 2:
            colorscale = []
            colors = get_color(scale, np.linspace(0, 1, len(slidervals) - 1))
            minval = min(slidervals)
            maxval = max(slidervals)
            normed_vals = [(x - minval) / (maxval - minval) for x in slidervals]
            for i, _ in enumerate(normed_vals[:-1]):
                colorscale.append([normed_vals[i], colors[i]])
                colorscale.append([normed_vals[i + 1], colors[i]])
        else:
            colorscale = scale

        ava_hm = go.Heatmap(
            x=feature_df.columns,
            y=feature_df.index,
            z=feature_df,
            colorscale=colorscale,
            zmin=slidervals[0],
            zmax=slidervals[-1],
            # colorbar=colorbar,
        )
        if type(meta_df) != type(None):
            meta_hm = go.Heatmap(
                x=meta_df.columns,
                y=meta_df.index,
                z=meta_df,
                colorscale=colorscale,
                zmin=slidervals[0],
                zmax=slidervals[-1],
                showscale=False,
            )
            f1 = go.Figure(meta_hm)
            for data in f1.data:
                fig.add_trace(data)

            f2 = go.Figure(ava_hm)
            for i in range(len(f2["data"])):
                f2["data"][i]["xaxis"] = "x2"

            for data in f2.data:
                fig.add_trace(data)

            fig.update_layout({"height": 800})
            fig.update_layout(
                xaxis={
                    "domain": [0.0, 0.20],
                    "mirror": False,
                    "showgrid": False,
                    "showline": False,
                    "zeroline": False,
                    #'ticks':"",
                    #'showticklabels': False
                }
            )
            # Edit xaxis2
            fig.update_layout(
                xaxis2={
                    "domain": [0.25, 1.0],
                    "mirror": False,
                    "showgrid": False,
                    "showline": False,
                    "zeroline": False,
                    #'showticklabels': False,
                    #'ticks':""
                }
            )
        else:
            f = go.Figure(ava_hm)
            for data in f.data:
                fig.add_trace(data)
            fig.update_layout({"height": 800})
            fig.update_layout(
                xaxis={
                    "mirror": False,
                    "showgrid": False,
                    "showline": False,
                    "zeroline": False,
                    "tickmode": "array",
                    "ticktext": feature_df.columns.str.slice(-8).tolist(),
                }
            )
        return fig

    ################################################################################
    ### Network Visualization Callbacks                                          ###
    ################################################################################
    @app.callback(
        Output("network-plot", "layout"), Input("network-callbacks-1", "value")
    )
    def update_layout(layout):
        return {"name": layout, "animate": True}

    @app.callback(
        Output("download-network", "data"),
        [
            Input("download-network-button", "n_clicks"),
            State("node-dropdown", "value"),
            State("degree", "value"),
            State({"role": "threshold", "index": ALL}, "value"),
            State({"role": "bounds-select", "index": ALL}, "value"),
        ],
    )
    def download_network(click, nodes, degree, thresholds, bounds):
        n_nodes = 0
        n_edges = 0
        attributes = list(dm_dict.keys())
        H = None
        if len(nodes) == 0:
            elements = []
        else:
            H = filter_graph(G, nodes, degree, attributes, thresholds, bounds)
        if H:
            nfile = NamedTemporaryFile("w")
            # nfile.name = 'tmp/network.graphml' TODO how can i change the name of this file?
            nx.readwrite.graphml.write_graphml(H, nfile.name)
            return dcc.send_file(nfile.name)
        return dash.no_update

    @app.callback(
        Output("node-data-store", "data"),
        [
            Input("interactive-button", "n_clicks"),
            State("node-dropdown", "value"),
            State("degree", "value"),
            State({"role": "threshold", "index": ALL}, "value"),
            State({"role": "bounds-select", "index": ALL}, "value"),
        ],
    )
    def update_elements(click, nodes, degree, thresholds, bounds):
        print("UPDATE ELEMENTS INVOKED")
        n_nodes = 0
        n_edges = 0
        attributes = list(dm_dict.keys())
        if len(nodes) == 0:
            nodes = [i["value"] for i in node_items]
        # else:
        H = filter_graph(G, nodes, degree, attributes, thresholds, bounds)
        connected = list(H.nodes)
        # Graph basics
        elements = nx_to_dash(H, nodes)
        n_nodes = len(H.nodes)
        n_edges = len(H.edges)
        # end else
        summary_data = {
            "nodes": nodes,
            'connected_nodes': connected,
            "degree": degree,
            "attributes": [],
            "n_nodes": n_nodes,
            "n_edges": n_edges,
        }
        for attr, thresh in zip(attributes, thresholds):
            summary_data["attributes"].append({"attribute": attr, "threshold": thresh})
        #print(summary_data)
        return {"elements": elements, "summary_data": summary_data}

    @app.callback(
        Output("network-plot", "elements"), 
        [Input("node-data-store", "modified_timestamp"), 
         State("node-data-store", "data")],
    )
    def update_network_plot(ts, data):
        if ts is None:
            raise PreventUpdate
        return data["elements"]

    @app.callback(
        Output("node-selected", "children"),
        [Input("node-data-store", "modified_timestamp"), 
        State("node-data-store", "data")],
    )
    def update_node_summary(ts, data):
        if ts is None:
            raise PreventUpdate
        
        summary_data = [
            dbc.ListGroupItem("Focal Node: {}".format(data["summary_data"]["nodes"])),
            dbc.ListGroupItem("Degree: {}".format(data["summary_data"]["degree"])),
        ]
        for attribute_record in data["summary_data"]["attributes"]:
            summary_data.append(
                dbc.ListGroupItem(
                    "{0} threshold: {1}".format(
                        attribute_record["attribute"], attribute_record["threshold"]
                    )
                ),
            )
        summary_data += [
            dbc.ListGroupItem("n Nodes: {}".format(data["summary_data"]["n_nodes"])),
            dbc.ListGroupItem("n Edges: {}".format(data["summary_data"]["n_edges"])),
        ]
        # summary = html.P("Focal Node: {0}\nDegree: {1}<br>LR Threshold: {2}<br>P Threshold: {3}<br>Nodes in selection: {4}<br>Edges in selection: {5}".format(node, degree, lr_threshold, p_threshold,n_nodes, n_edges))
        summary = dbc.ListGroup(
            summary_data,
        )
        return summary

    @app.callback(
        Output("network-plot", "stylesheet"), [Input("network-plot", "tapNode")],
    )
    def highlight_edges(node):
        if not node:
            return default_stylesheet

        stylesheet = [
            {
                "selector": "edge",
                "style": {
                    "opacity": 0.4,
                    #'width': 'mapData(lr, 50, 200, 0.75, 5)',
                },
            },
            {
                "selector": "node",
                "style": {
                    #'color': '#317b75',
                    "background-color": "#317b75",
                    "content": "data(label)",
                    "width": "mapData(degree, 1, 100, 25, 200)",
                },
            },
            {
                "selector": ".focal",
                "style": {
                    #'color': '#E65340',
                    "background-color": "#E65340",
                    "content": "data(label)",
                },
            },
            {
                "selector": ".other",
                "style": {
                    #'color': '#317b75',
                    "background-color": "#317b75",
                    "content": "data(label)",
                },
            },
            {
                "selector": 'node[id = "{}"]'.format(node["data"]["id"]),
                "style": {
                    "background-color": "#B10DC9",
                    "border-color": "purple",
                    "border-width": 2,
                    "border-opacity": 1,
                    "opacity": 1,
                    "label": "data(label)",
                    "color": "#B10DC9",
                    "text-opacity": 1,
                    "font-size": 12,
                    "z-index": 9999,
                },
            },
        ]
        for edge in node["edgesData"]:
            stylesheet.append(
                {
                    "selector": 'node[id= "{}"]'.format(edge["target"]),
                    "style": {
                        "background-color": "blue",
                        "opacity": 0.9,
                    },
                }
            )
            stylesheet.append(
                {
                    "selector": 'node[id= "{}"]'.format(edge["source"]),
                    "style": {
                        "background-color": "blue",
                        "opacity": 0.9,
                    },
                }
            )
            stylesheet.append(
                {
                    "selector": 'edge[id= "{}"]'.format(edge["id"]),
                    "style": {"line-color": "green", "opacity": 0.9, "z-index": 5000},
                }
            )
        return stylesheet

    ################################################################################
    ### Dendrogram Callbacks                                                     ###
    ################################################################################
    @app.callback(
        Output('clustergram-graph', 'figure'),
        [Input("node-data-store", "modified_timestamp"), 
        State("node-data-store", "data")],
        )
    def create_clustergram(ts, data):
        if not ts or isinstance(pa, type(None)) or isinstance(tree, type(None)): #TODO plot the P/A if there's no tree
            raise PreventUpdate
        # subset_pa = pa[data['summary_data']['nodes']]
        # fig = Figure()
        # ax = fig.subplots()
        # clustergram_kwargs = {
        #     'data': subset_pa,
        #     'row_linkage': tree,
        #     'cbar_pos' : None,
        #     'cmap' : sns.color_palette(['#f5f5f5', '#021657']),
        #     'method' : 'complete',
        #     'xticklabels':1,
        #     'yticklabels': False,
        #     'ax': ax,
        #     }
        # if not isinstance(metas, type(None)):
        #     #process the metadata and add a row_colors arg
        #     pass
        # #g = sns.clustermap(**clustergram_kwargs)
        # #create a bytes buffer, put the plot into bytes
        # # The buffer will populate an <img> HTML tag
        # ax.plot([1,2])
        # buf = io.BytesIO()
        # fig.savefig(buf, format='png')
        # data = base64.b64encode(buf.getbuffer()).decode('utf8')
        # buf.close()
        # return "data:image/png;base64.{}".format(data)

        subset_pa = pa[data['summary_data']['connected_nodes']]
        def phylo_linkage(y, method='single', metric='euclidean', optimal_ordering=False):
            """
            A hack to allow us to use Clustergram. Linkage is precomputed
            """
            return tree
        print("========")
        print(f"Treeshape: {tree.shape}")
        print(f"Subset PA shape: {subset_pa.shape}")
        print(f"PA shape: {pa.shape}")
        print("========")
        clustergram = dashbio.Clustergram(
            data = subset_pa.values,
            row_labels = list(subset_pa.index),
            column_labels = list(subset_pa.columns.values),
            link_fun = phylo_linkage,
            cluster='row',
            hidden_labels='row',
            height=900,
            width=1100,
            color_map= [
                [0.0, '#FFFFFF'],
                [1.0, '#EF553B']
            ]
        )
        return clustergram

    """
    @app.callback(
        Output('histogram-graph', 'figure'),
        [Input('histogram-button', 'n_clicks'),
        State('histogram-metric-select', 'value'),
        State('histogram-y-select', 'value')]
    )
    def show_histogram(click, metric_sel, y_sel):
        metric_map = {
                        '1': ('lr', 'p'),
                        '2': ('p', 'lr')
                    }
        y_map = {
                    '1': 'node_degree',
                    '2': 'n_nodes',
                    '3': 'n_edges'
                }
        dynamic_metric, static_metric = metric_map[str(metric_sel)]
        y = y_map[str(y_sel)]


        lte = lambda x,y: x<=y
        gte = lambda x,y: x>=y

        if dynamic_metric == 'lr':
            search = [25, 50, 100, 150]
            dfun = gte
            sfun = lte

        else:
            search = [0.05, 1e-5, 1e-9, 1e-12]
            dfun = lte
            sfun = gte

        if static_metric == 'p':
            static_threshold = 0.05
        else:
            static_threshold = 50

        records = []
        node_list = []
        for node in G.nodes:
            node_list.append((node, {**G.nodes[node]}))

        for dynamic_threshold in search:
            F= nx.Graph()
            F.add_nodes_from(node_list)

            edges = []
            for u,v,e in G.edges(data=True):
                if dfun(e[dynamic_metric], dynamic_threshold) and sfun(e[static_metric], static_threshold):
                #if e['lr'] >= lr_threshold and e['p'] <= p_threshold:
                    edges.append((u,v, e))

            F.add_edges_from(edges)

            graph_degree = F.degree()
            for i, node in enumerate(F.nodes):
                Sub = F.subgraph(neighborhood(F, node, 2))
                n_nodes = len(Sub.nodes)
                n_edges = len(Sub.edges)

                records.append({'node': node,
                                'node_degree': graph_degree[node],
                                'n_nodes': n_nodes,
                                'n_edges': n_edges,
                                dynamic_metric: dynamic_threshold,
                                static_metric: static_threshold,})

        rdf = pd.DataFrame.from_records(records)
        plot = px.histogram(rdf, x='node', y=y, facet_col=dynamic_metric)
        plot.update_layout({'height':800})
        return plot

    """
    ################################################################################
    ### Page Navigation callbacks                                                ###
    ################################################################################
    @app.callback(
        [
            Output("page-content", "children"),
            Output("page-1-nav", "className"),
            Output("page-2-nav", "className"),
            Output("page-3-nav", "className"),
        ],
        [
            Input("url", "pathname"),
        ],
    )
    def display_page(pathname):
        if pathname == "/page-1":
            return (
                page1_layout,
                "active",
                "",
                "",
            )
        elif pathname == "/page-2":
            return (
                page2_layout,
                "",
                "active",
                "",
            )

        elif pathname == "/page-3":
            return (
                page3_layout,
                "",
                "",
                "active",
            )

        else:
            return (
                landing_page_layout,
                "",
                "",
                "",
            )

    app.run_server(debug=True, dev_tools_ui=True)
