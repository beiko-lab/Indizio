# Some components are reused in each app. Put here for easier code readability
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

def make_navbar(active=0):
    classnames = ['', '', '']
    classnames[active] = "active"

    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Matrices", href="/page-1"),id='page-1-nav' ,className=classnames[0]),
            dbc.NavItem(dbc.NavLink("Network Visualization", href="page-2"),id='page-2-nav', className=classnames[1]),
            dbc.NavItem(dbc.NavLink("Network Statistics", href="page-3"),id='page-3-nav', className=classnames[2]),
        ],
        brand="Indizio",
        brand_href="/",
        color="primary",
        dark=True,
    )
    return navbar

def make_network_form(attributes):
    data = [
        dbc.Label('Select thresholding values'),

        dbc.InputGroup([
            dbc.InputGroupText("Degree (depth of neighborhood)"),
            dbc.Input(
                id='degree',
                placeholder='Degree (depth of neighborhood)',
                type='number', min=0, step=1, value=0
            ),
        ])
    ]
    for i, attr in enumerate(attributes):
        div = dbc.Row([
            dbc.Col([
                dbc.InputGroup([
                    dbc.InputGroupText("{} threshold".format(attr)),
                    dbc.Input(
                        id={'role': 'threshold', 'index':i},
                        placeholder = "{} threshold".format(attr),
                        type='number', value=0
                        ),
                ]),
            ]),
             dbc.Col([
                 dbc.RadioItems(
                     options=[
                         {'label': "Lower Bound", "value": 1},
                         {'label': "Upper Bound", "value": 2},
                     ],
                     value=1,
                     id={'role': "bounds-select", 'index':i},
                     inline=True,
                 ),
             ])
        ])
        data.append(div)
    return data



def make_network_filter_form(distance_labels):
    data = [
        dbc.Label('Select thresholding values'),
        dbc.Input(
                id='degree',
                placeholder='Degree (depth of neighborhood)',
                type='number', min=0, step=1, value=2
                ),
        dbc.FormText('Degree (depth of neighborhood)'),
        """
        dbc.Input(
            id='lr-threshold',
            placeholder="Likelihood Ratio lower bound",
            type='number', min=0, value=50.0
            ),
        dbc.FormText('Likelihood Ratio lower bound'),
        dbc.Input(
                id='p-threshold',
                placeholder="p-value upper bound",
                type='number', min=0, value=0.05),
        dbc.FormText("p-value upper bound")
        """
    ]
    for metric in distance_labels:
        input = dbc.Input(id)
