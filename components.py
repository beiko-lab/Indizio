# Some components are reused in each app. Put here for easier code readability
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
