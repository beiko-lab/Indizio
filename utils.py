import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx
import os
import operator
from tqdm import tqdm
from _plotly_utils.basevalidators import ColorscaleValidator
import plotly.colors
from PIL import ImageColor

################################################################################
### Network Utils                                                            ###
################################################################################

def nx_to_dash(G, nodes):
    nodesout = []
    for n in G.nodes:
        if n in nodes:
            nodesout.append({
                        'data': {'id':n, 'label':n, **G.nodes[n]},
                        'classes': 'focal',
            })
        else:
            nodesout.append({'data': {'id':n, 'label':n, **G.nodes[n]},
                        'classes':'other',
            })
    edges = []
    for e in G.edges:
        edges.append({'data': {'source': e[0], 'target': e[1], **G.edges[e]}})
    return nodesout + edges

def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in path_lengths.items()
                    if length <= n]

def filter_graph(G, nodes, d, attributes, thresholds, bounds):
    print("FILTER GRAPH")
    print(attributes, thresholds, bounds)
    op_dict = {1: operator.ge, #Threshold is a lower bound, so edges must be >= thresh
           2: operator.le, #Threshold is an upper bound, so edges must be <= thresh
           }
    subgraphs = []
    for node in nodes:
        print("fg ", node)
        node_list = []
        if d == 0:
            node_list.append(node)
        edges = []
        for u,v,e in G.edges(*node_list, data=True):
            keep_edge = True
            #if e['lr'] >= lr_threshold and e['p'] <= p_threshold:
                #edges.append((u,v))
            for attr, thresh, bound in zip(attributes, thresholds, bounds):
                op = op_dict[bound]
                if not op(e[attr], thresh):
                    keep_edge = False
            if keep_edge:
                edges.append((u,v))
        H=G.edge_subgraph(edges)
        if node in H.nodes:
            if d==0:
                subgraphs.append(H)
            else:
                subgraphs.append(H.subgraph(neighborhood(H, node, d)))
        else:
            subgraphs.append(G.subgraph([node]))
    return nx.compose_all(subgraphs)


################################################################################
### Parsing Utils                                                            ###
################################################################################

class SamplesheetError(Exception):
    pass


def parse_samplesheet(path):

    bad_sheet = "Wrong columns"
    invalid_codes= "Wrong codes"
    wrong_files = "Wrong files"
    no_file = 'File not found: {}'

    valid_codes = set(['M', 'DM', 'T', 'P'])

    df = pd.read_table(path, sep=',')
    print(df.columns)
    #Sheet must have exactly three columns.
    try:
        assert list(df.columns) == ['filepath', 'type', 'label']
    except:
        raise SamplesheetError(bad_sheet)

    # All file types must be in valid_codes
    try:
        assert set(df['type']) - valid_codes == set()
    except:
        raise SamplesheetError(invalid_codes)

    # Check file type requirements and limits
    type_counts = Counter(df['type'])
    try:
        assert type_counts['DM'] > 0 or type_counts['P'] == 1
        # Actually, allow multiple distance matrices.
        assert type_counts['P'] < 2 and type_counts['T'] < 2
    except:
        raise SamplesheetError(wrong_files)

    #Make sure all the files exist.
    for file in df['filepath']:
        try:
            assert os.path.isfile(file)
        except:
            raise SamplesheetError(no_file.format(file))

    #return the file/label tuples

    meta_files = df[df['type']=='M'][['label', 'filepath']].values
    distance_files = df[df['type']=='DM'][['label', 'filepath']].values
    tree_file = None
    pa_file = None
    if 'T' in df['type'].unique():
        tree_file = df[df['type']=='T'][['label', 'filepath']].values[0]
    if 'P' in df['type'].unique():
        pa_file = df[df['type']=='P'][['label', 'filepath']].values[0]

    return meta_files, distance_files, tree_file, pa_file

################################################################################
### Formatting Utils                                                         ###
################################################################################
def make_graph(meta_files, distance_files):
    G = nx.Graph()
    #add edges first
    edge_dfs = []
    edge_attrs = []
    for tup in distance_files:
        edge_attrs.append(tup[0])
        edge_dfs.append(tup[1])

    # Make sure the dfs are all same shapes
    assert len(list(set([df.shape[0] for df in edge_dfs]))) == 1
    assert len(list(set([df.shape[1] for df in edge_dfs]))) == 1

    stacked = [frame.where(np.triu(np.ones(frame.shape)).astype(bool)).stack() for frame in edge_dfs]
    pairs = stacked[0].index
    print("Constructing nodes. . .")
    nodes = list(edge_dfs[0].columns)

    for node in tqdm(nodes):
        G.add_node(node)

    print("Constructing edges. . .")
    #edges = []
    data = list(zip(edge_attrs, stacked))
    edge_dict = {k: dict() for k in pairs}
    for tup in tqdm(pairs):
        for attribute, df in data:
            edge_dict[tup][attribute] = df.loc[tup]

    edges = [(*k, v) for k, v in edge_dict.items()]
    G.add_edges_from(edges)
    #Need to add the metadata...

    return G

#the big one.
def initialize_data(path):
    m, d, t, p = parse_samplesheet(path)
    pa = None
    if type(p) != type(None):
        print("pa matrix found")
        pa = pd.read_table(p[1], sep=',', dtype=str)
        pa.set_index(pa.columns[0], inplace=True)
        pa = pa.astype(float)

    dms = []
    if type(d) != type(None):
        for tup in d:
            dms.append((tup[0], pd.read_table(tup[1], sep=',', index_col=0)))
    # if there is a pa matrix but no DM, we need to make a DM.
    if len(dms)==0:
        print("pearson")
        dms.append(('(abs) pearson', pa.corr().abs()))

    if type(m) != type(None):
        metas = []
        for tup in m:
            metas.append((tup[0], pd.read_table(tup[1], sep=',', index_col=0)))
    tree = None
    if type(t) != type(None):
        #load the tree
        pass

    return metas, dms, pa, tree


################################################################################
## These two color functions from                                             ##
## https://stackoverflow.com/questions/62710057/access-color-from-plotly-color-scale
################################################################################
def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale_name)

    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)

def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )
