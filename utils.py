import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx
import os
import operator
from tqdm import tqdm

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

    stacked = [frame.where(np.triu(np.ones(frame.shape)).astype(np.bool)).stack() for frame in edge_dfs]
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
        pa = pd.read_table(p[1], sep=',', dtype=str)
        pa.set_index(pa.columns[0], inplace=True)
        pa = pa.astype(float)

    dms = []
    if type(d) != type(None):
        for tup in d:
            dms.append((tup[0], pd.read_table(tup[1], sep=',', index_col=0)))
    # if there is a pa matrix but no DM, we need to make a DM.
    else:
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
