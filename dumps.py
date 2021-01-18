import simplejson as json
import networkx as nx

# https://prodevsblog.com/questions/139341/method-to-save-networkx-graph-to-json-graph/
def store(G, filename):
    json.dump(dict(nodes = [[n, G.nodes[n]] for n in G.nodes()],
                   edges = [[u, v, G.edges[u, v]] for u, v in G.edges()]),
                   open(filename, 'w'), indent = 2)
              
def load(filename):
    G = nx.Graph()
    d = json.load(open(filename))
    for node in d['nodes']:
        label = str(node[0])
        attributes = node[1]
        color = attributes.get('color', '#000000') # default black
        (x, y) = attributes['pos']
        value = attributes.get('value', '[]')
        G.add_node(label, pos = (x, y), value = value, color = color)
    for e in d['edges']:
        v = str(e[0])
        u = str(e[1])
        G.add_edge(u, v)
    return G

