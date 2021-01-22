import matplotlib.pyplot as plt
from dumps import store, load
import networkx as nx
from sys import argv

filename = argv[1]
print('Loading graph data from', filename)
G = load(filename)
for v in G.nodes:
    for u in G.neighbors(v):
        print(v, u, G.nodes[u]['value'])
pos = nx.get_node_attributes(G, 'pos')
nx.draw_networkx_nodes(G, pos, node_size = 10, node_shape='o')
nx.draw_networkx_edges(G, pos, width = 1, edge_color = 'green')
labels = dict()
for v in G:
    if len(v) == 2:
        (r, c) = v
    else:
        fields = v[1:-1].split('_')
        r = float(fields[2])
        c = float(fields[3])
    labels[v] = f'({r}, {c})'
nx.draw_networkx_labels(G, pos, labels, font_size = 4)
plt.axis("off")
plt.savefig(filename.replace('.json', '_graph.png'), bbox_inches = "tight", dpi = 150)
plt.clf()
