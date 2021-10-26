import matplotlib.pyplot as plt
from dumps import store, load
import networkx as nx
from sys import argv

tooMany = 100 # surpress the labels for large graphs
plt.rcParams['figure.figsize'] = 40, 30
verbose = False # print out the edge list
filename = argv[1]
print('Loading graph data from', filename)
G = load(filename)
rgb = nx.get_node_attributes(G, 'color')
color= [ '#%02x%02x%02x' % (int(rgb[c][0]),
                            int(rgb[c][1]),
                            int(rgb[c][2])) for c in rgb ] 
print(color)
channels = nx.get_node_attributes(G, 'value')
pos = nx.get_node_attributes(G, 'pos')
print('Drawing nodes...')
nx.draw_networkx_nodes(G, pos, node_size = 300, node_shape='o', node_color = color)
print('Drawing edges...')
nx.draw_networkx_edges(G, pos, width = 1, edge_color = 'black')
if G.order() < tooMany:
    labels = dict()
    for v in G:
        if len(v) == 2:
            (r, c) = v
        else:
            fields = v.split('_')
            r = float(fields[-2])
            c = float(fields[-1])
            labels[v] = f'({r}, {c})'
    print('Placing labels...')
    nx.draw_networkx_labels(G, pos, labels, font_size = 4)
print('Storing the figure...')    
plt.axis("off")
plt.savefig(filename.replace('.json', '_graph.png'), bbox_inches = "tight", dpi = 150)
plt.clf()
