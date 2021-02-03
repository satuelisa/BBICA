from dumps import store, load
from math import sqrt, fabs
from matplotlib import path
import simplejson as json
import networkx as nx
import os

from local import datasets, zone, average

validate = False # check if matching (row, col) correspond to the same coordinates
epsilon = 0.000001 
contract = True # whether to merge same-position cells from different frames
prune = True # whether to limit to the zone of interest
permitted = None
if prune:
    permitted = path.Path(zone)    
for dataset in datasets:
    print(f'Processing {dataset}...')
    for kind in [1, 2, 3]:
        print(f'Creating a global graph of kind {kind}...')        
        G = nx.Graph()
        for filename in os.listdir('.'):
            if dataset in filename and 'IMG' in filename and filename.endswith(f'_cells_{kind}.json'):
                Gg = load(filename)        
                pos = nx.get_node_attributes(Gg, 'pos')
                vals = nx.get_node_attributes(Gg, 'value')
                for node in Gg.nodes():
                    (px, py) = pos[node]
                    if not prune or  permitted.contains_points([(px, py)]): # not in the zone of interest                    
                        G.add_node(node, pos = pos[node], value = vals[node])
                for n1, n2 in Gg.edges():
                    if not prune or (G.has_node(n1) and G.has_node(n2)):
                        G.add_edge(n1, n2)
        if contract:
            pos = nx.get_node_attributes(G, 'pos')
            vals = nx.get_node_attributes(G, 'value')            
            print(f'Contracting (this takes a long time)...')
            original = [n for n in G.nodes()]
            for n1 in original:
                if G.has_node(n1): # not yet contracted
                    (x1, y1) = pos[n1]
                    d1 = n1.split('_')
                    f1 = ' '.join(d1[2:5]) # frame ID                    
                    r1 = d1[-2]
                    c1 = d1[-1]
                    for n2 in original:
                        if n1 != n2: # not the same node
                            d2 = n2.split('_')
                            f2 = ' '.join(d2[2:5]) # frame ID
                            if f1 != f2: # from different frames
                                if G.has_node(n2): # not yet contracted
                                    r2 = d2[-2]
                                    c2 = d2[-1]
                                    if r1 == r2 and c1 == c2:
                                        (x2, y2) = pos[n2]
                                        if validate:
                                            dx = x1 - x2
                                            dy = y1 - y2
                                            d = sqrt(dx**2 + dy**2)
                                            assert fabs(d) < epsilon # ensure similar coordinates
                                        G = nx.contracted_nodes(G, n1, n2)
                                        print(f'Contracted {n1} with {n2}')
                                        v = average(vals[n1], vals[n2])                                    
                                        G.nodes[n1]['value'] = v
                                        p = average(pos[n1], pos[n2]) 
                                        G.nodes[n1]['pos'] = p
        print(f'Storing a global graph of kind {kind}...')                                            
        store(G, f'{dataset}_global_{kind}.json')
