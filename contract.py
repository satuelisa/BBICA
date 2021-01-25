from dumps import store, load
import simplejson as json
import networkx as nx
import os

threshold = 0.0001 # how close do frame-level cell centers have to be to merge them (in dec lat/lon)
contract = False # whether to merge near-by cells from different frames
for dataset in ['aug27b', 'jul25b']:
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
                    G.add_node(node, pos = pos[node], value = vals[node])
                for n1, n2 in Gg.edges():
                    G.add_edge(n1, n2)
        if contract:
            print(f'Contracting with threshold {threshold} (this takes a long time)...')
            original = [n for n in G.nodes()]
            if G.has_node(n1): # not yet contracted
                (x1, y1) = pos[n1]
                f1 = n1.split('_')[1] # name of frame
                for n2 in original:                        
                    if G.has_node(n2): # not yet contracted
                        if n1 != n2:
                            f2 = n2.split('_')[1]
                            if f1 != f2: # from different frames
                                (x2, y2) = pos[n2]
                                dx = x1 - x2
                                dy = y1 - y2
                                d = sqrt(dx**2 + dy**2) 
                                if d < threshold:
                                    G = nx.contracted_nodes(G, n1, n2)
                                    print(f'Contracted {n1} with {n2} (distance {d})')
                                    v = average(vals[n1], vals[n2])                                    
                                    G.nodes[n1]['value'] = v
                                    p = average(pos[n1], pos[n2]) 
                                    G.nodes[n1]['pos'] = p
        print(f'Storing a global graph of kind {kind}...')                                            
        store(G, f'{dataset}_global_{kind}.json')

