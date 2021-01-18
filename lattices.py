from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from dumps import store, load
from random import random
from math import sqrt
import networkx as nx

colors = [['#d7ff1','#aafcb8','#8cd790'],
          ['#77af9c','#285943','#8f8fa0'],
          ['#acada1','#cdcec8','#e8e8f4'],
          ['#fff1D7', '#fcb8aa', '#d78c90']]

def rgb(hexstr): 
    return int(hexstr[1:3], 16), int(hexstr[3:5], 16), int(hexstr[5:], 16), 

verbose = False

def assign(kind, area, data = None, w = 1000, h = 1000, cr = 3, to = 10):
    G = nx.Graph()
    step = round(sqrt(area if kind == 1 else (4 * area / sqrt(3) if kind == 2 else 2 * area / 3**1.5)))
    half = step // 2
    rh = int(round((sqrt(3 * half**2))))
    dd  = half / rh
    canvas = Image.new('RGB', (w, h))
    draw = ImageDraw.Draw(canvas)    
    p = canvas.load()
    rs = step if kind == 1 else rh 
    cs = step if kind == 1 else (half if kind == 2 else 3 * step)
    co = kind
    ro = kind
    fr = -1 if kind == 3 else 0
    sizes = set()
    for row in range(fr, h // rs + ro):
        for column in range(w // cs + co):
            color = colors[(column + (0 if kind != 3 else row + column)) % 4][(row + (0 if kind != 3 else row - column)) % 3]
            up = ((row % 2) or (column % 2)) and (not (row % 2) or not (column % 2))
            incomplete = False
            pixels = set()
            counter = 0
            if kind == 1:
                for y in range(row * rs, (row + 1) * rs):
                    for x in range(column * cs, (column + 1) * cs):
                        if x >= 0 and x < w and y >= 0 and y < h:                        
                            p[x, y] = rgb(color)
                            pixels.add(w * y + x) # flattened array positions
                            counter += 1
                        else:
                            incomplete = True
            elif kind == 2: 
                ys = row * rs
                for dy in range(rh):
                    delta = round(dy * dd)
                    if up:
                        delta = round(half - dy * dd)
                    xs = column * half
                    for dx in range(-delta, delta):
                        x = xs + dx
                        y = ys + dy
                        if x >= 0 and x < w and y >= 0 and y < h:
                            p[x, y] = rgb(color)
                            pixels.add(w * y + x) # flattened array positions                            
                            counter += 1
                        else:
                            incomplete = True
            elif kind == 3:
                ys = row * rs - (0 if row % 2 else 2 * rh)             
                for dy in range(2 * rh):
                    xc = column * cs - half + (0 if row % 2 else 3 * half)
                    span = (half + round(dy * dd)) if dy <= rh else (step + round((rh - dy) * dd))
                    for rx in range(xc - span, xc + span):
                        x = rx 
                        y = ys + dy
                        if x >= 0 and x < w and y >= 0 and y < h:
                            p[x, y] = rgb(color)
                            pixels.add(w * y + x) # flattened array positions                                                        
                            counter += 1
                        else:
                            incomplete = True
            if not incomplete:
                yc = row * rs + rs // 2 if kind == 1 else (row * rs + (rh // 3 if up else (2 * rh) // 3) if kind == 2 else row * rs - (0 if row % 2 else 2 * rh) + rh)
                xc = column * cs + cs // 2 if kind == 1 else (column * cs if kind == 2 else column * cs - half + (0 if row % 2 else 3 * half))
                G.add_node((row, column), pos = (xc, yc), color = color, value = random())
                draw.text((xc - to, yc - to), f'({row}, {column})')
                for x in range(xc - cr, xc + cr):
                    for y in range(yc - cr, yc + cr):
                        p[x, y] = (0, 0, 0) # black centers
                if verbose and counter not in sizes:
                    print(row, column)
                sizes.add(counter)
    if verbose:
        print(sizes, G.order())

    for row in range(fr, h // rs + ro):
        for column in range(w // cs + co):
            v1 = (row, column)
            if not G.has_node(v1):
                continue
            N = []
            if kind == 1:
                N = [(row - 1,  column), (row + 1,  column), (row,  column - 1), (row,  column + 1)]
            elif kind == 2:
                N = [(row, column - 1), (row, column + 1)]
                up = ((row % 2) or (column % 2)) and (not (row % 2) or not (column % 2))
                if up: 
                    N.append((row - 1, column))
            else:
                N.append((row - 1, column + 1))
                N.append((row + 1, column - 1))
                N.append((row - 2, column))
                if row % 2:
                    N.append((row + 1, column))
                    N.append((row - 1, column - 2))
                    N.append((row + 3, column))
                else:
                    N.append((row - 3, column + 1))
                    N.append((row - 1, column))
                    N.append((row + 1, column + 2))
            for v2 in N:
                if G.has_node(v2) and not G.has_edge(v1, v2):
                        G.add_edge(v1, v2)
    store(G, f'test_{kind}.json')
    G = load(f'test_{kind}.json')
    if verbose:
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
            fields = v[1:-1].split(',')
            r = int(fields.pop(0))
            c = int(fields.pop(0))
        labels[v] = f'({r}, {c})'
    nx.draw_networkx_labels(G, pos, labels, font_size = 4)
    plt.axis("off")
    plt.savefig(f'test_{kind}_graph.png', bbox_inches="tight", dpi=150)
    plt.clf()
    canvas.save(f'test_{kind}.png')

area = 5000
for kind in [3, 2, 1]:
    assign(kind, area)    
