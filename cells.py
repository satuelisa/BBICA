from scipy.spatial import Voronoi, voronoi_plot_2d # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html
from matplotlib.collections import LineCollection
from collections import defaultdict
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from dumps import store, load
from extract import crop
from random import random
import simplejson as json
import networkx as nx
from os import path
import matplotlib
import os.path
import os

datasets = ['aug27b', 'jul25b']
bbox = ((-99.90024722222223, -99.89772500000001), (24.20827222222222, 24.21)) # lon, lat (bb.py from BarkBeetle)

### ADJUSTABLE PARAMETERS ###
goal = 100 # how many cells in terms of latitude 
#############################

overwrite = False # overwrite all output
verbose = False # print additional debug info
SAVE_ALL = True # save (tons of) images of individual cells
INDIVIDUAL = True # save graphs of individual frames
histos = False # output altitude and angle data to stdout

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8

def crop(filename, polygon):
    # https://stackoverflow.com/questions/22588074/polygon-crop-clip-using-python-pil
    original = Image.open(filename)
    (w, h) = original.size
    mask = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill = (0, 0, 0, 255), outline = None)
    joint = Image.new('RGBA', (w, h))
    joint.paste(original.crop((0, 0, w, h)), (0, 0), mask)
    joint.load()
    # https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil
    image_data = np.asarray(joint)
    image_data_bw = image_data.max(axis = 2)
    non_empty_columns = np.where(image_data_bw.max(axis = 0)>0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis = 1)>0)[0]
    cb = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    data = image_data[cb[0] : (cb[1] + 1), cb[2] : (cb[3] + 1) , :]
    return Image.fromarray(data, "RGBA")

# https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle 
def sign (triangle):
    p1, p2, p3 = triangle
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

# https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle 
def inside(triangle, point):
    t1, t2, t3 = triangle
    d1 = sign(pt, t1, t2)
    d2 = sign(pt, t2, t3)
    d3 = sign(pt, t3, t1)
    neg = d1 < 0 or d2 < 0 or d3 < 0
    pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (neg and pos)

# https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters
from math import pi, sin, cos, atan2, sqrt, floor, ceil
def dist(c1, c2):
    lat1, lon1 = c1
    lat2, lon2 = c2
    dLat = lat2 * pi / 180 - lat1 * pi / 180
    dLon = lon2 * pi / 180 - lon1 * pi / 180
    a = sin(dLat / 2) * sin(dLat / 2) + cos(lat1 * pi / 180) * cos(lat2 * pi / 180) * sin(dLon / 2) * sin(dLon / 2)
    return 6378.137 * 2 * atan2(sqrt(a), sqrt(1 - a)) * 1000

def parse(s, decimal = True): # from MinDec to DegDec
    ref = s[-1]
    fields = s[:-1].split(',')
    degrees = float(fields[0])
    minutes = float(fields[1])
    integer = floor(minutes)
    minutes, seconds = integer, 60 * (minutes - integer)
    if decimal:
        return (degrees + minutes / 60 + seconds / 60**2) * (-1 if ref in ['S','W'] else 1)
    else:
        return str(degrees)+ 'D' + str(minutes) + 'M' + str(seconds) + 'S ' + ref 

# exiftool -xmp -b <filename>
# https://stackoverflow.com/questions/6822693/read-image-xmp-data-in-python
# python3 -m pip install python-xmp-toolkit
from libxmp.utils import file_to_dict
def metadata(frame):
    xmp = file_to_dict(frame)
    info = dict()
    for category in xmp:
        gr = xmp[category]
        for data in gr:
            info[data[0]] = data[1:]
    return info

import numpy as np

# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def computeArea(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def frac(fs):
    f = fs.split('/')
    return int(f[0]) / int(f[1])

# https://www.earthdatascience.org/courses/use-data-open-source-python/multispectral-remote-sensing/vegetation-indices-in-python/
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import exiftool
import base64
import struct
modern = False #  images taken with an old firmware

# sensor specs
resolution = {'RGB': (4608, 3456),
              'NIR': (1280, 960),
              'RED': (1280, 960)}
def extract(frame, decimal = True):
    filename = frame + '_RGB.JPG'
    info = metadata(filename)
    if verbose:
        for key in info:
            print(key)
    yaw = float(info['Camera:Yaw'][0])
    pitch = float(info['Camera:Pitch'][0])
    roll = float(info['Camera:Roll'][0])
    assert int(info['exif:FocalPlaneResolutionUnit'][0]) == 4 # mm
    w = int(info['exif:PixelXDimension'][0])
    assert w == resolution['RGB'][0]
    wm = w * frac(info['exif:FocalPlaneXResolution'][0]) / 100000 # image width in meters 
    h = int(info['exif:PixelYDimension'][0])
    assert h == resolution['RGB'][1]    
    hm = h * frac(info['exif:FocalPlaneYResolution'][0]) / 100000 # image height in meters
    lat = info['exif:GPSLatitude'][0]
    lon = info['exif:GPSLongitude'][0]
    assert round(frac(info['exif:FocalLength'][0]), 3) == 4.88 # sensor specs
    assert int(info['exif:GPSAltitudeRef'][0]) == 0 # above sea level
    alt = frac(info['exif:GPSAltitude'][0])
    exif = Image.open(filename)._getexif() # just checking consistency
    if histos: 
        print(dataset, yaw, alt) 
    if exif is not None:
        for key, value in exif.items():
            name = TAGS.get(key, key)
            if name == 'ExifImageWidth':
                assert w == int(value) 
            elif name == 'ExifImageHeight':
                assert h == int(value)
    latC = parse(lat, decimal)
    lonC = parse(lon, decimal)
    filename = filename.replace('RGB.JPG', 'NIR.TIF')
    redfile = filename.replace('NIR', 'RED')
    NIR = np.asarray(Image.open(filename)).flatten() 
    Red = np.asarray(Image.open(redfile)).flatten()
    data = (NIR, Red)
    return lonC, latC, wm, hm, yaw, data

def average(d1, d2):
    means = []
    assert len(d1) == len(d2)
    for (v1, v2) in zip(d1, d2):
        means.append((v1 + v2) / 2)
    return means

def merge(sources, loc = None):
    means = []
    for s in sources:
        values = []
        for v in range(len(s)):
            if loc is None or v in loc:
                values.append(s[v])
        means.append(np.nanmean(values))
    return means

def combine(pixels, channel = None):
    nontransparent = pixels[pixels[:, : ,3] == 255, : 3];
    if channel is not None:
        return np.mean([p[channel] for p in nontransparent])
    return np.mean([np.mean(p) for p in pixels]) # grayscale from RGB (ignore A)

latSpan = bbox[1][1] - bbox[1][0]
lonSpan = bbox[0][1] - bbox[0][0]
step = latSpan / goal
area = step**2 # square area

def step(kind):
    return sqrt(area if kind == 1 else (4 * area / sqrt(3) if kind == 2 else 2 * area / 3**1.5))
    
def grid(kind):
    unit = step(kind)
    half = unit / 2
    h = sqrt(3) * half 
    hw = unit / 2
    hh = h / 2
    sixth = h / 6
    dy = unit if kind == 1 else (h if kind == 2 else 2 * h)
    dx = unit if kind == 1 else (half if kind == 2 else 3 * unit / 2)
    yp = 0
    y = bbox[1][0]
    N = defaultdict(list)
    centers = defaultdict()
    up = False
    while y < bbox[1][1]:
        xp = 0 if kind < 3 else yp % 2 # no alternating initial displacement for squares
        x = bbox[0][0]
        while x < bbox[0][1]:
            xc = x + half
            if kind == 1: # square
                N[(yp, xp)] = [(yp - 1,  xp), (yp + 1,  xp), (yp,  xp - 1), (yp,  xp + 1)]
                yc = y + half
            elif kind == 2: # triangle
                N[(yp, xp)] = [(yp, xp - 1), (yp, xp + 1)]
                up = ((yp % 2) or (xp % 2)) and (not (yp % 2) or not (xp % 2))
                if up:
                    N[(yp, xp)].append((yp - 1, xp))                
                yc = y + hh - sixth * (2 * up - 1)
            elif kind == 3: # hexagon
                N[(yp, xp)].append((yp - 1, xp - 1))
                N[(yp, xp)].append((yp - 1, xp + 1))
                N[(yp, xp)].append((yp + 1, xp - 1))
                N[(yp, xp)].append((yp + 1, xp + 1))                    
                N[(yp, xp)].append((yp, xp - 1))                
                N[(yp, xp)].append((yp, xp + 1))
                if xp % 2 == 0: 
                    N[(yp, xp)].append((yp - 1, xp + 2))
                else:
                    N[(yp, xp)].append((yp - 1, xp - 2))
                yr = y + half
                yc = yr if (xp - (yp % 2)) % 2 == 0 else yr + h
            else:
                print('Unknown grid type, exiting')
                quit()
            centers[(yp, xp)] = (xc, yc, up) # store center
            x += dx # proceed horizontally
            xp += 1
        y += dy # proceed vertically
        yp += 1
    return centers, N

# https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters
import matplotlib.patches as patches
import platform
ER = 6378.137 # earth radius in km
MD = (1 / ((2 * pi / 360) * ER)) / 1000

def cell(xc, yc, kind, up = True):
    unit = step(kind)
    sh = unit / 2
    height = sqrt(3) * sh
    sequence = []
    if kind == 1: # square cell
        sequence.append((xc - sh, yc - sh))
        sequence.append((xc + sh, yc - sh))
        sequence.append((xc + sh, yc + sh))
        sequence.append((xc - sh, yc + sh))        
    elif kind == 2: # triangle cell
        hs = height / 3
        if up:
            yt = yc + 2 * hs 
            yb = yc - hs 
            sequence.append((xc - sh, yb))
            sequence.append((xc, yt))
            sequence.append((xc + sh, yb))
        else:
            yt = yc + hs
            yb = yc - 2 * hs        
            sequence.append((xc - sh, yt))
            sequence.append((xc, yb))
            sequence.append((xc + sh, yt))        
    elif kind == 3: # hexagonal cell
        sequence.append((xc - sh, yc + height))
        sequence.append((xc + sh, yc + height))
        sequence.append((xc + unit, yc))        
        sequence.append((xc + sh, yc - height))
        sequence.append((xc - sh, yc - height))
        sequence.append((xc - unit, yc))
    return sequence

# https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
    return qx, qy

# https://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
def offset(lon, lat, dx, dy):
    dLat = dy / (1000 * ER)
    dLon = dx / (1000 * ER * cos(pi * lat / 180))
    return lon + dLon * 180/pi, lat + dLat * 180 / pi, 

def bounds(lonC, latC, xm, ym):
     xh = xm / 2
     yh = ym / 2
     x1, y1 = offset(lonC, latC, -xh, -yh)
     x2, y2 = offset(lonC, latC, xh, yh)
     return x1, y1, x2, y2
     
def pos2pixel(cell, lonC, latC, xm, ym, a, img = 'NIR'):
    x1, y1, x2, y2 = bounds(lonC, latC, xm, ym) # bbox lon / lat
    assert lonC > x1 and lonC < x2
    assert latC > y1 and latC < y2
    xc = (x1 + x2) / 2 # center lon
    yc = (y1 + y2) / 2 # center lat
    dx = MD * (wm / 2) / cos(yc * (pi / 180)) # offset lon
    dy = MD * (hm / 2) # offset lat
    angle = 2 * pi - (2 * pi * (a / 360)) # rotate the opposite way to rectify the frame horizontally
    translated = []
    sx = x2 - x1
    assert sx > 0
    sy = y2 - y1 
    assert sy > 0
    center = (xc, yc) # center in lon, lat
    for (x, y) in cell:
        xr, yr = rotate(center, (x, y), angle) # rotate the points around the center
        xt = int(round(resolution[img][0] * (xr - x1) / sx)) # translate relative positions into pixels
        yt = int(round(resolution[img][1] * (yr - y1) / sy))
        if not (xt > 0 and xt < resolution[img][0] and yt > 0 and yt < resolution[img][1]):
            return None # incomplete
        translated.append((xt, yt))
    assert len(cell) == len(translated) 
    return translated 

def rectangle(wm, hm, xc, yc, color = 'blue', alpha = 0.4, linewidth = 2):
    dx = MD * (wm / 2) / cos(yc * (pi / 180))  
    dy = MD * (hm / 2)
    return patches.Rectangle((xc - dx, yc - dy), 2 * dx, 2 * dy, edgecolor = color, facecolor = 'none', alpha = alpha, lw = linewidth)

for dataset in datasets:           
    directory = r'/elisa/Dropbox/Research/Topics/Arboles/CA/RAW/' + dataset + '/images/'
    osys = platform.system()
    if osys == 'Linux':
        directory = '/home' + directory
    elif osys == 'Darwin' :
        directory = '/Users' + directory
    else:
        print('Unsupported operating system')
        quit()
    centers = dict()
    rect = dict()
    frames = dict()
    for filename in os.listdir(directory):
        if filename.endswith('_RGB.JPG'):
            if path.exists(directory + filename.replace('RGB.JPG', 'NIR.TIF')) \
               and path.exists(directory + filename.replace('RGB.JPG', 'RED.TIF')):
                lonC, latC, wm, hm, yaw, d = extract(directory + filename[:-8])
                pos = (lonC, latC)
                frames[filename] = d
                (x, y) = pos 
                rect[filename] = (wm, hm, lonC, latC,yaw)
                lonIn = x > bbox[0][0] and x < bbox[0][1]
                latIn = y > bbox[1][0] and y < bbox[1][1]
                if latIn and lonIn:
                    centers[filename] = (x, y)
            else:
                if verbose:
                    print('# missing multispectral data for ' + filename)
    print('Processing', dataset)
    pos = list(centers.values())
    assert len(pos) >= 30 # at least thirty meaningful frames 
    x = [c[0] for c in pos] 
    y = [c[1] for c in pos]
    cells = Voronoi(pos)
    fig = voronoi_plot_2d(cells, show_vertices = False, line_colors = 'blue',
                          line_width = 2, line_alpha = 0.5, point_size = 3)
    plt.xlim(bbox[0])
    plt.ylim(bbox[1])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')    
    plt.savefig(dataset + '_voronoi.png')
    plt.clf()
    cl = []
    rl = []
    n = len(cells.points)
    areas = dict()
    for i in range(n):
        for filename in centers:
            if all(centers[filename] == cells.points[i]):
                frames[i] = filename
                break
        r = cells.regions[cells.point_region[i]]
        if -1 not in r:
            polygon = []
            complete = True
            for p in r:
                (x, y) = cells.vertices[p]
                if x > bbox[0][0] and x < bbox[0][1] and y > bbox[1][0] and y < bbox[1][1]:
                    polygon.append((x, y))
                else:
                    complete = False
            if complete:
                assert len(polygon) > 2
                areas[i] = computeArea(polygon)
    data = list(areas.values())
    expected = np.median(data)
    variation = 2 * expected
    G = nx.Graph()
    Gg = nx.Graph()
    merged = dict()
    for i in range(n):
        if i in areas:
            a = areas[i]
            if a > expected - variation and a < expected + variation:
                (x, y) = cells.points[i]
                label = frames[i]
                merged[label] = merge(frames[label])
                Gg.add_node(label, pos = (x, y), value = merged[label])
    for i in range(n):
        if Gg.has_node(frames[i]):
            iC = cells.regions[cells.point_region[i]]
            assert -1 not in iC
            for j in range(i + 1, n):
                if Gg.has_node(frames[j]):                    
                    jC = cells.regions[cells.point_region[j]]
                    assert -1 not in jC                        
                    shared = set(iC) & set(jC)
                    k = len(shared)
                    if k == 2: # a shared border 
                        Gg.add_edge(frames[i], frames[j])
                    elif k > 2: # overlapping
                        Gg = nx.contracted_nodes(Gg, frames[i], frames[j])
                        Gg.nodes[frames[i]]['value'] = average(merged[frames[i]], merged[frames[j]]) 
    store(Gg, f'{dataset}.json')
    Gg = load(f'{dataset}.json')    
    pos = nx.get_node_attributes(Gg, 'pos')
    value = nx.get_node_attributes(Gg, 'value')
    norm = matplotlib.colors.Normalize(vmin = 0.0, vmax = 3.0) # https://stackoverflow.com/questions/28752727/map-values-to-colors-in-matplotlib
    mapper = plt.cm.ScalarMappable(norm = norm, cmap = plt.cm.RdYlGn) # https://www.neonscience.org/calc-ndvi-tiles-py
    color = []
    for v in Gg:
        color.append(mapper.to_rgba(np.mean(value[v]))) # use the average when multiple channels
    nx.draw_networkx_nodes(Gg, pos, node_size = 12, node_shape='o', node_color = color, cmap = plt.cm.RdYlGn, linewidths = None, edgecolors = 'black') 
    nx.draw_networkx_edges(Gg, pos, width = 1, edge_color = 'black')
    plt.axis("off")
    plt.xlim(bbox[0])
    plt.ylim(bbox[1])
    plt.savefig(f'{dataset}_graph.png', bbox_inches = "tight", dpi = 150)
    plt.clf()
    for kind in [1, 2, 3]:
        print(f'Extracting type {kind} cells for {dataset}')        
        positions, neighborhoods = grid(kind)
        cx = [v[0] for v in positions.values()]
        cy = [v[1] for v in positions.values()]
        values = dict()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(bbox[0])
        ax.set_ylim(bbox[1])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.scatter(x, y, marker = 'o', color = 'black')
        ax.scatter(cx, cy, marker = 'o', color = 'red', alpha = 0.3)
        ax.set_xlim(bbox[0])
        ax.set_ylim(bbox[1])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        zones = dict()
        for c in positions:
            (xc, yc, up) = positions[c]
            p = cell(xc, yc, kind, up)
            zones[c] = p
            polygon = patches.Polygon(p, edgecolor = 'green', facecolor = 'none', alpha = 0.7, lw = 1)        
            ax.add_patch(polygon)
        print(f'Polygons of kind {kind} computed for {dataset}')
        td = ax.transData # https://stackoverflow.com/questions/4285103/matplotlib-rotating-a-patch
        for filename in rect:
            if INDIVIDUAL:
                target = f'{dataset}_{filename}_cells_{kind}.json'
                if not overwrite and path.exists(target):
                    print(f'A graph of type {kind} for {filename} of {dataset} already exists')
                    continue # no need to reprocess (delete this for a full wipe)
            Gf = nx.Graph()
            wm, hm, lonC, latC, a = rect[filename]
            r = rectangle(wm, hm, lonC, latC)
            (xf, yf) = centers[filename]
            rotation = matplotlib.transforms.Affine2D().rotate_deg_around(xf, yf, a - 90) + td # west is zero, north is no rotation
            r.set_transform(rotation)
            ax.add_patch(r)
            for c in positions:
                (row, column) = c
                p = zones[c]
                coords = pos2pixel(p, lonC, latC, wm, hm, a, 'RGB' if 'RGB' in filename else 'NIR')
                if coords is not None: # whole cell
                    contents = crop(directory + filename, coords)
                    if SAVE_ALL:
                        contents.save(f'{dataset}_{filename}_{kind}_{row}_{column}.png')
                    d = np.asarray(contents) # pixels
                    v = (combine(d, 0), # R
                         combine(d, 1), # G 
                         combine(d, 2)) # B
                    (xc, yc, up) = positions[c]
                    if INDIVIDUAL:
                        Gf.add_node(f'{dataset}_{filename}_{kind}_{row}_{column}', pos = (xc, yc), color = v, value = v)
                    G.add_node(f'{dataset}_{filename}_{kind}_{row}_{column}', pos = (xc, yc), color = v, value = v)                    
                    values[c] = v
            for c in neighborhoods:
                (r1, c1) = c
                n1 = f'{dataset}_{filename}_{kind}_{r1}_{c1}'
                for nc in neighborhoods[c]:
                    (r2, c2) = nc                    
                    n2 = f'{dataset}_{filename}_{kind}_{r2}_{c2}'                    
                    if G.has_node(n1) and G.has_node(n2) and not G.has_edge(n1, n2) and not G.has_edge(n2, n1): # undirected
                        G.add_edge(n1, n2)                        
                        if INDIVIDUAL:
                            Gf.add_edge(n1, n2)
            if INDIVIDUAL:
                store(Gf, target)
                print(f'Graph exported for {filename} of {dataset}')
        ax.ticklabel_format(useOffset=False)
        plt.savefig(f'{dataset}_cells_{kind}.png',  bbox_inches = "tight", dpi = 150)
        plt.clf()

