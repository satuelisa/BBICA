# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html
from math import pi, sin, cos, tan, atan2, sqrt, floor, ceil, fabs
from scipy.spatial import Voronoi, voronoi_plot_2d 
from matplotlib.collections import LineCollection
from PIL.ExifTags import TAGS, GPSTAGS
import matplotlib.patches as patches
from collections import defaultdict
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from dumps import store, load
from pylab import rcParams
from random import random
import simplejson as json
import networkx as nx
from os import path
import numpy as np
import matplotlib
import exiftool
import platform
import base64
import struct
import os.path
import os

def average(vectors):
    n = len(vectors)
    k = len(vectors[0]) 
    result = [0] * k
    for v in vectors:
        for i in range(k):
            result[i] += v[i]
    return [ r / n for r in result ]

def dec2rgb(color):
    return [ c / 256 for c in color ]

flightAltitude = 100 # meters, set on the drones
FieldOfView =  65.5 # degrees, drone property
fov = (2 * pi) * (FieldOfView /  360) # radians
FocalPlaneResolutionUnit = {
    '2': 25.4, # inch in the standard
    '3': 10, # centimeter in the standard
    '4': 1 # mm for Pix4D
}

# python3 -m pip install python-xmp-toolkit
from libxmp.utils import file_to_dict

# local files
from extract import crop
from local import datasets, bbox, zone, channels, shapes
rcParams['figure.figsize'] = 12, 8

### ADJUSTABLE PARAMETERS ###
goal = 60
MINCELLS = 30 # how many cells must have content for frame to be included
threshold = 20 # discarding of grayish tones 
# we used 20 for the squares (kind 1) originally
# ran the whole thing with 25 but the ML was not impressive
contentProp = 0.6 # skip cell where this proportion of the pixels were discarded as gray
# we used 0.5 for the squares (kind 1) originally
# (0.3 is too severe)
# (0.6 did not yield good results)

# https://rechneronline.de/earth-radius/#:~:text=Earth%20radius%20at%20sea%20level,(3958.756%20mi)%20on%20average.
# using latitude 24.2091 and altitude 2230 m (de los metadatos)
altitude = 2230 # m
altThr = 25 # m (threshold)
planar = 7 # degrees, how tilted the drone can be
mm2m = 100 * 10 # millimeters in a meter (m -> cm -> mm)
EARTH = 6376.796 # km
MD = (1 / ((2 * pi / 360) * EARTH)) / 1000 # meters

resolution = {True: (4608, 3456), False: (1280, 960)} # sensor specs (True for RGB)
modern = False #  images taken with an old firmware

# output control flags
overwrite = False # overwrite all output
verbose = True # print additional debug info
SAVE = set() # which cells to save (kind, row, column)
RAW = False # save the raw cell (tons of times)
ALTITUDE = False # print altitude from metadata
SHOW_MASK = False # show filtering mask (tons of times)
INDIVIDUAL = True # save graphs of individual frames (needed for contract.py)
DRAW_CELLS = True # draw each cell border on the overview plot
DRAW_CELL_CENTER = True # draw each cell center
DRAW_FRAME_CENTER = True
DRAW_FRAME_BORDER = True
SAVE_STAGES = False
histos = False # output altitude and angle data to stdout if True
overview = False # no cell-level computations if True

def crop(filename, polygon, mask = None):
    # https://stackoverflow.com/questions/22588074/polygon-crop-clip-using-python-pil
    original = Image.open(filename)
    (w, h) = original.size
    if 'TIF' in filename: # for the TIF images that are single-channel
        rpl = filename.replace('TIF', 'png') # make those into png
        if not path.exists(rpl):
            data = np.asarray(original).flatten(order = 'C') # data into 1D
            data = np.round(np.interp(data, (data.min(), data.max()), (0, 255))) # normalize and discretize
            gs  = Image.fromarray(data.reshape(h, w, order = 'C'))
            rgb = gs.convert('RGB')
            a = Image.new('L', (w, h), 255) # alpha channel (for masking)
            rgb.putalpha(a)
            rgb.save(rpl)
        original = Image.open(rpl)
    cellmask = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(cellmask)
    draw.polygon(polygon, fill = (0, 0, 0, 255), outline = None)
    joint = Image.new('RGBA', (w, h))
    joint.paste(original.crop((0, 0, w, h)), (0, 0), cellmask)
    joint.load()
    # https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil
    image_data = np.asarray(joint)
    image_data_bw = image_data.max(axis = 2)
    non_empty_columns = np.where(image_data_bw.max(axis = 0) > 0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis = 1) > 0)[0]
    cb = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    data = image_data[cb[0] : (cb[1] + 1), cb[2] : (cb[3] + 1) , :]
    cell = Image.fromarray(data, "RGBA")
    if RAW: # overwrites the channels as it goes
        cell.save('raw.png')
    if 'RGB' in filename: # filter out pixels that are gray in tone
        (w, h) = cell.size
        total = w * h
        skip = 1
        pixels = cell.load()
        for row in range(h):
            for col in range(w):
                r, g, b, a = pixels[col, row]
                if max(r, g, b) - min(r, g, b) <= threshold: # too gray
                    cell.putpixel((col, row), (0, 0, 0, 0)) # transparent black
                    skip += 1
        if skip / total > contentProp:
            if verbose: 
                print('SKIPPING', skip / total, 'is above', contentProp)
            return None
    else:
        if SHOW_MASK:
            cell.show()
        assert mask is not None
        (wkm, hkm) = mask.size
        (w, h) = cell.size
        if w != wkm:
            mask = mask.resize((w, h))
        joint = Image.new('RGBA', (w, h))
        joint.paste(cell, (0, 0), mask)
        cell = joint
        if SHOW_MASK:
            cell.show()
    return cell

def mask(img):
    (w, h) = img.size
    pixelmask = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pix = img.load()
    target = pixelmask.load()
    for row in range(h):
        for col in range(w):
            r, g, b, a = pix[col, row]
            if a > 0: # only let the pixels in the non-transparent positions through the mask
                target[col, row] = (0, 0, 0, 255)
    return pixelmask

# https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle 
def sign(triangle):
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
def dist(c1, c2):
    lat1, lon1 = c1
    lat2, lon2 = c2
    dLat = lat2 * pi / 180 - lat1 * pi / 180
    dLon = lon2 * pi / 180 - lon1 * pi / 180
    a = sin(dLat / 2) * sin(dLat / 2) + cos(lat1 * pi / 180) * cos(lat2 * pi / 180) * sin(dLon / 2) * sin(dLon / 2)
    return EARTH * 2 * atan2(sqrt(a), sqrt(1 - a)) # km

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

# https://www.earthdatascience.org/courses/use-data-open-source-python/multispectral-remote-sensing/vegetation-indices-in-python/    
# exiftool -xmp -b <filename>
# https://stackoverflow.com/questions/6822693/read-image-xmp-data-in-python
def metadata(frame):
    xmp = file_to_dict(frame)
    info = dict()
    for category in xmp:
        gr = xmp[category]
        for data in gr:
            info[data[0]] = data[1:]
    return info

# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def computeArea(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def frac(fs):
    f = fs.split('/')
    return int(f[0]) / int(f[1])

def extract(frame, decimal = True):
    filename = frame + '_RGB.JPG' # the unenhanced JPG has the metadata
    info = metadata(filename)
    if verbose:
        for key in info:
            print(key)
    yaw = float(info['Camera:Yaw'][0]) # in degrees, frontal rotation
    pitch = float(info['Camera:Pitch'][0]) # in degrees, up/down nose 
    roll = float(info['Camera:Roll'][0]) # in degrees, side rotation
    if fabs(pitch) > planar or fabs(roll) > planar: # discard if not close to "flat"
        return None, None, None, None, None
    orient = int(info['tiff:Orientation'][0])
    assert orient == 3 # Rotate 180 is what we expect
    yaw -= 180
    ru = info['exif:FocalPlaneResolutionUnit'][0]
    resUnit = FocalPlaneResolutionUnit.get(ru, None)
    assert resUnit is not None # it needs to be 2, 3, or 4
    w = int(info['exif:PixelXDimension'][0])
    assert w == resolution[True][0]
    xr = frac(info['exif:FocalPlaneXResolution'][0])
    wm0 = w * xr * resUnit / mm2m # image width in m on the focal plane
    h = int(info['exif:PixelYDimension'][0])
    assert h == resolution[True][1]    
    yr = frac(info['exif:FocalPlaneYResolution'][0])
    hm0 = h * yr * resUnit / mm2m # image height in m on the focal plane
    wm = 2 * (flightAltitude * tan(fov / 2)) # image width in meters on assumed ground level
    hm = wm * (hm0 / wm0) # conserve the proportions from the focal plane
    lat = info['exif:GPSLatitude'][0]
    lon = info['exif:GPSLongitude'][0]
    assert round(frac(info['exif:FocalLength'][0]), 3) == 4.88 # sensor specs
    assert int(info['exif:GPSAltitudeRef'][0]) == 0 # above sea level
    alt = frac(info['exif:GPSAltitude'][0])
    if fabs(altitude - alt) > altThr:
        return None, None, None, None, None
    if ALTITUDE:
        print(alt)
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
    return lonC, latC, wm, hm, yaw

def combine(frames):
    pixels = np.asarray(contents['RGB'])
    rv = list()
    gv = list()
    bv = list()
    for r, g, b in pixels[pixels[:, : ,3] == 255, : 3]: # only non-transparent pixels
        rv.append(r)
        gv.append(g)
        bv.append(b)
    if len(rv) > 0:
        data = [np.mean(rv), np.mean(gv), np.mean(bv)] + [np.mean(np.asarray(contents[ch])) for ch in channels]        
        return [round(r) for r in data]
    return [None] 

latSpan = bbox[1][1] - bbox[1][0]
lonSpan = bbox[0][1] - bbox[0][0]
step = latSpan / goal
area = step ** 2 # square area

def step(kind):
    return sqrt(area if kind == 1 else (4 * area / sqrt(3) if kind == 2 else 2 * area / 3 ** 1.5))
    
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
def shape(xc, yc, kind, up = True):
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
    dLat = dy / (1000 * EARTH) # meters
    dLon = dx / (1000 * EARTH * cos(pi * lat / 180))
    return lon + dLon * 180 / pi, lat + dLat * 180 / pi, 

def bounds(lonC, latC, xm, ym):
     xh = xm / 2
     yh = ym / 2
     x1, y1 = offset(lonC, latC, -xh, -yh)
     x2, y2 = offset(lonC, latC, xh, yh)
     return x1, y1, x2, y2
     
def pos2pixel(cell, lonC, latC, xm, ym, a, rgb = True):
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
        xt = int(round(resolution[rgb][0] * (xr - x1) / sx)) # translate relative positions into pixels
        yt = int(round(resolution[rgb][1] * (yr - y1) / sy))
        if not (xt > 0 and xt < resolution[rgb][0] and yt > 0 and yt < resolution[rgb][1]):
            return None # incomplete
        translated.append((xt, yt))
    assert len(cell) == len(translated) 
    return translated 

def rectangle(wm, hm, xc, yc):
    dx = MD * (wm / 2) / cos(yc * (pi / 180))  
    dy = MD * (hm / 2)
    return xc - dx, yc - dy, 2 * dx, 2 * dy

for dataset in datasets:           
    log = open(f'log_{dataset}.txt', 'w')
    directory = r'/elisa/Dropbox/Research/Topics/Arboles/CA/RAW/' + dataset + '/images/'
    osys = platform.system()
    if osys == 'Linux':
        directory = '/home' + directory
    elif osys == 'Darwin' :
        directory = '/Volumes/' + directory.replace('elisa', 'dropbox')
#        directory = '/Users' + directory
    else:
        print('Unsupported operating system')
        quit()
    centers = dict()
    frameRGB = dict()
    for filename in os.listdir(directory + '/enhanced'):
        if filename.endswith('_RGB.png'):
            # process only if all channels are present
            if path.exists(directory + filename.replace('RGB.png', 'NIR.TIF')) \
               and path.exists(directory + filename.replace('RGB.png', 'RED.TIF')) \
               and path.exists(directory + filename.replace('RGB.png', 'GRE.TIF')) \
               and path.exists(directory + filename.replace('RGB.png', 'REG.TIF')):
                lonC, latC, wm, hm, yaw = extract(directory + filename[:-8])
                if lonC is None:
                    continue # do NOT process frames from initial flight altitudes
                pos = (lonC, latC)
                (x, y) = pos 
                frameRGB[filename] = (wm, hm, lonC, latC, yaw)
                lonIn = x > bbox[0][0] and x < bbox[0][1]
                latIn = y > bbox[1][0] and y < bbox[1][1]
                if latIn and lonIn:
                    centers[filename] = (x, y)
            else:
                if verbose:
                    print('# missing multispectral data for ' + filename)
    print('Processing', dataset)
    pos = list(centers.values())
    fc = len(pos)
    assert fc >= 20 # at least this many meaningful frames
    print(f'Combining information from {fc} frames')
    x = [c[0] for c in pos] 
    y = [c[1] for c in pos]
    cells = Voronoi(pos)
    fig, ax = plt.subplots()
    voronoi_plot_2d(cells, ax = ax, show_vertices = False, line_colors = 'blue',
                          line_width = 2, line_alpha = 0.5, point_size = 3)
    plt.xlim(bbox[0])
    plt.ylim(bbox[1])
    square = patches.Polygon(np.array(zone), edgecolor = 'black', facecolor = 'none', alpha = 0.3, lw = 2) 
    ax.add_patch(square)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(dataset + '_voronoi.png')
    plt.clf()
    print('Voronoi cells drawn.')
    cl = []
    rl = []
    n = len(cells.points)
    areas = dict()
    for i in range(n):
        for filename in centers:
            if all(centers[filename] == cells.points[i]):
                break # duplicate
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
    for i in range(n):
        if i in areas:
            a = areas[i]
            if a > expected - variation and a < expected + variation:
                (x, y) = cells.points[i]
                Gg.add_node(i, pos = (x, y))
    for i in range(n):
        if Gg.has_node(i):
            iC = cells.regions[cells.point_region[i]]
            assert -1 not in iC
            for j in range(i + 1, n):
                if Gg.has_node(j):                    
                    jC = cells.regions[cells.point_region[j]]
                    assert -1 not in jC                        
                    shared = set(iC) & set(jC)
                    k = len(shared)
                    if k == 2: # a shared border 
                        Gg.add_edge(i, j)
                    elif k > 2: # overlapping
                        Gg = nx.contracted_nodes(Gg, i, j)
    store(Gg, f'{dataset}.json')
    Gg = load(f'{dataset}.json')    
    pos = nx.get_node_attributes(Gg, 'pos')
#    norm = matplotlib.colors.Normalize(vmin = 0.0, vmax = 3.0)
# https://stackoverflow.com/questions/28752727/map-values-to-colors-in-matplotlib
#    mapper = plt.cm.ScalarMappable(norm = norm, cmap = plt.cm.RdYlGn)
# https://www.neonscience.org/calc-ndvi-tiles-py
#    nx.draw_networkx_nodes(Gg, pos, node_size = 12, node_shape='o', cmap = plt.cm.RdYlGn, linewidths = None, edgecolors = 'black')
    nx.draw_networkx_nodes(Gg, pos, node_size = 12, node_shape='o', linewidths = None, edgecolors = 'black') 
    nx.draw_networkx_edges(Gg, pos, width = 1, edge_color = 'black')
    plt.axis("off")
    plt.xlim(bbox[0])
    plt.ylim(bbox[1])
    plt.savefig(f'{dataset}_graph.png', bbox_inches = "tight", dpi = 150)
    plt.clf()
    for kind in shapes:
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
        if DRAW_CELL_CENTER:
            ax.scatter(cx, cy, marker = 'o', color = 'red', alpha = 0.3) # centers of the cells
        ax.set_xlim(bbox[0])
        ax.set_ylim(bbox[1])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        zones = dict()
        colors = defaultdict(list)
        for c in positions:
            (xc, yc, up) = positions[c]
            p = shape(xc, yc, kind, up)
            zones[c] = p
        print(f'{len(positions)} cells of kind {kind} computed for {dataset}')
        td = ax.transData # https://stackoverflow.com/questions/4285103/matplotlib-rotating-a-patch
        for filename in frameRGB:
            if INDIVIDUAL:
                target = f'{dataset}_{filename}_cells_{kind}.json'
                if not overwrite and path.exists(target):
                    print(f'A graph of type {kind} for {filename} of {dataset} already exists')
                    continue # no need to reprocess (delete this for a full wipe)
            print(f'Processing {filename} to store the result at {target}')
            Gf = nx.Graph()
            wm, hm, lonC, latC, a = frameRGB[filename]
            if DRAW_FRAME_CENTER:
                ax.scatter(lonC, latC, marker = 'o', color = 'black', alpha = 0.7) # centers of the frames
            xs, ys, rw, rh = rectangle(wm, hm, lonC, latC)
            (xf, yf) = centers[filename]
            ra = a - 90 # west is zero, north is no rotation, based on drone tests,
            rotation = matplotlib.transforms.Affine2D().rotate_deg_around(xf, yf, ra) + td 
            r = patches.Rectangle((xs, ys), rw, rh, edgecolor = 'blue', facecolor = 'none', alpha = 0.6, lw = 1)
            r.set_transform(rotation)
            im = Image.open(directory + 'enhanced/' + filename)
            # flip vertically as the y-axis grows the opposite way in images and on the plot
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
            iw, ih = im.size
            assert iw == resolution[True][0] and ih == resolution[True][1]
            xu = iw / rw
            yu = ih / rh
            # right size, right position, rotation
            it = matplotlib.transforms.Affine2D().scale(1 / xu, 1 / yu).translate(xs, ys).rotate_deg_around(xf, yf, ra) + td
            ax.imshow(im, transform = it, alpha = 0.5) 
            if DRAW_FRAME_BORDER:
                ax.add_patch(r)
            if SAVE_STAGES:
                ax.ticklabel_format(useOffset = False)
                # overwrite after each new frame has been added                
                plt.savefig(f'{dataset}_cells_{kind}.png',  bbox_inches = "tight", dpi = 300)
            if overview:
                print('Cell-level computations suppressed')
                continue
            skipped = 0
            content = 0
            print('Determining cell contents')
            for c in positions:
                (row, column) = c
                p = zones[c]
                coords = { True: pos2pixel(p, lonC, latC, wm, hm, a, True),
                           False: pos2pixel(p, lonC, latC, wm, hm, a, False) }
                if None not in coords.values(): # whole cell for all channels
                    cell = crop(directory + 'enhanced/' + filename, coords[True])
                    if cell is None: # it was too gray
                        print(f'Cell at {coords[True]} too gray', file = log)
                        if verbose:
                            print('-', end='')
                        skipped += 1
                    else: # there is content to process
                        print(f'Cell at ({row}, {column}) is significant', file = log)
                        content += 1
                        if verbose:
                            print('+', end='')                        
                        contents = { 'RGB': cell }
                        ntg = mask(contents['RGB']) # a mask to filter the not-too-gray positions
                        if SHOW_MASK:
                            contents['RGB'].show()
                            ntg.show()
                        for ch in channels:
                            contents[ch] = crop(directory + filename.replace('RGB.png', f'{ch}.TIF'), coords[False], mask = ntg)
                        v = combine(contents)
                        (xc, yc, up) = positions[c]
                        cellcolor = v[:3]
                        otherchannels = v[3:]
                        colors[c].append(v)
                        redRGB = cellcolor[0]
                        greenRGB = cellcolor[1]
                        NIR = otherchannels[0]
                        RED = otherchannels[1]
                        differences = [ greenRGB - redRGB, NIR - RED ]
                        if INDIVIDUAL:
                            Gf.add_node(f'{dataset}_{filename}_{kind}_{row}_{column}',
                                        pos = (xc, yc), color = cellcolor, value = otherchannels, state = differences)
                        G.add_node(f'{dataset}_{filename}_{kind}_{row}_{column}',
                                   pos = (xc, yc), color = cellcolor, value = otherchannels, state = differences)
                        values[c] = v
                        if (kind, row, column) in SAVE:
                            for ch in contents:
                                contents[ch].save(f'{dataset}_{filename[:-8]}_{ch}_{kind}_{row}_{column}.png')
            if verbose:
                print('Done processing')
            print(f'Encountered {content} significant cells of kind {kind}', file = log)
            if content < MINCELLS:                
                print(f'Too few significant cells: {content} / {MINCELLS}, skipping the frame')
                continue
            print(f'Encountered {content} significant cells of kind {kind}')            
            print(f'Skipped {skipped} incomplete cells of kind {kind}', file = log)
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
                if Gf.order() > 0:
                    store(Gf, target)
                    print(f'Graph exported for {filename} of {dataset}', file = log)
                    print(f'{filename} of {dataset} done')
                else:
                    print(f'{filename} of {dataset} produced no valid vertices', file = log)
            if DRAW_CELLS:                    
                for c in positions:
                    p = zones[c]
                    contents = colors[c]
                    if len(contents) > 0: # use the average
                        color = dec2rgb(average(contents)[:3]) # first three channels only
                        polygon = patches.Polygon(p, edgecolor = 'cyan', facecolor = color, alpha = 0.4, lw = 1)
                        ax.add_patch(polygon)
                    else: # not filled, none of the cells had contents
                        polygon = patches.Polygon(p, edgecolor = 'cyan', facecolor = 'none', alpha = 0.8, lw = 1)
                        ax.add_patch(polygon)                        
        ax.ticklabel_format(useOffset=False)
        plt.savefig(f'{dataset}_cells_{kind}.png',  bbox_inches = "tight", dpi = 300) # the overview visualization
        plt.clf()
    log.close()        

