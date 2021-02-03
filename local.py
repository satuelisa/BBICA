datasets = ['aug27b', 'jul25b']
bbox = ((-99.90024722222223, -99.89772500000001),
        (24.20827222222222, 24.21)) # lon, lat (bb.py from BarkBeetle)
zone = ((-99.9000, 24.2089),
        (-99.8993, 24.20995),
        (-99.8986, 24.2094),
        (-99.8993, 24.20835)) # area of interest (four corners in lon, lat)

def average(d1, d2):
    means = []
    assert len(d1) == len(d2)
    for (v1, v2) in zip(d1, d2):
        means.append((v1 + v2) / 2)
    return means
