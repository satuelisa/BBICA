datasets = ['jul25b', 'aug27b']
# datasets = datasets[::-1] # invert order to do august first
channels = ['NIR', 'RED', 'REG', 'GRE'] # the multi- spectral TIF images 
bb = [[-99.90024722222223, -99.89772500000001],
        [24.20827222222222, 24.21]] # lon, lat (bb.py from BarkBeetle)
margin = 0.001
for x in range(2):
    for y in range(2):
        c = -1 if y == 0 else 1
        bb[x][y] = bb[x][y] + c * margin
bbox = ((bb[0][0], bb[0][1]), (bb[1][0], bb[1][1]))
    
zone = ((-99.9000, 24.2089),
        (-99.8993, 24.20995),
        (-99.8986, 24.2094),
        (-99.8993, 24.20835)) # area of interest (four corners in lon, lat)


