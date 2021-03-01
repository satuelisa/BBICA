from PIL import Image
from sys import argv

filename = argv[1]
img = Image.open(filename)
pix = img.load()
(w, h) = img.size
threshold = 30
for x in range(w):
    for y in range(h):
        (r, g, b, a) = pix[x, y]
        if a == 255:
            if max(r, g, b) - min(r, g, b) < threshold:
                pix[x, y] = (0, 0, 0, 0)
img.save(f'{filename[:-4]}_filtered.png')
