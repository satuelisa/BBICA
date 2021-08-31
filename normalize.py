from PIL import Image

def normalize(inputImage, outputImage):
    im = Image.open(inputImage)
    pixels = im.load() 
    w, h = im.size
    rLow = 255
    gLow = 255
    rHigh = 0
    gHigh = 0
    for x in range(w):
        for y in range(h):
            (r, g, b, a) = pixels[x, y]
            rLow = min(r, rLow)
            rHigh = max(r, rHigh)
            gLow = min(g, gLow)
            gHigh = max(g, gHigh)
    print(f'Normalizing R from [{rLow}, {rHigh}] to [0, 255]')
    print(f'Normalizing G from [{gLow}, {gHigh}] to [0, 255]')
    rSpan = rHigh - rLow
    gSpan = gHigh - gLow
    for x in range(w):
        for y in range(h):
            (r, g, b, a) = pixels[x, y]
            rN = int(round(255 * (r - rLow) / rSpan))
            gN = int(round(255 * (g - gLow) / gSpan))
            pixels[x, y] = (rN, gN, 0, 255)
    im.save(outputImage)

for image in ['ex1.png', 'ex2.png']:
    normalize(image, 'norm' + image)
