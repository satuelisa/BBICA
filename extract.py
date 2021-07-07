# based on https://stackoverflow.com/questions/22588074/polygon-crop-clip-using-python-pil
import numpy
from PIL import Image, ImageDraw

def crop(filename, coords):
    im = Image.open(filename).convert('RGBA')
    imArray = numpy.asarray(im)
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(coords, outline = 1, fill = 1)
    mask = numpy.array(maskIm)
    newImArray = numpy.empty(imArray.shape, dtype = 'uint8')
    newImArray[:,:,:3] = imArray[:,:,:3]
    newImArray[:,:,3] = 255 * mask
    return Image.fromarray(newImArray, 'RGBA')

if __name__ == '__main__':
    dataset = 'jul25b'
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
    test = 'IMG_170827_150122_0087_RGB.JPG'
    cell = [(20, 20), (180, 40), (220, 300), (70, 400)]
    extracted = crop(directory + test, cell)
    extracted.save('test.png')

 
