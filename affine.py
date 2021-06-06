import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms

filename = '/Volumes/dropbox/Dropbox/Research/Topics/Arboles/CA/RAW/jul25b/images/enhanced/IMG_170725_192210_0156_RGB.png'
#img = plt.imread(filename)
img = Image.open(filename)
w, h = img.size
fig = plt.figure()
ax = fig.add_subplot(111)
x0, x1 = -1, 11
y0, y1 = -1, 11
ax.set_xlim((x0, x1))
ax.set_ylim((y0, y1))
# desired angle
angle = 60
# target location as the lower left corner
xp = 6
yp = 4
# target size
xt = 10
yt = (xt / w) * h
# unit conversion (pixels per target unit)
units = 1000
tr = transforms.Affine2D().scale(1 / units).rotate_deg(angle).translate(xp, yp)
im = np.asarray(img)
plt.plot(xp, yp, 'ro') # target location
ax.imshow(im, transform = tr + ax.transData)
plt.show()
