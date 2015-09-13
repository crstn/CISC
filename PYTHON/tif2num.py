from PIL import Image
import numpy as np

@profile
def read_tiff():
    im = Image.open('/Users/carsten/Dropbox/Code/CISC/Data/glurpop2000/glup00ag.tif')
    # im.show()
    imarray = np.array(im)

    print imarray.shape
    print im.size
    print np.min(imarray)
    print np.max(imarray)

if __name__ == '__main__':
    read_tiff()
