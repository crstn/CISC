from PIL import Image
import numpy as np

@profile
def read_tiff():
    im = Image.open('/Users/carsten/Downloads/TIFFS/glurextents.tif')
    # im.show()
    imarray = np.array(im)

    print imarray.shape
    print im.size
    print np.min(imarray)
    print np.max(imarray)

if __name__ == '__main__':
    read_tiff()
