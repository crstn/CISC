from PIL import Image
import numpy as np
import os

#@profile
def read_tiff():
    im = Image.open('/Users/carsten/Downloads/gl_grumpv1_pcount_00_ascii_30/glup00ag-clipped.tif')
    # im.show()
    imarray = np.array(im)

    print imarray.shape
    print im.size
    print np.min(imarray)
    print np.max(imarray)

    os.chdir('/Users/carsten/Downloads/Numpy')
    np.save('glup00ag-clipped', imarray)

if __name__ == '__main__':
    read_tiff()
