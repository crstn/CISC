from PIL import Image
import numpy as np

imarray = np.load("/Users/carsten/Downloads/Numpy/gluntlbnds-clipped-int.npy")

output = Image.fromarray(imarray)
output.save('/Users/carsten/Downloads/Numpy/gluntlbnds-clipped-int.tif')
