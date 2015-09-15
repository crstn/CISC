from PIL import Image
import numpy as np

# imarray = np.load("/Users/carsten/Dropbox/Code/CISC/Data/NumpyLayers/gluntlbnds-clipped-int.npy")
# output = Image.fromarray(imarray)
# output.save('/Users/carsten/Dropbox/Code/CISC/Data/NumpyLayers/gluntlbnds-clipped-int.tif')

imarray = np.load("/Users/carsten/Dropbox/Code/CISC/Data/NumpyLayers/glup00ag-clipped.npy")
output = Image.fromarray(imarray)
output.save('/Users/carsten/Dropbox/Code/CISC/Data/NumpyLayers/glup00ag-clipped.tif')

# imarray = np.load("/Users/carsten/Dropbox/Code/CISC/Data/NumpyLayers/glurextents-int.npy")
# output = Image.fromarray(imarray)
# output.save('/Users/carsten/Dropbox/Code/CISC/Data/NumpyLayers/glurextents-int.tif')
