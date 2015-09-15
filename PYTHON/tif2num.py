from PIL import Image
from osgeo import gdal
import numpy as np
import os

os.chdir('/Users/carsten/Dropbox/Code/CISC/Data/NumpyLayers')

# urban/rural
im = Image.open('UrbanRural.tif')
imarray = np.array(im)
# convert to 8-bit int, large enough for our data -> save space on disk
np.save('UrbanRural', imarray.astype('int8'))

# national boundaries
# this one did not load using PIL (no idea why), so we're doing gdal here
src = gdal.Open('/Users/carsten/Dropbox/Code/CISC/Data/NumpyLayers/NationOutlines.tif', gdal.GA_Update)
band = src.GetRasterBand(1)
imarray = np.array(band.ReadAsArray())
# convert to 16-bit int, large enough for our data -> save space on disk
np.save('NationOutlines', imarray.astype('int16'))

# population numbers
im = Image.open('Population2000.tif')
imarray = np.array(im)
# we'll change this from float to int, no point in keeping "half people"
np.save('Population2000', imarray.astype('int'))
