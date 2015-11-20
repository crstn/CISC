from PIL import Image
from osgeo import gdal
import numpy as np
import os

os.chdir(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia')

files = ["GLUR/GLUR_asia.tif", "Nations/GLBNDS_asia1.tif", "Population 2000/pop_2000_asia.tif", "Population 2010/pop_2010_asia.tif"]

for f in files:
    src = gdal.Open(f, gdal.GA_Update)
    band = src.GetRasterBand(1)
    imarray = np.array(band.ReadAsArray())
    # Make sure the numpy array all have the same dimensions:
    print f
    print imarray.shape

    np.save(f, imarray)
