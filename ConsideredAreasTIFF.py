# coding: utf-8
#!/usr/bin/env python

import os
import time
import sys
import pync
import os.path
import glob
import numpy as np
import tif2num as tn
import PopFunctions as pop
from PIL import Image
from osgeo import gdal

# makes a geotiff showing all pixels that have been considered in the simulation

inputDir = os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'
outputDir = os.path.expanduser('~') + '/Desktop/Test/'

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# load the reference tiff:
reffile = gdal.Open(os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff')
geotransform = reffile.GetGeoTransform()
rasterXSize = reffile.RasterXSize
rasterYSize = reffile.RasterYSize
projection = reffile.GetProjection()
#
# load the same file as a numpy array:
tiff = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff')
tiff.fill(0)

# find all row/col files:
rowfiles = glob.glob(inputDir+'*-rows.npy')
colfiles = glob.glob(inputDir+'*-cols.npy')

for r, c in zip(rowfiles, colfiles):
    tiff[np.load(r), np.load(c)] = 1

pop.array_to_raster_noref(tiff, outputDir + '/consideredAreas.tiff', geotransform, rasterXSize, rasterYSize, projection)
