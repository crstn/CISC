# coding: utf-8
#!/usr/bin/env python

import os, time, numpy as np, tif2num as tn, PopFunctions as pop, sys
from PIL import Image
from osgeo import gdal
import pync

grumpfile = os.path.expanduser('~') + '/Dropbox/CISC Data/GLUR Raster/GLUR_Pop20101.tiff'
globcoverfile = os.path.expanduser('~') + '/Dropbox/CISC Data/Globcover2009_V2/GLOBCOVER_L4_200901_200912_V2.3_clipped_resampled.tif'
dest = os.path.expanduser('~') + '/Dropbox/CISC Data/GLUR Raster/GLUR_Pop20101_suburban.tiff'

print "Loading GRUMP urban/rural TIFF and converting to NumPy array"
grump = pop.openTIFFasNParray(grumpfile)

print grump.size

print "Loading GlobCover TIFF and converting to NumPy array"
globcover = pop.openTIFFasNParray(globcoverfile)

print grump.size

# replace all cells in grump with value 3 where globcover value is 190 (urban)
grump[globcover == 190] = 3

pop.array_to_raster(grump, dest, grumpfile)
