# coding: utf-8
#!/usr/bin/env python

import os, time, numpy as np, tif2num as tn, PopFunctions as pop, sys
from PIL import Image
from osgeo import gdal
import pync

globcoverfile = os.path.expanduser('~') + '/Dropbox/CISC Data/Globcover2009_V2/GLOBCOVER_L4_200901_200912_V2.3_clipped_resampled.tif'
dest = os.path.expanduser('~') + '/Dropbox/CISC Data/GLUR Raster/GlobCover_UrbanAras.tiff'

print "Loading GlobCover TIFF and converting to NumPy array"
globcover = pop.openTIFFasNParray(globcoverfile)

print "Replacing urban values"
globcover[globcover == 190] = 2

print "Replacing water values"
globcover[globcover == 210] = 0

print "Making everything else rural"
globcover[globcover > 2] = 1

print "saving to " + dest
pop.array_to_raster(globcover, dest, globcoverfile)

print "Done ¯\_(ツ)_/¯"
