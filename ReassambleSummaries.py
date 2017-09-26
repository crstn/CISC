# coding: utf-8
#!/usr/bin/env python

from PIL import Image
from osgeo import gdal
import numpy as np
import PopFunctions as pop
import sys
import pync
import os
import os.path
import glob

"""Creates global GeoTIFFs from the summaries of a series of simulations.
   Iterates through a folder structure and for each folder where a the
   output of a simulation summary exists, puts a combined GeoTIFF for
   each year/maptype into the same folder. """

f = os.path.expanduser('~') + '/Desktop/chad/summaries'
years = range(2010, 2101, 10)


originalsDir = os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries'


# this dictionary will hold all the cell indexes (rows/cols) per country,
# so that we don't need to load them over and over again (they don't change over time)
rowcols = {}


"""Returns country ID, year, and map type based on a filname, e.g.
   disassembleFileName('528-2080-popmin.npy')
   will return
   '528', '2080', 'popmin' """
def disassembleFileName(filename):
    noext = filename.split(".")[0] #remove extension
    parts = noext.split("-")
    return parts[0], parts[1], parts[2]



"""Adds the rows/cols array for the given country to the rowcols dict
   if it is not there yet."""
def loadRowsCols(country):
    if country not in rowcols:
        rowcols[country] = {}
        rowcols[country]['rows'] = np.load(originalsDir+'/'+country+'.0-rows.npy')
        rowcols[country]['cols'] = np.load(originalsDir+'/'+country+'.0-cols.npy')



print 'Loading reference GeoTiff'

# load the reference tiff:
reffile = gdal.Open(os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff')
geotransform = reffile.GetGeoTransform()
rasterXSize = reffile.RasterXSize
rasterYSize = reffile.RasterYSize
projection = reffile.GetProjection()




for folder, dirs, files in os.walk(f):
    for y in years:
        # make a dictionary that will store our assemled world grids for each year:
        grids = {}

        for fil in files:
            if fil.endswith(".npy"):
                country, year, maptype = disassembleFileName(fil)

                # only proceed if the year in the file name matches the current year;
                # otherwise skip this file
                if str(y) == year:

                    # if there is no grid for this map type yet, make one first:
                    if maptype not in grids:
                        print "generating empty grid for " +str(y) + " " + maptype
                        grids[maptype] = np.zeros((16920, 43200)) # TODO change if we ever go to higher resolution

                    # load the row and column cells for that country, in case we don't have them yet:
                    loadRowsCols(country)
                    # replace the values in the global pop raster:
                    grids[maptype][rowcols[country]['rows'], rowcols[country]['cols']] = np.load(folder+'/'+fil)

        # print grids

        # done, save all grids:
        for maptype, grid in grids.iteritems():
            f = folder+'/'+str(y)+'-'+maptype+'.tiff'
            print "saving " + f
            if maptype == 'urbanization': # urbanization chances are saved in interval [0,1], so we need floats here
                pop.array_to_raster_noref(grid, f, geotransform, rasterXSize, rasterYSize, projection, datatype=gdal.GDT_Float32)
            else: # everything else will be rounded to integers
                pop.array_to_raster_noref(grid, f, geotransform, rasterXSize, rasterYSize, projection)
