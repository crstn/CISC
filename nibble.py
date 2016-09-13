#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script checks the countries raster for any cells that do have population,
# but do not belong to a country, and then assigns the majority country of the
# n x n neighborhood to them

import os, numpy as np, tif2num as tn
from PIL import Image
from osgeo import gdal
import PopFunctions as pop

# input:
countriesTIFF = os.path.expanduser('~') + '/Dropbox/CISC Data/Nations Raster/Natural Earth Data/ne_10m_admin_0_countries_updated.tiff'
popTIFF = os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2000 Raster/Pop_2000_clipped.tiff'

# output will be written to:
nibbledTIFF = os.path.expanduser('~') + '/Dropbox/CISC Data/Nations Raster/ne_10m_admin_0_countries_updated_nibbled.tiff'

print "Loading countries TIFF and converting to NumPy array"
countryBoundaries = pop.openTIFFasNParray(countriesTIFF)

shape = countryBoundaries.shape
countryBoundaries = countryBoundaries.ravel()

print "Loading population 2000 TIFF and converting to NumPy array"
population = pop.openTIFFasNParray(popTIFF).ravel()

print "Replacing NAN in country raster for cells that do have population"
a = countryBoundaries <= 0
b = population > 0


nationless = np.all((a,b), axis=0)
iteration = 0

# Returns the item that appears most often in a list. If two or more occur equally often (e.g., [1,1,2,2,3]), the first one will be returned (1 in our example).
# Doesn't check if the input list is empty (or if it even is a list, for that matter...)!
def getMajority(x):
    unique, counts = np.unique(x, return_counts=True)
    return unique[counts==np.max(counts)][0]

print "Nationless cells: "+str(countryBoundaries[nationless].size)
print "This will take a while, took about an hour on my laptop"
for cell in range(nationless.size):
    if nationless[cell]:
        nsize = 3 # start with 3x3 neighborhood

        neighborvalues = []
        while(len(neighborvalues) == 0):
            neighborIndexes = pop.getNeighbours(cell, shape, nsize)

            for n in neighborIndexes:
                # ignore cells that don't belong to a country either:
                if countryBoundaries[n] > 0:
                    neighborvalues.append(countryBoundaries[n])

            # use the majority for that cell to fill in
            if(len(neighborvalues) > 0):
                countryBoundaries[cell] = getMajority(neighborvalues)
            else:
                # if all cells in the current neighborhood don't belong to a
                # country either, make the search neighborhood bigger
                nsize = nsize + 2

pop.array_to_raster(countryBoundaries.reshape(shape), nibbledTIFF, countriesTIFF)

print "That's all folks ¯\_(ツ)_/¯"
