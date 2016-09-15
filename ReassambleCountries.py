# coding: utf-8
#!/usr/bin/env python

import os, numpy as np, tif2num as tn
from PIL import Image
from osgeo import gdal
import PopFunctions as pop

print "Loading countries TIFF and converting to NumPy array"
countryBoundaries = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/Nations Raster/ne_10m_admin_0_countries_updated_nibbled.tiff')

print "Loading urban-rural TIFF and converting to NumPy array"
urbanRural2010 = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/GLUR Raster/GLUR_Pop20101.tiff')

print "Loading population 2010 TIFF and convert to NumPy array"
population2010 = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff')

print "Making copies of the last two for every year that we have projections for"
urbanRuralDict = {'2010': urbanRural2010}
popDict = {'2010': population2010}

os.chdir(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries')

years = []

# some functions we'll need:

# Look up address in memmory, useful to make sure two arrays are
# actually different, and don't point to the same array
def wo(arr):
    pointer, read_only_flag = arr.__array_interface__['data']
    print pointer

# Returns a copy of a subblock from a 2D array.
# Size of the subblock is determined by
# startrow,endrow (included!), startcol,endcol (included!)
# Does NOT throw an error message of part of the block is
# outside of a
def copySubBlock(a,startrow,endrow,startcol,endcol):
    return np.copy(a[startrow:endrow+1,startcol:endcol+1])

# Replaces a rectangular block at positions x, y
# in array A with the content of array B.
# Does NOT check if going out of bounds of the array!
# Returns the updated array A. This works DIRECTLY on array a,
#, i.e., not returning a copy.
def replaceBlockInArray(a, b, startrow, startcol):
    x, y = b.shape
    endrow = x+startrow
    endcol = y+startcol
    a[startrow:endrow, startcol:endcol] = b
    return a

# Replaces cells in A with the values from the same cells in B. Cell indicies
# are specified in "where", which contains the indicies for the cells to be replaced
# a and b need to be of equal size!
# Returns the updated array A. This works DIRECTLY on array a,
#, i.e., not returning a copy.
def replaceCellsInArray(a, b, where):
    a[where] = b [where]
    return a

for filename in os.listdir('./Projections'):
    if filename.endswith(".npy"):
        # the year is between the dashes
        start = filename.find('-')+1
        end = filename.rfind('-')
        if filename[start:end] not in years:
            years.append(filename[start:end])



# make copies of the global raster for every year:
for year in years:
    urbanRuralDict[year] = np.copy(urbanRural2010)
    popDict[year] = np.copy(population2010)


# for year in years:
#     wo(popDict[year])


# iterate through folder with projections for each country:
for filename in os.listdir('./Projections'):
    if filename.endswith(".npy"):
        print "Processing " + filename

        firstDash = filename.find('-')
        secondDash = filename.rfind('-')

        country = filename[:firstDash]
        year    = filename[firstDash+1:secondDash]
        maptype = filename[secondDash+1:]

        # replace the country in the global raster with the projected values
        f = country+".0-boundary.npy"
        justCountryBoundary = np.load(f).astype(int)
        projected = np.load('./Projections/'+filename)

        x, y = np.where(countryBoundaries==int(country))

        if maptype == "pop.npy":
            # cut this block from the global population projections raster:
            subblock = copySubBlock(popDict[year], np.min(x), np.max(x), np.min(y), np.max(y))
        elif maptype == "urbanRural.npy":
            subblock = copySubBlock(urbanRuralDict[year], np.min(x), np.max(x), np.min(y), np.max(y))

        # replace the population numbers in this subblock, but ONLY within the borders of the country:
        where = np.where(justCountryBoundary == int(country))
        replaceCellsInArray(subblock, projected, where)

        #  put the subblock back into its place in the original raster:
        if maptype == "pop.npy":
            replaceBlockInArray(popDict[year], subblock, np.min(x), np.min(y))
        elif maptype == "urbanRural.npy":
            replaceBlockInArray(urbanRuralDict[year], subblock, np.min(x), np.min(y))

for year in years:
    print "Saving TIFFs for " + str(year)

    pop.array_to_raster(popDict[year], os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/Global/pop-'+str(year)+'.tiff', os.path.expanduser('~') + '/Dropbox/CISC Data/Nations Raster/ne_10m_admin_0_countries_updated_nibbled.tiff')

    pop.array_to_raster(urbanRuralDict[year], os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/Global/urbanRural-'+str(year)+'.tiff', os.path.expanduser('~') + '/Dropbox/CISC Data/Nations Raster/ne_10m_admin_0_countries_updated_nibbled.tiff')
