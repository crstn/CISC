# coding: utf-8
#!/usr/bin/env python

import os, time, numpy as np, tif2num as tn, PopFunctions as pop, sys
from PIL import Image
from osgeo import gdal
import pync
import os.path

src = '/Volumes/Solid Guy/SSP5 2017-02-17/'

if len(sys.argv) != 2:
    print "Call this script with the year to reassemble, e.g.:"
    print "python ReassambleCountries.py 2020"
    sys.exit()

# wait until the files for USA (840) are there; this is the last large country,
# when its simulations for the current year are there, the simulation is complete.
while not os.path.exists(src+"840-"+sys.argv[1]+"-pop.npy"):
    print "Waiting for "+src+"840-"+sys.argv[1]+"-pop.npy"+" to finish..."
    time.sleep(10)
while not os.path.exists(src+"840-"+sys.argv[1]+"-urbanRural.npy"):
    print "Waiting for "+src+"840-"+sys.argv[1]+"-urbanRural.npy"+" to finish..."
    time.sleep(10)

print time.ctime()

print "Loading countries TIFF and converting to NumPy array"
countryBoundaries = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/Nations Raster/ne_10m_admin_0_countries_updated_nibbled.tiff')

print "Loading urban-rural TIFF and converting to NumPy array"
urbanRural2010 = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/GLUR Raster/GLUR_Pop20101.tiff')

print "Loading population 2010 TIFF and convert to NumPy array"
population2010 = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff')

print "Making copies of the last two for every year that we have projections for"
urbanRuralDict = {'2010': urbanRural2010}
popDict = {'2010': population2010}

countriesDir = os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'
os.chdir(src)

# some functions we'll need:

# Look up address in memmory, useful to make sure two arrays are
# actually different, and don't point to the same array
def wo(arr):
    pointer, read_only_flag = arr.__array_interface__['data']
    print pointer

# Returns a copy of a subblock from a 2D array.
# Size of the subblock is determined by
# startrow,endrow (included!), startcol,endcol (included!)
# Does NOT throw an error message if part of the block is
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

# Replaces cells in a with the values from the same cells in b. Cell indicies
# are specified in "where", which contains the indicies for the cells to be replaced.
# a and b need to be of equal size!
# Returns the updated array a. This works DIRECTLY on array a,
#, i.e., not returning a copy.
def replaceCellsInArray(a, b, where):
    a[where] = b [where]
    return a

# Replaces cells in a that are outside of all country boundaries with -1
# (which will then be set as NAN when saving as TIFF).
# Returns the updated array a. This works DIRECTLY on array a,
#, i.e., not returning a copy.
def outlineCountries(a):
    a[countryBoundaries == 0] = -1
    return a

years = []
years.append(sys.argv[1])

# make empty versions of the global raster for every year to fill in later:
for year in years:
    urbanRuralDict[year] = np.copy(urbanRural2010)
    popDict[year] = np.zeros(population2010.shape, dtype=np.int)

print "Reassembling global map for the following years:"
print years
# for year in years:
#     wo(popDict[year])

# iterate through folder with projections for each country:
for filename in os.listdir('.'):
    if filename.endswith(".npy"):
        if os.stat(filename).st_size == 0:
            print filename + " is empty - skipping"
        else:

            firstDash = filename.find('-')
            secondDash = filename.rfind('-')

            country = filename[:firstDash]
            year    = filename[firstDash+1:secondDash]
            maptype = filename[secondDash+1:]

            if year in years:
                print "Processing " + filename
                # replace the country in the global raster with the projected values
                f = countriesDir + country + ".0-boundary.npy"
                justCountryBoundary = np.load(f).astype(int)
                try:
                    projected = np.load(filename)

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
                except ValueError as e:
                    print filename + " contains an error - skipped."


for year in years:
    print "Saving TIFFs for " + str(year)

    ref = os.path.expanduser('~') + '/Dropbox/CISC Data/Nations Raster/ne_10m_admin_0_countries_updated_nibbled.tiff'

    pop.array_to_raster(outlineCountries(popDict[year]), src+'/pop-'+str(year)+'.tiff', ref)

    pop.array_to_raster(urbanRuralDict[year], src+'/urbanRural-'+str(year)+'.tiff', ref)

print time.ctime()
pync.Notifier.notify('Reassembling '+sys.argv[1]+' complete ¯\_(ツ)_/¯ ', title='Python')
