# coding: utf-8
#!/usr/bin/env python

from PIL import Image
from osgeo import gdal
import numpy as np
import tif2num as tn
import PopFunctions as pop
import sys
import time
import pync
import os
import os.path
import glob

def printMsg():
    print "The script will find all .npy files for all scenrios / urban/rural combinations"
    print "in the respective folders and patch, starting from startyear to (incl.) endyear "
    print "Call e.g. via: python PatchCountries.py 2010 2050 "
    print "them into the corresponding population and urban/rural TIFFs in the same folder."
    print "This was made for fixing Individual countries; for a whole reassably,"
    print "run ReassambleCountries.py."
    sys.exit()

if len(sys.argv) != 3:
    printMsg()

try:
    startyear = int(sys.argv[1])
    endyear = int(sys.argv[2])
except Exception as e:
    printMsg()


countriesDir = os.path.expanduser('~') + '/Dropbox/CISCdata/IndividualCountries/'

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


print "Loading countries TIFF and converting to NumPy array"
countriesFile = os.path.expanduser('~') + '/Dropbox/CISCdata/NationsRaster/ne_10m_admin_0_countries_updated_nibbled.tiff'
countryBoundaries = pop.openTIFFasNParray(countriesFile)


# use this as the reference tiff later:

reffile = gdal.Open(countriesFile)
geotransform = reffile.GetGeoTransform()
rasterXSize = reffile.RasterXSize
rasterYSize = reffile.RasterYSize
projection = reffile.GetProjection()



root = os.path.expanduser('~') + '/Dropbox/CISCdata/IndividualCountries/Projections/'

for m in ["GRUMP", "GlobCover"]:
    for ssp in ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]:
        src = root+m+'/'+ssp+'/'
        for year in range (startyear, endyear+1, 10):
            for kind in ["pop", "urbanRural"]:
                # find all .npy files for that year / kind (pop and urban/rural):
                files = glob.glob(src+'*-'+str(year)+'-'+kind+'.npy')
                if len(files) > 0:
                    # load the original tiff to paste into:
                    print "Loading "+kind+" TIFF for "+str(year)+" and converting to NumPy array"
                    globalRaster = pop.openTIFFasNParray(src+kind+"-"+str(year)+".tiff")

                    for filename in files:
                        print "Processing " + filename

                        # get the country ID from the file name:
                        lastslash = filename.rfind('/')
                        f = filename[lastslash+1:]
                        country = f[:f.find('-')]


                        # replace the country in the global raster with the .npy values
                        cf = countriesDir + country + ".0-boundary.npy"

                        justCountryBoundary = np.load(cf).astype(int)
                        try:
                            updated = np.load(filename)

                            x, y = np.where(countryBoundaries==int(country))

                            # cut this block from the global raster:
                            subblock = copySubBlock(globalRaster, np.min(x), np.max(x), np.min(y), np.max(y))

                            # replace the population numbers in this subblock, but ONLY within the borders of the country:
                            where = np.where(justCountryBoundary == int(country))
                            replaceCellsInArray(subblock, updated, where)

                            #  put the subblock back into its place in the original raster:
                            replaceBlockInArray(globalRaster, subblock, np.min(x), np.min(y))

                        except ValueError as e:
                            print filename + " contains an error - skipped."

                    print "Saving TIFF"

                    pop.array_to_raster_noref(globalRaster, root+m+'/'+ssp+'/'+kind+'-'+str(year)+'-updated.tiff', geotransform, rasterXSize, rasterYSize, projection)

print time.ctime()
pync.Notifier.notify('Patching complete ¯\_(ツ)_/¯ ', title='Python')
