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

# throw away the numpy files after reassembling the GeoTIFF?
deleteNPY = False

# if(len(sys.argv) < 3):
    # print """Call this script with the following arguments:
    #     1. folder for a series of summaries
    #     2. SSP
    #     3. Urban/rural model
    #     4. the IDs of the countries to reassemble
    #
    # python ReassembleCountries.py /Users/Donald/simulations/summaries SSP3 GRUMP 123 156 276"""
    # sys.exit()

#check that folder exists:
if not os.path.exists(sys.argv[1]):
    print sys.argv[1] + " doesn't exist."
    sys.exit()


root = os.path.expanduser('~') + '/Dropbox/CISC Data/'
originalsDir = root + 'IndividualCountries'
simulationsDir = sys.argv[1]
outputDir = sys.argv[1]
# ssp = sys.argv[2]
# urbanRuralVersion = sys.argv[3]
# countries = sys.argv[4:]
countries = sys.argv[2:]


ssps = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]
urbanRuralVersions = ["GlobCover", "GRUMP"]


if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# keep track of the years we have data for
years = range(2010, 2101, 10)


# this dictionary will hold all the cell indexes (rows/cols) per country,
# so that we don't need to load them over and over again (they don't change over time)
rowcols = {}

print 'Loading reference GeoTiff'

# load the reference tiff:
reffile = gdal.Open(os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff')
geotransform = reffile.GetGeoTransform()
rasterXSize = reffile.RasterXSize
rasterYSize = reffile.RasterYSize
projection = reffile.GetProjection()


def disassembleFileName(filename):
    firstDash = filename.find('-')
    secondDash = filename.rfind('-')
    country = filename[:firstDash]
    year    = int(filename[firstDash+1:secondDash])
    maptype = filename[secondDash+1:-4]

    return country, year, maptype


# adds the rows/cols array for the given country to the rowcols dict
# if it is not there yet
def loadRowsCols(country):
    if country not in rowcols:
        rowcols[country] = {}
        rowcols[country]['rows'] = np.load(originalsDir+'/'+country+'.0-rows.npy')
        rowcols[country]['cols'] = np.load(originalsDir+'/'+country+'.0-cols.npy')


# checks if all files for the given pattern are completed
# by comparing it to the countries. Returns True if the simulation is complete,
# False otherwise.
def simulationComplete(pattern):
    donefiles = glob.glob(pattern)
    # extract the country IDs:
    for i in range(len(donefiles)):
        fn = donefiles[i].split('/')[-1]
        # just the country ID:
        donefiles[i] = disassembleFileName(fn)[0]

    # now check if donefiles is a subset of the list of countries we need to process:
    if set(countries).issubset(set(donefiles)):
        return True
    else:
        return False



# load the global population and urban rural tiffs
# it will hold urbanization prospects [0..1] later, so we need to cast it to float:
# print "Loading urban-rural TIFF and converting to NumPy array"
# urbanRural = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/GLUR Raster/'+urbanRuralVersion+'_UrbanRural.tiff').astype(float)







# make sure we have the rows/cols for all countries simulated in this year:
for urbanRuralVersion in urbanRuralVersions:

    for ssp in  ssps:

        print "Loading population 2010 TIFF and convert to NumPy array"
        population = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff')

        print "Loading urban/rural 2010 TIFF and convert to NumPy array"
        urbanRural = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/GLUR Raster/'+urbanRuralVersion+'_UrbanRural.tiff')


        #  find all years we have simulations for:
        d = simulationsDir + '/' + urbanRuralVersion + "/" + ssp


        for y in years:

            # before we start, make sure that all the simulations for this year are actually done
            # by checking if all the output files are there. If not, wait:
            # while not simulationComplete(d+'/*-'+str(y)+'-popmean.npy'):
            while not simulationComplete(d+'/*-'+str(y)+'-pop.npy'):
                print "waiting for population simulation to complete..."
                time.sleep(1)


            # here we go: find all pop simulation files for this year:
            # for filename in glob.glob(d+'/*-'+str(y)+'-popmean.npy'):
            for filename in glob.glob(d+'/*-'+str(y)+'-pop.npy'):
                f = filename.split('/')[-1]

                if os.stat(d+'/'+f).st_size > 0: # skip empty files
                    country, year, maptype = disassembleFileName(f)
                    # load their extents, only required once
                    loadRowsCols(country)
                    # replace the values in the global pop raster:
                    population[rowcols[country]['rows'], rowcols[country]['cols']] = np.load(d+'/'+f)

                if deleteNPY:
                    os.remove(filename)

            # save the global tiff to the output folder:
            pop.array_to_raster_noref(population, outputDir + '/' + urbanRuralVersion + '/' + ssp + '/popmean-'+str(year)+'.tiff', geotransform, rasterXSize, rasterYSize, projection)

            # repeat for urban/rural:
            # for filename in glob.glob(d+'/*-'+str(y)+'-urbanization.npy'):
            for filename in glob.glob(d+'/*-'+str(y)+'-urbanRural.npy'):
                f = filename.split('/')[-1]

                if os.stat(d+'/'+f).st_size > 0: # skip empty files
                    country, year, maptype = disassembleFileName(f)
                    # replace the values in the global pop raster:
                    urbanRural[rowcols[country]['rows'], rowcols[country]['cols']] = np.load(d+'/'+f)

                if deleteNPY:
                    os.remove(filename)

            # save the global tiff to the output folder:
            # pop.array_to_raster_noref(urbanRural, outputDir + '/' + urbanRuralVersion + '/' + ssp + '/urbanization-'+str(year)+'.tiff', geotransform, rasterXSize, rasterYSize, projection, datatype=gdal.GDT_Float32)
            pop.array_to_raster_noref(urbanRural, outputDir + '/' + urbanRuralVersion + '/' + ssp + '/urbanization-'+str(year)+'.tiff', geotransform, rasterXSize, rasterYSize, projection)
