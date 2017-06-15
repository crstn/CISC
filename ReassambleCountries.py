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

ssps = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]
urbanRuralVersions = ["GlobCover", "GRUMP"]

root = os.path.expanduser('~') + '/Dropbox/CISC Data/'
originalsDir = root + 'IndividualCountries'
simulationsDir = '/Volumes/Solid Guy/SSPs 2017-06-16'
outputDir = simulationsDir + '/Global'

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




for urbanRuralVersion in urbanRuralVersions:

    print 'Starting ' + urbanRuralVersion

    if os.path.isdir(simulationsDir + '/' + urbanRuralVersion):

        # create folder for the output:
        if not os.path.exists(outputDir + '/' + urbanRuralVersion):
            os.makedirs(outputDir + '/' + urbanRuralVersion)

        for ssp in ssps:

            print 'Starting ' + ssp

            if os.path.isdir(simulationsDir + '/' + urbanRuralVersion + "/" + ssp):

                # create folder for the output:
                if not os.path.exists(outputDir + '/' + urbanRuralVersion + "/" + ssp):
                    os.makedirs(outputDir + '/' + urbanRuralVersion + "/" + ssp)

                # load the global population and urban rural tiffs
                print "Loading urban-rural TIFF and converting to NumPy array"
                urbanRural = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/GLUR Raster/'+urbanRuralVersion+'_UrbanRural.tiff')

                print "Loading population 2010 TIFF and convert to NumPy array"
                population = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff')


                #  find all years we have simulations for:
                d = simulationsDir + '/' + urbanRuralVersion + "/" + ssp

                # make sure we have the rows/cols for all countries simulated in this year:
                for y in years:
                    # here we go: find all pop simulation files for this year:
                    for filename in glob.glob(d+'/*-'+str(y)+'-pop.npy'):
                        filename = filename.split('/')[-1]

                        if os.stat(d+'/'+filename).st_size > 0: # skip empty files
                            country, year, maptype = disassembleFileName(filename)
                            # load their extents, only required once
                            loadRowsCols(country)
                            # replace the values in the global pop raster:
                            population[rowcols[country]['rows'], rowcols[country]['cols']] = np.load(d+'/'+filename)

                    # save the global tiff to the output folder:
                    pop.array_to_raster_noref(population, outputDir + '/' + urbanRuralVersion + '/' + ssp + '/pop-'+str(year)+'.tiff', geotransform, rasterXSize, rasterYSize, projection)

                    # repeat for urban/rural:
                    for filename in glob.glob(d+'/*-'+str(y)+'-urbanRural.npy'):
                        filename = filename.split('/')[-1]

                        if os.stat(d+'/'+filename).st_size > 0: # skip empty files
                            country, year, maptype = disassembleFileName(filename)
                            # replace the values in the global pop raster:
                            urbanRural[rowcols[country]['rows'], rowcols[country]['cols']] = np.load(d+'/'+filename)

                    # save the global tiff to the output folder:
                    pop.array_to_raster_noref(urbanRural, outputDir + '/' + urbanRuralVersion + '/' + ssp + '/urbanRural-'+str(year)+'.tiff', geotransform, rasterXSize, rasterYSize, projection)


            else:
                print "No simulations found for " + ssp + " found in " + urbanRuralVersion + ", skipping."
    else:
        print "No simulations found for " + urbanRuralVersion + ", skipping."
