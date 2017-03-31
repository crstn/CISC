# coding: utf-8
#!/usr/bin/env python

from osgeo import gdal
import numpy as np
import csv             #for reading csv
import os, sys
import pandas as pd

models = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]
years =  range(2020, 2101, 10)


def loadTIFF(f):
    # load population layer
    src = gdal.Open(f, gdal.GA_Update)
    band = src.GetRasterBand(1)
    nparray = np.array(band.ReadAsArray())
    return nparray.ravel()



cities = loadTIFF(os.path.expanduser('~') + '/Dropbox/CISC Data/SDEI-Global-UHI/sdei-global-uhi-2013.tiff')

# we can do this once here because the cities layer doesn't change
print "Checking cells outside of cities"
outthere = cities == 0



for m in models:

    print
    print ' ----------------------------------'
    print
    print m

    datadir = os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/Global/2017-03-30/'+m

    for y in years:

        print

        # load population layer
        pop = loadTIFF(datadir+'/pop-'+str(y)+'.tiff')
        pop[pop < 0] = 0  #set all population values below 0 as 0

        # load urban/rural layer
        ur = loadTIFF(datadir+'/urbanRural-'+str(y)+'.tiff')

        urban = ur == 3

        # globaltotalraster = np.nansum(pop)
        # gtr = "{:,}".format(globaltotalraster)
        # print 'Total raster: ' + gtr
        #
        # urbantotalraster = np.nansum(pop[np.where(ur == 3)])
        # utr = "{:,}".format(urbantotalraster)
        # print 'Urban raster: ' + utr

        numUrb = len(np.where(np.logical_and(urban, outthere))[0])
        numUrb = "{:,}".format(numUrb)

        popUrb = np.nansum(pop[np.logical_and(urban, outthere)])
        popUrb = "{:,}".format(popUrb)

        print str(y) + " # urban cells outside of cities:    " + numUrb
        print str(y) + " urban population outside of cities: " + popUrb
