# coding: utf-8
#!/usr/bin/env python

from osgeo import gdal
import numpy as np
import csv             #for reading csv
import os, sys
import pandas as pd

models = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]
years =  [2030, 2050, 2070, 2100]

def getNumberFromCSV(model, year, rastertype):
    csvdir = os.path.expanduser('~') + '/Dropbox/CISCdata/SSPs/'
    data = pd.read_csv(csvdir+rastertype+'-'+model+'.csv')
    sums = data.sum()
    return sums[str(year)]


for m in models:

    print
    print ' ----------------------------------'
    print
    print m
    print

    datadir = os.path.expanduser('~') + '/Dropbox/CISCdata/IndividualCountries/Projections/Global/2017-03-30/'+m

    for y in years:

        print ' --- '
        print y
        print

        # load population layer
        src = gdal.Open(datadir+'/pop-'+str(y)+'.tiff', gdal.GA_Update)
        band = src.GetRasterBand(1)
        pop = np.array(band.ReadAsArray())
        pop[pop < 0] = 0  #set all population values below 0 as 0

        # load urban/rural layer
        src = gdal.Open(datadir+'/urbanRural-'+str(y)+'.tiff', gdal.GA_Update)
        band = src.GetRasterBand(1)
        ur = np.array(band.ReadAsArray())


        globaltotalraster = np.nansum(pop)
        gtr = "{:,}".format(globaltotalraster)
        print 'Total raster: ' + gtr

        globaltotalcsv = getNumberFromCSV(m, y, 'pop')
        gtc = "{:,}".format(globaltotalcsv)
        print 'Total csv: ' + gtc

        totaldiff = globaltotalraster - globaltotalcsv
        td = "{:,}".format(totaldiff)
        print 'Total difference: ' + td

        urbantotalraster = np.nansum(pop[np.where(ur == 3)])
        utr = "{:,}".format(urbantotalraster)
        print 'Urban raster: ' + utr

        urbantotalcsv = getNumberFromCSV(m, y, 'urbpop')
        utc = "{:,}".format(urbantotalcsv)
        print 'Urban csv: ' + utc

        urbandiff = urbantotalraster - urbantotalcsv
        udiff = "{:,}".format(urbandiff)
        print 'Urban difference: ' + udiff
