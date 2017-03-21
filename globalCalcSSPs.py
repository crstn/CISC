# coding: utf-8
#!/usr/bin/env python

from osgeo import gdal
import numpy as np
import csv             #for reading csv
import os, sys

models = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]
years = range(2020,2101,10)

filename = os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/Global/SSPstats.csv'

# Make sure we don't overwrite an existing citiesPop.csv, this one takes
# hours to generate!
if os.path.exists(filename):
    print filename+" already exists. If you want to recalculate it, delete it first."
    sys.exit()


with open(filename, 'a') as outputFile:
    outputFile.write(';2020;2030;2040;2050;2060;2070;2080;2090;2100\n')  #write the header row

    for m in models:

        outputFile.write(m+';')

        datadir = os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/Global/'+m

        for y in years:
            src = gdal.Open(datadir+'/pop-'+str(y)+'.tiff', gdal.GA_Update)
            band = src.GetRasterBand(1)
            pop = np.array(band.ReadAsArray())
            pop[pop < 0] = 0  #set all population values below 0 as 0

            globaltotal = np.nansum(pop)
            g = "{:,}".format(globaltotal)

            print m + " in " + str(y) +": " + g

            if y == 2100:
                outputFile.write(g+'\n')
            else:
                outputFile.write(g+';')
