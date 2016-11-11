# coding: utf-8
#!/usr/bin/env python

# This one compares the population numbers from our individual countries .npy files
# to the numbers for the same country in the global TIFF

import os
import csv
import time
import numpy as np
import tif2num as tn
import PopFunctions as pop
from PIL import Image
from osgeo import gdal

countriesDir = os.path.expanduser(
    '~') + '/Dropbox/CISC Data/IndividualCountries/'
os.chdir(countriesDir)

WTP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC Data/DESA/WPP2015_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.csv')), "Country code")


def getPopNoIndividual(country, year):
    # load the country boundary
    f = countriesDir + country + ".0-boundary.npy"
    boundary = np.load(f).astype(int)

    # now load the population numbers for the country / year
    f = countriesDir + "/Projections/" + country + "-" + year + "-pop.npy"
    if os.stat(f).st_size > 0:
        pop = np.load(f)
        return np.sum(pop[boundary == int(country)])
    else:
        return 0


years = []
countries = []

for filename in os.listdir('./Projections'):
    if filename.endswith(".npy"):
        # find the year
        start = filename.find('-') + 1
        end = filename.rfind('-')
        if filename[start:end] not in years:
            years.append(filename[start:end])
        # find the country
        if filename[:start - 1] not in countries:
            countries.append(filename[:start - 1])

world = []

print "Loading tiffs..."

# load the global tiffs:
for year in years:
    # load the global tiff as numpy array for this year
    f = countriesDir + "/Projections/Global/pop-" + year + ".tiff"
    world.append(pop.openTIFFasNParray(f))

print "Done loading tiffs, let the comparison begin: 👊"

# now let's compare the numbers for every simulation step and every country:
for index, year in enumerate(years):
    differencesum = 0
    if index > 0:
        for country in countries:

            # First the NPY numbers:
            laststepnpy = getPopNoIndividual(country, years[index-1])
            print "NPY population for " + pop.getCountryByID(country, WTP) + " in " + years[index-1] + ": " + str(laststepnpy)
            thisstepnpy = getPopNoIndividual(country, year)
            print "NPY population for " + pop.getCountryByID(country, WTP) + " in " + year + ": " + str(thisstepnpy)
            diffnpy = thisstepnpy - laststepnpy
            print "NPY increase for " + pop.getCountryByID(country, WTP) + " from " + years[index-1] + " to " + year + ": " + str(diffnpy)

            # Then the same for the tiff:
            t = world[index-1]
            laststeptiff = np.sum(t[t == int(country)])
            print "TIFF population for " + pop.getCountryByID(country, WTP) + " in " + years[index-1] + ": " + str(laststeptiff)

            t = world[index]
            thissteptiff = np.sum(t[t == int(country)])
            print "TIFF population for " + pop.getCountryByID(country, WTP) + " in " + year + ": " + str(thissteptiff)

            difftiff = thissteptiff - laststeptiff
            print "TIFF increase for " + pop.getCountryByID(country, WTP) + " from " + years[index-1] + " to " + year + ": " + str(difftiff)

            # and calculate the difference
            differencesum = differencesum + (diffnpy - difftiff)

        print " "
        print " --- "
        print "Total difference in " + str(year) + ": " + str(differencesum)
        print " --- "
        print " "
