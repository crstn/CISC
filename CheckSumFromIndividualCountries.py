# coding: utf-8
#!/usr/bin/env python

# calcuates the global total from the individual countries .npy files

import os
import time
import numpy as np
import tif2num as tn
import PopFunctions as pop
from PIL import Image
from osgeo import gdal

countriesDir = os.path.expanduser(
    '~') + '/Dropbox/CISC Data/IndividualCountries/'
os.chdir(countriesDir)


def getPopNo(country, year):
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

for year in years:
    summ = 0
    for country in countries:
        summ = summ + getPopNo(country, year)

    print year + ": " + str(summ / 1000000000.0)
