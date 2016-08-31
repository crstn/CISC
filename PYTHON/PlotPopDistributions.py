import os, datetime, sys, operator, logging, math, csv, matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from osgeo import gdal, osr

print('Starting')


# Turns a list of dictionaries into a single one:
def transposeDict(listOfDicts, pk):
    output = {}
    for dic in listOfDicts:
        output[dic[pk]] = dic
    return output


os.chdir(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia/');



# This will get rid of some floating point issues (well, reporting of them!)
old_settings = np.seterr(invalid="ignore")

# run the simulation only on specific countries, or on all countries found in the input files?
runCountries = "all"
# runCountries = ["392", "764", "496", "144", "524"] # look up the country codes in the WUP or WTP csv files; make sure to put in quotes!

WTP = transposeDict(csv.DictReader(open('WTP2014_Asia.csv')), "Country Code")

urbanCell = 2
ruralCell = 1
MAJ = 'Major area, region, country or area'

matplotlib.style.use('fivethirtyeight')

# if we are running on all countries, make an array that contains all country IDs from WTP top iterate over later:
if(runCountries == "all"):
    print('Running simulation on all countries.')
    runCountries = []
    for country in WTP:
        if country in ["158","920","921","922","5500","5501"]:
            print 'Skipping ' + country
        else:
            runCountries.append(country)
else:
    print('Running simulation only on the following countries:')
    for country in runCountries:
        print(country)

print('Reading Numpy arrays')

# in this dataset: 1=rural, 2=urban
# we flatten ("ravel") all arrays to 1D, so we don't have to deal with 2D arrays:
urbanRural = np.load('GLUR_asia.tif.npy').ravel()

countryBoundaries = np.load('Nations_Asia.tif.npy').ravel()

# load population raster datasets for 2010
pop2010 = np.load('Population_2010_Asia.tif.npy').ravel()

# these arrays use very small negative numbers as NULL,
# let's convert these to NAN
pop2010[pop2010==np.nanmin(pop2010)] = np.nan

# make an array of all indexes; we'll use this later:
allIndexes = np.arange(countryBoundaries.size)


# this part is tricky: split up the number of countries into an nXn grid (roughly), then use those as subplots:
dim = np.ceil(np.sqrt(len(runCountries)))
subplots = np.array_split(runCountries, dim)
f, subplots = pyplot.subplots(len(subplots), len(subplots[0]), sharex='col', sharey='row', figsize=(30,30))

i = 0
j = 0

for country in runCountries:

    c = WTP[str(country)][MAJ]    

    # fetch the urban and rural cells for the current country:
    u = pop2010[
        np.logical_and(countryBoundaries == int(country),
                       urbanRural == urbanCell)]

    # sub-array with rural pop numbers for current country
    r = pop2010[
        np.logical_and(countryBoundaries == int(country),
                       urbanRural == ruralCell)]

    print c
    print "Urban: " + str(u.size)
    print "Rural: " + str(r.size)
    print " "

    # then chuck the histograms into the next subplot:
    subplots[i][j].hist(np.log(r+1), normed=True, bins=100, alpha=0.5, label='Rural')
    subplots[i][j].hist(np.log(u+1), normed=True, bins=100, alpha=0.5, label='Urban')
    subplots[i][j].set_title(c)

    if i == dim-1:
        i = 0
        j = j + 1
    else:
        i = i + 1

# pyplot.legend(loc='upper right')
# pyplot.show()
pyplot.savefig(os.path.expanduser('~') + '/Dropbox/Code/CISC/PYTHON/histograms/_hist-facets.pdf', bbox_inches='tight')

print('Done.')
