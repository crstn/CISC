import os, datetime, sys, operator, logging, math, csv, matplotlib
import numpy as np
import pandas as pd
import PopFunctions as pop
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


os.chdir(os.path.expanduser('~') + '/Dropbox/CISCdata/DESA');

WTP = transposeDict(csv.DictReader(open('WPP2015_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES - countries only.csv')), "Country code")

MAJ = 'Major area, region, country or area'

# matplotlib.style.use('fivethirtyeight')

step = 10

years = range(2015+step,2101,step)

for country in WTP:
    proj = [];
    for year in years:
        # calculate difference compared to last year:
        thisyear = pop.getNumberForYear(WTP, year, country)
        lastyear = pop.getNumberForYear(WTP, year-step, country)
        diff = thisyear - lastyear
        growthrate = float(diff) / float(lastyear)
        proj.append(growthrate)

    try:
        # pyplot.plot(years, proj, label = pop.getCountryByID(int(country), WTP))
        pyplot.plot(years, proj)
    except UnicodeDecodeError:
        print "Error for " + pop.getCountryByID(int(country), WTP)



    # .hist(np.log(r+1), normed=True, bins=100, alpha=0.5, label='Rural')
    # subplots[i][j].hist(np.log(u+1), normed=True, bins=100, alpha=0.5, label='Urban')
    # subplots[i][j].set_title(c)
    #
    # if i == dim-1:
    #     i = 0
    #     j = j + 1
    # else:
    #     i = i + 1

pyplot.legend(loc='upper right')
pyplot.show()
# pyplot.savefig(os.path.expanduser('~') + '/Dropbox/Code/CISC/PYTHON/histograms/_hist-facets.pdf', bbox_inches='tight')

print('Done.')
