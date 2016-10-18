from osgeo import gdal, osr
import os, datetime, sys, operator, logging, math, csv
import numpy as np
from datetime import datetime
from PIL import Image

import PopFunctions as pop

# This will get rid of some floating point issues (well, reporting of them!)
old_settings = np.seterr(invalid="ignore")

# some global variables that most functions need access to:
populationOld = []
populationNew = []
allIndexes = []
countryBoundaries = []
urbanRural = []
WTP = 0
WUP = 0

def main():

    endyear = 2100

    global populationOld, populationNew, allIndexes, countryBoundaries, urbanRural, referencetiff, WTP, WUP, runCountries

    # we'll read in the first command line arugument as the country ID we'll work on
    country = sys.argv[1]

    logging.info('Starting...')
    logging.info('Reading CSVs')

    # world URBAN population
    WUP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC Data/DESA/WUPto2100_Peter_MEAN.csv')), "Country Code")
    # world TOTAL population
    WTP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC Data/DESA/WPP2015_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.csv')), "Country code")

    try:
        print " --- "
        print "Starting " + pop.getCountryByID(country, WTP) + "("+country+")"
        print " --- "
    except KeyError:
        print " --- "
        print "ERROR: COUNTRY " + country + " NOT IN CSV"
        print "Skipping, saving empty .npy files"
        print " --- "
        # save empty files so that the parallel processing
        # moves on to the next country
        year = 2020
        step = 10
        while year <= endyear:
            open(os.path.expanduser('~') + "/Dropbox/CISC Data/IndividualCountries/Projections/"+country+"-"+str(year)+"-urbanRural.npy", 'a')
            open(os.path.expanduser('~') + "/Dropbox/CISC Data/IndividualCountries/Projections/"+country+"-"+str(year)+"-pop.npy", 'a')
            year = year + step


    logging.info('Reading Numpy arrays')

    # in this dataset: 1=rural, 2=urban
    urbanRural = np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+country+'.0-urbanRural.npy')

    # save the shape of these arrays for later, so that we
    # can properly reshape them after flattening:
    matrix = urbanRural.shape

    # we flatten all arrays to 1D, so we don't have to deal with 2D arrays:
    urbanRural = urbanRural.ravel()
    countryBoundaries = np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+country+'.0-boundary.npy').ravel()

    # load population raster datasets for 2000 and 2010
    populationOld = np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+country+'.0-pop2000.npy').ravel()
    populationNew = np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+country+'.0-pop2010.npy').ravel()

    # these arrays use very small negative numbers as NULL,
    # let's convert these to NAN
    populationOld[populationOld==np.nanmin(populationOld)] = np.nan
    populationNew[populationNew==np.nanmin(populationNew)] = np.nan

    # next, we'll cast the pop numbers to int (should save some memory):
    populationOld = populationOld.astype(np.int64)
    populationNew = populationNew.astype(np.int64)

    # make an array of all indexes; we'll use this later:
    allIndexes = np.arange(countryBoundaries.size)

    logging.info("Growing population...")

    year = 2020
    step = 10
    while year <= endyear:

        populationProjected = populationNew

        # pop.logSubArraySizes(populationProjected, year, country, WTP, countryBoundaries, urbanRural)

        # adjust for the difference between raster and csv projection data:
        pop.adjustPopulation(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural, allIndexes, matrix)

        # run the urbanization, but only if the urban population has increased!
        if (pop.getNumberForYear(WUP, year, country) > pop.getNumberForYear(WUP, year-10, country)):
            urbanRural = pop.urbanize(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural, allIndexes, matrix)
        # else:
        #     print "Skipping urbanization."
        #     print pop.getCountryByID(country, WTP) + " urban pop " + str(year) +": " + str(pop.getNumberForYear(WUP, year, country))
        #     print pop.getCountryByID(country, WTP) + " urban pop " + str(year-10) +": " + str(pop.getNumberForYear(WUP, year-10, country))


        # save the numpy arrays
        np.save(os.path.expanduser('~') + "/Dropbox/CISC Data/IndividualCountries/Projections/"+country+"-"+str(year)+"-urbanRural.npy", urbanRural.reshape(matrix))
        np.save(os.path.expanduser('~') + "/Dropbox/CISC Data/IndividualCountries/Projections/"+country+"-"+str(year)+"-pop.npy", populationNew.reshape(matrix))


        # also save as a tiff (not georeferenced, just to look at the data in QGIS)
        # Turn this off when in production!
        # img = Image.fromarray(urbanRural.reshape(matrix))
        # img.save(os.path.expanduser('~') + "/Dropbox/CISC Data/IndividualCountries/Projections/"+country+"-"+str(year)+"-urbanRural.tiff")
        #
        # img = Image.fromarray(populationNew.astype(float).reshape(matrix))
        # img.save(os.path.expanduser('~') + "/Dropbox/CISC Data/IndividualCountries/Projections/"+country+"-"+str(year)+"-pop.tiff")

        # prepare everything for the next iteration

        populationOld = populationNew
        populationNew = populationProjected
        year = year + step

    logging.info('Done.')




if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "This script expects a country ID as parameter, e.g."
        print "python ProjectPopulationOneCountry.py 156"
        print "to project the population for China. Check the WUP/WTP csv files for the IDs."
        sys.exit()

    logging.basicConfig(level=logging.ERROR,  # toggle this between INFO for debugging and ERROR for "production"
                        filename='output-'+datetime.utcnow().strftime("%Y%m%d")+ '-'+sys.argv[1]+'.log',
                        filemode='w',
                        format='%(asctime)s, line %(lineno)d %(levelname)-8s %(message)s')
    main()
