from osgeo import gdal, osr
import os, datetime, sys, operator, logging, math, csv
import numpy as np
from datetime import datetime
from PIL import Image

import PopFunctions as pop

# This will get rid of some floating point issues (well, reporting of them!)
old_settings = np.seterr(invalid="ignore")

# this will be added to the output file names, useful for testing.
postfix = "spilltest"

# how many times do we want to simulate?
RUNS = 1

# some global variables that most functions need access to:
populationOld = []
populationNew = []
allIndexes = []
countryBoundaries = []
urbanRural = []
WTP = 0
WUP = 0

def main():
    global populationOld, populationNew, allIndexes, countryBoundaries, urbanRural, referencetiff, WTP, WUP, runCountries, postfix

    # we'll read in the first command line arugument as the country ID we'll work on
    country = sys.argv[1]

    logging.info('Starting...')
    logging.info('Reading CSVs')

    # world URBAN population
    WUP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/Code/CISC/Data/DESA/WUP2014Urban.csv')), "Country Code")
    # world TOTAL population
    WTP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/Code/CISC/Data/DESA/WTP2014.csv')), "Country Code")

    logging.info('Reading Numpy arrays')

    # in this dataset: 1=rural, 2=urban
    urbanRural = np.load(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/IndividualCountries/'+country+'-urbanRural.npy')

    # save the shape of these arrays for later, so that we
    # can properly reshape them after flattening:
    matrix = urbanRural.shape

    # we flatten all arrays to 1D, so we don't have to deal with 2D arrays:
    urbanRural = urbanRural.ravel()
    countryBoundaries = np.load(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/IndividualCountries/'+country+'-boundary.npy').ravel()

    # load population raster datasets for 2000 and 2010
    populationOld = np.load(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/IndividualCountries/'+country+'-pop2000.npy').ravel()
    populationNew = np.load(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/IndividualCountries/'+country+'-pop2010.npy').ravel()

    # these arrays use very small negative numbers as NULL,
    # let's convert these to NAN
    populationOld[populationOld==np.nanmin(populationOld)] = np.nan
    populationNew[populationNew==np.nanmin(populationNew)] = np.nan

    # make an array of all indexes; we'll use this later:
    allIndexes = np.arange(countryBoundaries.size)

    logging.info("Growing population...")

    year = 2020
    step = 10
    while year <= 2050:

        populationProjected = populationNew

        pop.logSubArraySizes(populationProjected, year, country, WTP, countryBoundaries, urbanRural)

        # adjust for the difference between raster and csv projection data:
        pop.adjustPopulation(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural, allIndexes, matrix)

        # run the urbanization
        urbanRural = pop.urbanize(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural, allIndexes, matrix)


        # save the numpy arrays
        np.save(os.path.expanduser('~') + "/Dropbox/CISC - Global Population/IndividualCountries/Projections/"+country+"-"+str(year)+"-urbanRural.npy", urbanRural.reshape(matrix))
        np.save(os.path.expanduser('~') + "/Dropbox/CISC - Global Population/IndividualCountries/Projections/"+country+"-"+str(year)+"-pop.npy", populationNew.reshape(matrix))


        # also save as a tiff (not georeferenced, just to look at the data in QGIS)
        img = Image.fromarray(urbanRural.reshape(matrix))
        img.save(os.path.expanduser('~') + "/Dropbox/CISC - Global Population/IndividualCountries/Projections/"+country+"-"+str(year)+"-urbanRural.tiff")

        img = Image.fromarray(populationNew.reshape(matrix))
        img.save(os.path.expanduser('~') + "/Dropbox/CISC - Global Population/IndividualCountries/Projections/"+country+"-"+str(year)+"-pop.tiff")



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
                        filename='output-'+datetime.utcnow().strftime("%Y%m%d")+'.log',
                        filemode='w',
                        format='%(asctime)s, line %(lineno)d %(levelname)-8s %(message)s')
    main()
