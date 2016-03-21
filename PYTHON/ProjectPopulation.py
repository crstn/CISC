from osgeo import gdal, osr
import os, datetime, sys, operator, logging, math, csv
import numpy as np
from datetime import datetime

import PopFunctions as pop

# This will get rid of some floating point issues (well, reporting of them!)
old_settings = np.seterr(invalid="ignore")

# run the simulation only on specific countries, or on all countries found in the input files?
# runCountries = "all"
runCountries = ["392", "764"] # look up the country codes in the WUP or WTP csv files; make sure to put in quotes!

# this will be added to the output file names, useful for testing.
postfix = "quicktest"

# perform a linear projection first, then adjust randomly?
projectingLinear = False

# how many times do we want to simulate?
RUNS = 1
# this is a GeoTIFF that we'll use as a reference for our output - same bbox, resolution, CRS, etc.
referencetiff = ""

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

    logging.info('Starting...')

    logging.info("Reading reference GeoTIFF")
    referencetiff = os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia/GLUR/GLUR_Asia.tif'

    logging.info('Reading CSVs')

    # world URBAN population
    WUP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia/WUP2014Urban_Asia.csv')), "Country Code")
    # world TOTAL population
    WTP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia/WTP2014_Asia.csv')), "Country Code")

    # if we are running on all countries, make an array that contains all country IDs from WTP top iterate over later:
    if(runCountries == "all"):
        logging.info('Running simulation on all countries.')
        runCountries = []
        for country in WTP:
            runCountries.append(country)
    else:
        logging.info('Running simulation only on the following countries:')
        for country in runCountries:
            logging.info(country)

    logging.info('Reading Numpy arrays')

    # in this dataset: 1=rural, 2=urban
    urbanRural = np.load(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia/GLUR_asia.tif.npy')

    # save the shape of these arrays for later, so that we
    # can properly reshape them after flattening:
    matrix = urbanRural.shape

    # we flatten all arrays to 1D, so we don't have to deal with 2D arrays:
    urbanRural = urbanRural.ravel()
    countryBoundaries = np.load(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia/Nations_Asia.tif.npy').ravel()

    # load population raster datasets for 2000 and 2010
    populationOld = np.load(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia/Population_2000_Asia.tif.npy').ravel()
    populationNew = np.load(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia/Population_2010_Asia.tif.npy').ravel()

    # these arrays use very small negative numbers as NULL,
    # let's convert these to NAN
    populationOld[populationOld==np.nanmin(populationOld)] = np.nan
    populationNew[populationNew==np.nanmin(populationNew)] = np.nan

    # let's save the 2000 and 2010 tiffs, just to have all the output in one folder:
    pop.array_to_raster(populationOld.reshape(matrix),
                             os.path.expanduser('~') + "/Dropbox/CISC - Global Population/Asia/Projections/Population-0-2000_"+postfix+".tif", referencetiff)
    pop.array_to_raster(populationNew.reshape(matrix),
                             os.path.expanduser('~') + "/Dropbox/CISC - Global Population/Asia/Projections/Population-0-2010_"+postfix+".tif", referencetiff)

    # make an array of all indexes; we'll use this later:
    allIndexes = np.arange(countryBoundaries.size)

    logging.info("Growing population...")

    for run in range(RUNS):

        logging.info( "Run no. " + str(run))

        year = 2020
        step = 10
        while year <= 2050:

            # start by applying a linear projection to the WHOLE raster?
            if(projectingLinear):
                populationProjected = pop.projectLinear(populationOld, populationNew)
            else:
                # if not, we'll just start from the last raster:
                populationProjected = populationNew

            # loop through countries:
            for country in runCountries:

                pop.logSubArraySizes(populationProjected, year, country, WTP, countryBoundaries, urbanRural)

                # adjust for the difference between raster and csv projection data:
                pop.adjustPopulation(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural, allIndexes)

                logging.info(" ----------------- ")


            # Save to GeoTIFF
            logging.info('Saving GeoTIFF.')
            # transform back to 2D array with the original dimensions:

            pop.array_to_raster(populationNew.reshape(matrix),
                                     os.path.expanduser('~') + "/Dropbox/CISC - Global Population/Asia/Projections/Population-"+str(run)+"-"+str(year)+"_"+postfix+".tif", referencetiff)

            # prepare everything for the next iteration

            populationOld = populationNew
            populationNew = populationProjected
            year = year + step

    logging.info('Done.')




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,  # toggle this between INFO for debugging and ERROR for "production"
                        filename='output-'+datetime.utcnow().strftime("%Y%m%d")+'.log',
                        filemode='w',
                        format='%(asctime)s, line %(lineno)d %(levelname)-8s %(message)s')
    try:
        main()
    except Exception, e:
        logging.exception(e)
