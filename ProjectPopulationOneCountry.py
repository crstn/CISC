from osgeo import gdal, osr
import os, datetime, sys, operator, logging, math, csv
import numpy as np
from datetime import datetime
from PIL import Image

import PopFunctions as pop

# select from SSP1 through SSP5
scenario = 'SSP5'

# target = os.path.expanduser('~') + "/Dropbox/CISC Data/IndividualCountries/Projections/"
target = '/Volumes/Solid Guy/SSP5 2017-02-17/'

# Turn saving of TIFFS for debugging on or off:
savetiffs = False

# overwrite existing projections for the same country?
overwrite = True

endyear = 2100

# This will get rid of some floating point issues (well, reporting of them!)
# old_settings = np.seterr(invalid="ignore")

# some global variables that most functions need access to:
populationOld = []
populationNew = []
allIndexes = []
countryBoundaries = []
urbanRural = []
WTP = 0
WUP = 0



def main():

    global populationOld, populationNew, allIndexes, countryBoundaries, urbanRural, referencetiff, WTP, WUP, runCountries, endyear, target

    # we'll read in the first command line arugument as the country ID we'll work on
    country = sys.argv[1]

    logging.info('Starting...')
    logging.info('Reading CSVs')

    # TOTAL population per country
    # WTP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC Data/DESA/WPP2015_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.csv')), "Country code")
    WTP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC Data/SSPs/pop-'+scenario+'.csv')), "Country code")
    # URBAN population per country
    # WUP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC Data/DESA/WUPto2100_Peter_MEAN.csv')), "Country Code")
    WUP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC Data/SSPs/urbpop-'+scenario+'.csv')), "Country code")


    try:
        print " --- "
        print "Starting " + str(pop.getCountryByID(country, WTP)) + "("+str(country)+")"
        # just to see if this works:
        test = pop.getNumberForYear(WTP, 2020, country)

    except KeyError:
        print " --- "
        print "ERROR: COUNTRY " + country + " NOT IN CSV"
        print "Skipping, saving empty .npy files"
        print " --- "

        with open("skippedcountries.log", "a") as myfile:
            myfile.write(str(country)+', ')

        # save empty files so that the parallel processing
        # moves on to the next country
        year = 2020
        step = 10
        while year <= endyear:
            open(target+country+"-"+str(year)+"-urbanRural.npy", 'a')
            open(target+country+"-"+str(year)+"-pop.npy", 'a')
            year = year + step

        return

    logging.info('Reading Numpy arrays')

    urbanRural = np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+country+'.0-urbanSuburbanRural.npy')

    # save the shape of these arrays for later, so that we
    # can properly reshape them after flattening:
    matrix = urbanRural.shape



    # we flatten all arrays to 1D, so we don't have to deal with 2D arrays:
    urbanRural = urbanRural.ravel()

    countryBoundaries = np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+country+'.0-boundary.npy').ravel()

    # load population raster datasets for 2000 and 2010
    populationOld = np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+country+'.0-pop2000.npy').ravel()

    populationNew = np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+country+'.0-pop2010.npy').ravel()

    if savetiffs:
        # also save copies of the input for visualization
        img = Image.fromarray(urbanRural.reshape(matrix))
        img.save(os.path.expanduser('~') + "/Desktop/Projections/"+country+"-2010-urbanRural.tiff")

        img = Image.fromarray(populationOld.astype(float).reshape(matrix))
        img.save(os.path.expanduser('~') + "/Desktop/Projections/"+country+"-2000-pop.tiff")

        img = Image.fromarray(populationNew.astype(float).reshape(matrix))
        img.save(os.path.expanduser('~') + "/Desktop/Projections/"+country+"-2010-pop.tiff")


    # calculate thresholds for urbanization before we start the simulation:

    urbanthreshold, suburbanthreshold = pop.getThresholds(country, populationOld, countryBoundaries, urbanRural, WTP)


    # these arrays use very small negative numbers as NULL,
    # let's just set these to 0:
    populationOld[populationOld < 0] = 0
    populationNew[populationNew < 0] = 0

    # next, we'll cast the pop numbers to int (should save some memory):
    populationOld = populationOld.astype(np.int64)
    populationNew = populationNew.astype(np.int64)

    # make an array of all indexes; we'll use this later:
    allIndexes = np.arange(countryBoundaries.size)

    logging.info("Growing population...")

    year = 2020
    step = 10
    while year <= endyear:

        logging.info(" --- ")
        logging.info(str(year))
        logging.info(" --- ")


        populationProjected = populationNew

        # pop.logSubArraySizes(populationProjected, year, country, WTP, countryBoundaries, urbanRural)

        # adjust for the difference between raster and csv projection data:
        pop.adjustPopulation(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural, allIndexes, matrix)

        # run the urbanization
        urbanRural = pop.urbanize(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural, allIndexes, matrix, urbanthreshold, suburbanthreshold)
        # after the urbanization, we have to re-adjust the population, because # otherwise the numbers for urban and rural will be off from the DESA numbers
        pop.adjustPopulation(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural, allIndexes, matrix)


        # save the numpy arrays
        np.save(target + country + "-"+str(year)+"-urbanRural.npy", urbanRural.reshape(matrix))
        np.save(target + country + "-"+str(year)+"-pop.npy", populationNew.reshape(matrix))


        if savetiffs:
            # also save as a tiff (not georeferenced, just to look at the data in QGIS)
            # Turn this off when in production!
            img = Image.fromarray(urbanRural.reshape(matrix))
            img.save(os.path.expanduser('~') + "/Desktop/Projections/"+country+"-"+str(year)+"-urbanRural.tiff")

            img = Image.fromarray(populationNew.astype(float).reshape(matrix))
            img.save(os.path.expanduser('~') + "/Desktop/Projections/"+country+"-"+str(year)+"-pop.tiff")

        # pop.logDifference(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural)

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

    if os.path.isfile(target + sys.argv[1]+"-"+str(endyear)+"-pop.npy") and not overwrite :
        print "Simulations for " +sys.argv[1]+ " already done; overwriting is turned off."
    else:
        main()
