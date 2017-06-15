# coding: utf-8
#!/usr/bin/env python

from osgeo import gdal, osr
import os, datetime, sys, operator, logging, math, csv
import numpy as np
from datetime import datetime
from PIL import Image

import PopFunctions as pop

target = os.path.expanduser('~') + "/Dropbox/CISC Data/IndividualCountries/Projections/Test2/"
# target = '/Volumes/Solid Guy/Sandbox/'

src = os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'

# Turn logging of urban / rural / total population at every step on of off:
checkNumbers = False

# overwrite existing projections for the same country?
overwrite = False

endyear = 2050 # TODO CHANGE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# This will get rid of some floating point issues (well, reporting of them!)
# old_settings = np.seterr(invalid="ignore")

# some global variables that most functions need access to:
populationOld = []
populationNew = []
countryBoundaries = []
urbanRural = []
WTP = 0
WUP = 0



def main():

    global populationOld, populationNew, countryBoundaries, urbanRural, referencetiff, WTP, WUP, runCountries, endyear, target

    country = sys.argv[1]
    scenario = sys.argv[2]
    urbanRuralVersion = sys.argv[3]

    # create output dir if it doesn't exist yet:
    target = target + urbanRuralVersion + "/" + scenario + "/"
    if not os.path.exists(target):
        os.makedirs(target)

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
        test = pop.getNumberForYear(WTP, 2010, country)

    except KeyError:
        print " --- "
        print "ERROR: COUNTRY " + country + " NOT IN CSV"
        print "Skipping, saving empty .npy files"
        print " --- "

        with open("logs/skippedcountries.log", "a") as myfile:
            myfile.write(str(country)+', ')

        # save empty files so that the parallel processing
        # moves on to the next country
        year = 2010
        step = 10
        while year <= endyear:
            open(target +country+"-"+str(year)+"-urbanRural.npy", 'a')
            open(target +country+"-"+str(year)+"-pop.npy", 'a')
            year = year + step

        return

    logging.info('Reading Numpy arrays')

    urbanRural = np.load(src+country+'.0-UrbanRural-'+urbanRuralVersion+'.npy')

    # load population raster datasets for 2000 and 2010
    populationOld = np.load(src+country+'.0-pop2000.npy')
    populationNew = np.load(src+country+'.0-pop2010.npy')

    #load the row and column indexes
    rows  = np.load(src+country+'.0-rows.npy')
    cols  = np.load(src+country+'.0-cols.npy')

    #load the cell areas:
    areas = np.load(src+country+'.0-areas.npy')

    # these arrays use very small negative numbers as NULL,
    # let's just set these to 0:
    populationOld[populationOld < 0] = 0
    populationNew[populationNew < 0] = 0

    # next, we'll cast the pop numbers to int (should save some memory):
    populationOld = populationOld.astype(np.int64)
    populationNew = populationNew.astype(np.int64)

    logging.info("Starting simulation...")

    year = 2010
    step = 10
    while year <= endyear:

        logging.info(" --- ")
        logging.info(str(year))
        logging.info(" --- ")


        populationProjected = populationNew

        # pop.logSubArraySizes(populationProjected, year, country, WTP, countryBoundaries, urbanRural)

        # adjust for the difference between raster and csv projection data:
        pop.adjustPopulation(populationProjected, year, country, WTP, WUP, urbanRural, rows, cols, areas)

        # Skip the urbanization for 2010, because we know the urban extents;
        # the purpose of running the population adjstment for 2010 was just to make
        # sure that our simulations start from a situation where the numbers in the CSVs
        # match the maps.
        if(year > 2010):
            # run the urbanization
            # calculate densities:
            densities = np.divide(populationProjected, areas)
            urbanRural = pop.urbanize(densities, urbanRural, country, year, WUP)
            # after the urbanization, we have to re-adjust the population, because
            # otherwise the numbers for urban and rural will be off from the IIASA numbers
            pop.adjustPopulation(populationProjected, year, country, WTP, WUP, urbanRural, rows, cols, areas)


        # save the numpy arrays
        np.save(target + country + "-"+str(year)+"-urbanRural.npy", urbanRural)
        np.save(target + country + "-"+str(year)+"-pop.npy", populationNew)


        if checkNumbers:
            pop.logDifference(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural)


        # pop.logDifference(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural)

        # prepare everything for the next iteration
        populationOld = populationNew
        populationNew = populationProjected
        year = year + step



    logging.info('Done.')




if __name__ == '__main__':

    if len(sys.argv) != 4:
        print "This script takes three arguments:"
        print "1. The country ID (e.g., 156 for China)"
        print "2. The scenario (SSP1 to SSP5)"
        print "3. The urban/rural version (GRUMP or GlobCover)"
        print ""
        sys.exit();

    logging.basicConfig(level=logging.ERROR,  # toggle this between INFO for debugging and ERROR for "production"
                        filename='logs/output-'+datetime.utcnow().strftime("%Y%m%d")+ '-'+sys.argv[1]+'-'+sys.argv[2]+'-'+sys.argv[3]+'.log',
                        filemode='w',
                        format='%(asctime)s, line %(lineno)d %(levelname)-8s %(message)s')

    if os.path.isfile(target + sys.argv[1]+"-"+str(endyear)+"-pop.npy") and not overwrite :
        print "Simulations for " +sys.argv[1]+ " already done; overwriting is turned off."
    else:
        main()
