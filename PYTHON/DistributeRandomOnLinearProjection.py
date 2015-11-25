from osgeo import gdal, osr
import os, datetime, sys, operator, logging, math, csv
import numpy as np
import numpy2geotiff as npgt
from datetime import datetime

# This will get rid of some floating point issues (well, reporting of them!)
old_settings = np.seterr(invalid="ignore")

# perform a linear projection first, then adjust randomly?
projectingLinear = False

# how many times do we want to simulate?
RUNS = 1
# this is a GeoTIFF that we'll use as a reference for our output - same bbox, resolution, CRS, etc.
referencetiff = ""

# some constants:
ruralCell = 1
urbanCell = 2

# some global variables that most functions need access to:
populationOld = []
populationNew = []
allIndexes = []
countryBoundaries = []
urbanRural = []
WTP = 0
WUP = 0

MAJ = 'Major area, region, country or area'

def main():
    global populationOld, populationNew, allIndexes, countryBoundaries, urbanRural, referencetiff, WTP, WUP

    logging.info('Starting...')

    logging.info("Reading reference GeoTIFF")
    referencetiff = os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia/GLUR/GLUR_Asia.tif'

    logging.info('Reading CSVs')

    # world URBAN population
    WUP = transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia/WUP2014Urban_Asia.csv')), "Country Code")
    # world TOTAL population
    WTP = transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia/WTP2014_Asia.csv')), "Country Code")

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
    array_to_raster(populationOld.reshape(matrix),
                             os.path.expanduser('~') + "/Dropbox/CISC - Global Population/Asia/Projections/Population-0-2000_test.tif")
    array_to_raster(populationNew.reshape(matrix),
                             os.path.expanduser('~') + "/Dropbox/CISC - Global Population/Asia/Projections/Population-0-2010_test.tif")

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
                populationProjected = projectLinear(populationOld, populationNew)
            else:
                # if not, we'll just start from the last raster:
                populationProjected = populationNew

            # loop through countries:
            for country in WTP:

                # adjust for the difference between raster and csv projection data:
                adjustPopulation(populationProjected, year, country)

                logging.info(" ----------------- ")


            # Save to GeoTIFF
            logging.info('Saving GeoTIFF.')
            # transform back to 2D array with the original dimensions:

            array_to_raster(populationNew.reshape(matrix),
                                     os.path.expanduser('~') + "/Dropbox/CISC - Global Population/Asia/Projections/Population-"+str(run)+"-"+str(year)+"_test.tif")

            # prepare everything for the next iteration

            populationOld = populationNew
            populationNew = populationProjected
            year = year + step

    logging.info('Done.')


########################################################
# Some convenience functions
########################################################

# logs the difference for urban and rural population
# between whats in the populationProjected and the
# DESA population projection CSV
def logDifference(populationProjected, year, country):
    urbraster = np.sum(populationProjected[
        np.logical_and(countryBoundaries == int(country),
                       urbanRural == urbanCell)])
    rurraster = np.sum(populationProjected[
        np.logical_and(countryBoundaries == int(country),
                       urbanRural == ruralCell)])

    popcsv = int(WTP[country][str(year)]) * 1000
    urbcsv = int(WUP[country][str(year)]) * 1000
    rurcsv = (popcsv-urbcsv)

    urbDiff = urbcsv - urbraster
    rurDiff = rurcsv - rurraster

    c = WTP[str(country)][MAJ]
    logging.info("Urban difference for " + c + ": " + str(urbDiff))
    logging.info("Rural difference for " + c + ": " + str(rurDiff))


# this one just compares the numbers from the raster to the CSV
# and then calls the corresponding functions to add or remove people.
def adjustPopulation(populationProjected, year, country):

    # figure out the difference between our linear projection
    # and what's in the table:

    urbraster = np.sum(populationProjected[
        np.logical_and(countryBoundaries == int(country),
                       urbanRural == urbanCell)])
    rurraster = np.sum(populationProjected[
        np.logical_and(countryBoundaries == int(country),
                       urbanRural == ruralCell)])

    popcsv = int(WTP[country][str(year)]) * 1000
    urbcsv = int(WUP[country][str(year)]) * 1000
    rurcsv = (popcsv - urbcsv)

    urbDiff = urbcsv - urbraster
    rurDiff = rurcsv - rurraster

    # This probably slows things down a bit... we've already computed the
    # required values, let's just use these...
    logDifference(populationProjected, year, country)

    logging.info("Adjusting")

    # urban:
    if (urbDiff > 0):  # add people
        logging.info("adding urban population")
        populationProjected = addPopulation(populationProjected, urbDiff,
                                            country, urbanCell)
    else:   # remove people
        logging.info("removing urban population")
        populationProjected = removePopulation(populationProjected,
                                               np.abs(urbDiff), country,
                                               urbanCell)

    # and rural:
    if (rurDiff > 0):  # add people
        logging.info("adding rural population")
        populationProjected = addPopulation(populationProjected, rurDiff,
                                            country, ruralCell)
    else:   # remove people
        logging.info("removing rural population")
        populationProjected = removePopulation(populationProjected,
                                               np.abs(rurDiff), country,
                                               ruralCell)

    logDifference(populationProjected, year, country)

    return populationProjected


def addPopulation(populationProjected, pop, country, cellType):

    try:
        randoms = np.all((countryBoundaries == int(country),
                          urbanRural == cellType), axis=0)
        if np.sum(randoms) < 0:
            logging.error("Can't add population to "
                          + WTP[str(country)][MAJ]
                          + ", country and " + str(cellType) + "conditions not"
                          + "satisfied?")
            return populationProjected

        randomIndexes = np.random.choice(allIndexes[randoms], pop)
        np.add.at(populationProjected, randomIndexes, 1)
    except Exception, e:
        logging.error("Could not add population to cells of type "
                      + str(cellType) + " in "
                      + WTP[str(country)][MAJ])
        logging.error(e)

    return populationProjected



def removePopulation(populationProjected, pop, country, cellType):

    try:
        # Added the condition that the cell has to have more than 0 population
        # Since we're doing subtract at with 1, this means we should create
        # fewer 'negative' cells...
        randoms = np.all((countryBoundaries == int(country),
                          urbanRural == cellType,
                          populationProjected > 0.0), axis=0)
        randomIndexes = np.random.choice(allIndexes[randoms], pop)
        np.subtract.at(populationProjected, randomIndexes, 1)

        while(populationProjected[populationProjected < 0.0].size > 0):
            # select random cells again, based on the number of people we need to remove again:

            # randoms = np.logical_and(countryBoundaries == country,
            #                          np.logical_and(populationProjected > 0.0,
            #                                         urbanRural == cellType))
            # This looks a more 'clean' way to implement the above:
            # http://stackoverflow.com/a/20528566/1256988
            a = countryBoundaries == int(country)
            b = populationProjected > 0.0
            c = urbanRural == cellType
            randoms = np.all((a, b, c), axis=0)
            # # Could also do:
            # abc = np.array((a, b, c))
            # randoms = np.logical_and.reduce(abc)

            less = populationProjected < 0.0
            count = np.abs(np.sum(populationProjected[less]))

            randomIndexes = np.random.choice(allIndexes[randoms], count)
            # set cells < 0 to 0
            populationProjected[less] = 0.0
            # and then remove the people we have just added somewhere else:
            if randomIndexes.size > 0.0:
                np.subtract.at(populationProjected, randomIndexes, 1)
            else:
                logging.info("Tried to remove more people than possible;\n"
                             "all cells of type " + str(cellType) + " have "
                             "already been set to 0.")

    except Exception, e:
        logging.error("Could not remove population from cells of type "
                      + str(cellType) + " in "
                      + WTP[str(country)][MAJ])
        logging.error(e)

    return populationProjected



# Turns a list of dictionaries into a single one:
def transposeDict(listOfDicts, pk):
    output = {}
    for dic in listOfDicts:
        output[dic[pk]] = dic
    return output


# calculates the change for each cell between first and second,
# and applies that change to second. Generates an array that is
# essentially a linear continuation of the trend between first and second.
def projectLinear(first, second):
    out = np.add(second, np.subtract(second, first))
    # replace all values below 0 with 0
    out[out<0] = 0
    return out



# saves a raster as a geotiff to dst_filename
def array_to_raster(array, dst_filename):
    global referencetiff

    reffile = gdal.Open(referencetiff)

    geotransform = reffile.GetGeoTransform()

    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(
        dst_filename,
        reffile.RasterXSize,
        reffile.RasterYSize,
        1,
        gdal.GDT_Float32, )

    dataset.SetGeoTransform(geotransform)

    dataset.SetProjection(reffile.GetProjection())
    dataset.GetRasterBand(1).SetNoDataValue(-3.4028230607371e+38)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        filename='output-'+datetime.utcnow().strftime("%Y%m%d")+'.log',
                        filemode='w',
                        format='%(asctime)s, line %(lineno)d %(levelname)-8s %(message)s')
    try:
        main()
    except Exception, e:
        logging.exception(e)
