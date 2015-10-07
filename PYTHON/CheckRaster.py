from osgeo import gdal, osr
import os, datetime, sys, operator, logging, math, csv
import numpy as np
import numpy2geotiff as npgt
from datetime import datetime


# how many times do we want to simulate?
RUNS = 1
# this is a GeoTIFF that we'll use as a reference for our output - same bbox, resolution, CRS, etc.
referencetiff = ""
population = []
allIndexes = []
countryBoundaries = []
urbanRural = []

def main():
    global population, allIndexes, countryBoundaries, urbanRural, referencetiff

    logging.info('; CountryCode; Country; Total cells; Urban cells; Rural cells')

    filedir = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
    # this is the root folder of this project:
    dir = os.path.abspath(os.path.join(filedir, os.pardir))

    # world URBAN population
    WUP = transposeDict(csv.DictReader(open(os.path.join(dir, "Data/DESA/WUP2014Urban.csv"))), "Country Code")
    # world TOTAL population
    WTP = transposeDict(csv.DictReader(open(os.path.join(dir, "Data/DESA/WTP2014.csv"))), "Country Code")

    # logging.info(WUP)
    # logging.info(WTP)

    logging.info('Reading Numpy arrays')

    # in this dataset: 1=rural, 2=urban
    urbanRural = np.load(os.path.join(dir, "Data/NumpyLayers/UrbanRural.npy"))

    # save the shape of these arrays for later, so that we
    # can properly reshape them after flattening:
    matrix = urbanRural.shape

    # we flatten all arrays to 1D, so we don't have to deal with 2D arrays:
    urbanRural = urbanRural.ravel()
    countryBoundaries = np.load(os.path.join(dir, "Data/NumpyLayers/NationOutlines.npy")).ravel()

    # print all unique country codes from this array:
    rasterCountries = np.unique(countryBoundaries)
    # logging.info(rasterCountries)
    csvCountries = np.array(list(WTP)).astype(int)
    # logging.info(csvCountries)

    logging.info("Countries in raster, but not in CSV:")
    for rc in np.sort(rasterCountries):
        if not (rc in csvCountries):
            logging.info(rc)

    logging.info("Countries in CSV, but not in raster:")
    for cs in np.sort(csvCountries):
        if not (cs in rasterCountries):
            logging.info(cs)

    # population = np.load(os.path.join(dir, "Data/NumpyLayers/Population2000.npy")).ravel()
    #
    # # replace no data values with 0:
    # population[population == -9223372036854775808] = 0
    #
    # # and an array of all indexes; we'll use this later:
    # allIndexes = np.arange(countryBoundaries.size)


    # for countryCode in WTP:
    #     if int(countryCode) < 900:
    #
    #         country = WTP[countryCode]
    #         wupcountry = WUP[countryCode]
    #
    #         allCells   = countryBoundaries[countryBoundaries==int(countryCode)].size
    #         urbanCells = allIndexes[np.logical_and(countryBoundaries==int(countryCode), urbanRural==1)].size
    #         ruralCells = allIndexes[np.logical_and(countryBoundaries==int(countryCode), urbanRural==2)].size
    #
    #         logging.info("; " + countryCode + "; "+country["Major area, region, country or area"]+"; "+str(allCells)+"; "+str(urbanCells)+"; "+str(ruralCells))




########################################################
# Some convenience functions
########################################################


# Turns a list of dictionaries into a single one:
def transposeDict(listOfDicts, pk):
    output = {}
    for dic in listOfDicts:
        output[dic[pk]] = dic
    return output




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        filename='rasterCheck-'+datetime.utcnow().strftime("%Y%m%d%H%M")+'.log',
                        filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    try:
        main()
    except Exception, e:
        logging.exception(e)
