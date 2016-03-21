from osgeo import gdal, osr
import os, datetime, sys, operator, logging, math, csv
import numpy as np
from datetime import datetime

# some constants:
ruralCell = 1
urbanCell = 2
MAJ = 'Major area, region, country or area'

# Spillover: go into a loop where we remove population above limit and "push" them to the neighboring cells
# (I was thinking about calling this function "gentrify"...)
# IMPORTANT: we only select by country and whether the cell is overcrowded,
# so that we can spill over into other cell types, i.e. from urban to rural
def spillover(populationProjected, country, limit):

    overcrowded = np.all((countryBoundaries == country, populationProjected > limit), axis = 0)
    # for every overcrowded cell, distribute the surplus population randomly among its neighbors
    for fullCell in allIndexes[overcrowded]:
        surplus = populationProjected[fullCell] - limit # by how much are we over the limit?
        # reset those cells to the limit value:
        populationProjected[fullCell] = limit
        # and move the extra people to the neighbors:
        wilsons = getNeighbours(fullCell, shape)
        rI = np.random.choice(wilsons, surplus)
        np.add.at(populationProjected, rI, 1)

    print populationProjected.reshape(shape)
    return populationProjected


def logSubArraySizes(populationProjected, year, country, WTP, countryBoundaries, urbanRural):

    logging.info("  ----   ")
    logging.info("Array sizes for " + WTP[str(country)][MAJ])

    logging.info("Population: " + str(populationProjected[countryBoundaries == int(country)].size))
    logging.info("Nations: " + str(countryBoundaries[countryBoundaries == int(country)].size))
    logging.info("Urban: " + str(urbanRural[np.logical_and(countryBoundaries == int(country), urbanRural == urbanCell)].size))
    logging.info("Rural: " + str(urbanRural[np.logical_and(countryBoundaries == int(country), urbanRural == ruralCell)].size))
    logging.info("  ----   ")


# logs the difference for urban and rural population
# between whats in the populationProjected and the
# DESA population projection CSV
def logDifference(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural):
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
def adjustPopulation(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural, allIndexes):

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
    logDifference(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural)

    logging.info("Adjusting")

    # urban:
    if (urbDiff > 0):  # add people
        logging.info("adding urban population")
        populationProjected = addPopulation(populationProjected, urbDiff,
                                            country, urbanCell, WTP, WUP, countryBoundaries, urbanRural, allIndexes)
    else:   # remove people
        logging.info("removing urban population")
        populationProjected = removePopulation(populationProjected,
                                               np.abs(urbDiff), country,
                                               urbanCell, WTP, WUP, countryBoundaries, urbanRural, allIndexes)

    # and rural:
    if (rurDiff > 0):  # add people
        logging.info("adding rural population")
        populationProjected = addPopulation(populationProjected, rurDiff,
                                            country, ruralCell, WTP, WUP, countryBoundaries, urbanRural, allIndexes)
    else:   # remove people
        logging.info("removing rural population")
        populationProjected = removePopulation(populationProjected,
                                               np.abs(rurDiff), country,
                                               ruralCell, WTP, WUP, countryBoundaries, urbanRural, allIndexes)

    logDifference(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural)

    return populationProjected


def addPopulation(populationProjected, pop, country, cellType, WTP, WUP, countryBoundaries, urbanRural, allIndexes):

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



def removePopulation(populationProjected, pop, country, cellType, WTP, WUP, countryBoundaries, urbanRural, allIndexes):

    try:
        # Added the condition that the cell has to have more than 0 population
        # Since we're doing subtract at with 1, this means we should create
        # fewer 'negative' cells...

        a = countryBoundaries == int(country)
        b = populationProjected >= 1.0
        c = urbanRural == cellType
        randoms = np.all((a, b, c), axis=0)

        randomIndexes = np.random.choice(allIndexes[randoms], pop)
        np.subtract.at(populationProjected, randomIndexes, 1)

        while(populationProjected[populationProjected < 0.0].size > 0):
            # select random cells again, based on the number of people we need to remove again:

            # a and c don't change (see above), but b does. Not sure whether we really need to do this,
            # but just to be safe:
            b = populationProjected >= 1.0
            randoms = np.all((a, b, c), axis=0)

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



# Returns an array of indexes that correspond to the 3x3 neighborhood of the index cell
# in a raveled (1D) matrix based on the # shape of the original (2D) matrix.
# Returns only neighbors within shape, exlcuding the input cell
def getNeighbours(index, shape):
    twoDIndex = oneDtoTwoD(index, shape)
    row = twoDIndex[0]
    col = twoDIndex[1]

    neighbors = []

    for r in range(-1, 2):
        for c in range(-1, 2):
            rn = row + r
            cn = col + c
            if r != 0 or c !=0: # don't add the original cell
                if 0 <= rn < shape[0] and 0 <= cn < shape[1]: # don't add neighbors that are outside of the shape!
                    neighbors.append(twoDtoOneD(rn, cn, shape))

    return neighbors


# Computes the "raveled" index from a 2D index. Shape is a tuple (rows, columns).
# WARNING: does NOT check whether row and col are outside of shape!
def twoDtoOneD(row, col, shape):
    return (row * shape[1]) + col



# Computes the 2D index as a tuple (row, column) from its "raveled" index.
# Shape is a tuple (rows, columns).
def oneDtoTwoD(index, shape):
    return int(index/shape[1]), int(index%shape[1])



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
def array_to_raster(array, dst_filename, referencetiff):

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
