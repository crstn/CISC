from osgeo import gdal, osr
import os, datetime, sys, operator, logging, math, csv, bottleneck
import numpy as np
from datetime import datetime

# some constants:
ruralCell = 1
urbanCell = 2
MAJ = 'Major area, region, country or area'

# Factor to simulate densities in the highest density areas going down.
# TODO: talk to Peter to see whether this is a good idea, and what a good number might be.
# For now, we keep this at 1.0, ie. no thinning
thinningFactor = 0.8

# number of top cells to take into account when calculating the average of the N cells with the highest population per country:
topNcells = 20


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
def adjustPopulation(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural, allIndexes, shape):

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
                                            country, urbanCell, WTP, WUP, countryBoundaries, urbanRural, allIndexes, shape)
    else:   # remove people
        logging.info("removing urban population")
        populationProjected = removePopulation(populationProjected,
                                               np.abs(urbDiff), country,
                                               urbanCell, WTP, WUP, countryBoundaries, urbanRural, allIndexes)

    # and rural:
    if (rurDiff > 0):  # add people
        logging.info("adding rural population")
        populationProjected = addPopulation(populationProjected, rurDiff,
                                            country, ruralCell, WTP, WUP, countryBoundaries, urbanRural, allIndexes, shape)
    else:   # remove people
        logging.info("removing rural population")
        populationProjected = removePopulation(populationProjected,
                                               np.abs(rurDiff), country,
                                               ruralCell, WTP, WUP, countryBoundaries, urbanRural, allIndexes)

    logDifference(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural)

    return populationProjected


# returns the values of the top N cells
def getTopNCells(N, populationProjected):
    return bottleneck.partsort(populationProjected, populationProjected.size-N)[-N:]




def addPopulation(populationProjected, pop, country, cellType, WTP, WUP, countryBoundaries, urbanRural, allIndexes, shape):

    # try:

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


    if(cellType == urbanCell):

        logging.info("Checking if we need to spill over...")

        # before we start, we'll assume that a certain share of the current max is the
        # limit after which we spill over into neighboring cells:

        a = countryBoundaries == int(country)
        b = urbanRural == urbanCell

        topN = getTopNCells(topNcells, populationProjected[np.all((a, b), axis=0)])

        # we'll use the mean of the top n cells of each country as the maximum
        mx = np.sum(topN) / topNcells
        # ... considering the thinning factor
        limit = mx * thinningFactor

        logging.info(limit)

        # Print some stats after the spillover:

        urb = urbanRural == urbanCell
        rur = urbanRural == ruralCell

        logging.info("Rural max:" + str(np.nanmax(populationProjected[np.all((a, rur), axis=0)])))
        logging.info("Urban min:"  + str(np.nanmin(populationProjected[np.all((a, urb), axis=0)])))
        logging.info("Urban max:"  + str(np.nanmax(populationProjected[np.all((a, urb), axis=0)])))


        # Repeat the spillover function as long as there are cells above the limit
        # TODO: this may run into an infinite loop!
        while (int(np.nanmax(populationProjected[np.all((a, b), axis=0)])) > int(limit)):

            currentMax = np.nanmax(populationProjected[np.all((a, b), axis=0)])

            logging.info("Limit: " + str(limit))

            logging.info("Current max:" + str(currentMax))

            # logging.info("Are we over the limit? " + str(int(currentMax) > int(limit)))

            c = populationProjected > limit

            logging.info("Cells over limit: " + str(populationProjected[np.all((a, b, c), axis=0)]))
            logging.info("Indexes: " + str(allIndexes[np.all((a, b, c), axis=0)]))

            populationProjected = spillover(populationProjected, country, limit, countryBoundaries, urbanRural, allIndexes, shape)

        # Print some stats after the spillover:

        logging.info("Rural max:" + str(np.nanmax(populationProjected[np.all((a, rur), axis=0)])))
        logging.info("Urban min:"  + str(np.nanmin(populationProjected[np.all((a, urb), axis=0)])))
        logging.info("Urban max:"  + str(np.nanmax(populationProjected[np.all((a, urb), axis=0)])))


    # except Exception, e:
    #     logging.error("Could not add population to cells of type "
    #                   + str(cellType) + " in "
    #                   + WTP[str(country)][MAJ])
    #     logging.error(e)

    return populationProjected



def removePopulation(populationProjected, pop, country, cellType, WTP, WUP, countryBoundaries, urbanRural, allIndexes):

    # try:
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

    # except Exception, e:
    #     logging.error("Could not remove population from cells of type "
    #                   + str(cellType) + " in "
    #                   + WTP[str(country)][MAJ])
    #     logging.error(e)

    return populationProjected


# Spillover: remove population above limit and "push" them to the neighboring cells
# (I was thinking about calling this function "gentrify"...)
# Function allows spill over into other cell types, i.e. from urban to rural, to simulate urban growth
def spillover(populationProjected, country, limit, countryBoundaries, urbanRural, allIndexes, shape):


    a = countryBoundaries == int(country)
    b = populationProjected > limit
    c = urbanRural == urbanCell

    overcrowded = np.all((a, b, c), axis = 0)
    # for every overcrowded cell, distribute the surplus population randomly among its neighbors
    for fullCell in allIndexes[overcrowded]:

        logging.info("Spilling over "+str(fullCell))

        surplus = populationProjected[fullCell] - limit # by how much are we over the limit?
        # reset those cells to the limit value:
        populationProjected[fullCell] = limit
        # and move the extra people to the neighbors:
        wilsons = getNeighbours(fullCell, shape)
        rI = np.random.choice(wilsons, surplus)
        np.add.at(populationProjected, rI, 1)

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
