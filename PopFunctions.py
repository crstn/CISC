from osgeo import gdal, osr
import os
import datetime
import sys
import operator
import logging
import math
import csv
import bottleneck
import numpy as np
from datetime import datetime

# some constants:
ruralCell = 1
suburbanCell = 2
urbanCell = 3
MAJ = 'Major area, region, country or area'

# Factor to simulate densities in the highest density areas going down.
# TODO: 0.8 might be a bit too low, i.e. cause too much thinning. Play
# around with different values.
thinningFactor = 0.95

# number of top cells to take into account when calculating the average of the N cells with the highest population per country:
# TODO: Maybe this should be a % of all cells in a country, rather than a
# fixed number? E.g. 3%?
topNcells = 50


def logSubArraySizes(populationProjected, year, country, WTP, countryBoundaries, urbanRural):

    logging.info("  ----   ")
    logging.info("Array sizes for " + WTP[str(country)][MAJ])

    logging.info("Population: " +
                 str(populationProjected[countryBoundaries == int(country)].size))
    logging.info("Urban: " + str(urbanRural[np.logical_and(
        countryBoundaries == int(country), urbanRural == urbanCell)].size))
    logging.info("Rural: " + str(urbanRural[np.logical_and(
        countryBoundaries == int(country), urbanRural == ruralCell)].size))
    logging.info("  ----   ")


# logs the difference for urban and rural population
# between whats in the populationProjected and the
# DESA population projection CSV
def logDifference(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural):
    incountry = countryBoundaries == int(country)
    u         = urbanRural == urbanCell
    r         = urbanRural == ruralCell

    urbraster=np.nansum(populationProjected[
        np.logical_and(incountry, u)])
    rurraster = np.nansum(populationProjected[
        np.logical_and(incountry, r)])

    popcsv = getNumberForYear(WTP, year, country, 1000)
    urbcsv = getNumberForYear(WUP, year, country)
    rurcsv = (popcsv - urbcsv)

    urbDiff = urbcsv - urbraster
    rurDiff = rurcsv - rurraster

    c = WTP[str(country)][MAJ]
    logging.info("Urban difference for " + c + " in " + str(year) + ": " + str(urbDiff))
    logging.info("Rural difference for " + c + " in " + str(year) + ": " + str(rurDiff))


# Looks up the population number for a country in a given year from a table.
# Multiplication factor can be used on tables that count in thousands, like the
# world total population (WTP). Defaults to 1 (no multiplication).
def getNumberForYear(table, year, country, multiply=1):
    # CSVs have weird number formatting with blanks (thanks for nothing, Excel),
    # that's why we need the "replace" bit
    return int(table[country][str(year)].replace(" ", "")) * multiply

# Looks up the country name by UN ID
def getCountryByID(country, WTP):
    try:
        return WTP[str(country)][MAJ]
    except KeyError:
        print "No country found for ID " + str(country)
        logging.error("No country found for ID " + str(country))
        return 0

# Calculates the population thresholds for turning a cell from rural to suburban and from suburban to urban, respectively.
# Current approach: Urban threshold is the mean between the mean pop for urban cells
# and the mean pop for suburban cells. Suburban threshold is the suburban median.
def getThresholds(country, population, countryBoundaries, urbanRural, WTP):
    a = countryBoundaries == int(country)
    b = urbanRural == urbanCell

    # some countries don't have urban cells, they need special treatment:
    urban = population[np.all((a, b), axis=0)]
    if urban.size == 0:
        # set threshold to 1000 TODO: change?
        urbanMedian = 1000
    else:
        urbanMedian = np.nanmedian(urban)

    b = urbanRural == suburbanCell

    suburbanMedian = np.nanmedian(population[np.all((a, b), axis=0)])

    urbanthreshold = (urbanMedian + suburbanMedian) / 2
    suburbanthreshold = suburbanMedian

    return urbanthreshold, suburbanthreshold


# turns suburban into urban cells (and rural into suburban) if national thresholds (see getThreshold) are exceeded
def urbanize(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural, allIndexes, shape, urbanthreshold, suburbanthreshold):

    a = countryBoundaries == int(country)
    b = urbanRural == urbanCell

    # print " "
    # print "No. urban cells before urbanization in " +str(year)+ ": " +
    # str(urbanRural[urbanRural == urbanCell].size)

    # topN = getTopNCells(topNcells, populationProjected[np.all((a, b), axis=0)])
    # we'll use the mean of the top n URBAN cells of each country as the
    # threshold
    # mx = np.nansum(topN) / topNcells

    # limit = getThreshold(country, populationProjected,
    #                      countryBoundaries, urbanRural, WTP)

    # check suburban cells in this country for population threshold:
    b = urbanRural == suburbanCell
    c = populationProjected > urbanthreshold

    # turn these cells urban
    urbanRural[np.all((a, b, c), axis = 0)] = urbanCell

    #repeat for turning rural cells suburban:
    b = urbanRural == ruralCell
    c = populationProjected > suburbanthreshold

    # turn these cells suburban
    urbanRural[np.all((a, b, c), axis = 0)] = suburbanCell

    # for every matching cell, check whether at least 3 neighbors are already
    # urban:
    # for cell in allIndexes[np.all((a, b, c), axis = 0)]:

        # wilsons=getNeighbours(cell, shape, 3)
        #
        # # if so, turn urban
        # # TODO this is pretty inefficient
        # urbanNeighbors=0
        # for w in wilsons:
        #     if urbanRural[w] == urbanCell:
        #         urbanNeighbors=urbanNeighbors + 1
        #
        # if(urbanNeighbors >= 3):
        # urbanRural[cell]=urbanCell

    # print "No. urban cells after urbanization: " + str(urbanRural[urbanRural
    # == urbanCell].size)

    return urbanRural



# this one just compares the numbers from the raster to the CSV
# and then calls the corresponding functions to add or remove people.
def adjustPopulation(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural, allIndexes, shape):

    # figure out the difference between the populationProjected
    # input raster and what's in the table:

    urbraster=np.nansum(populationProjected[
        np.logical_and(countryBoundaries == int(country),
                       urbanRural == urbanCell)])
    rurraster=np.nansum(populationProjected[
        np.logical_and(countryBoundaries == int(country),
                       urbanRural == ruralCell)])

    popcsv=getNumberForYear(WTP, year, country, 1000)
    urbcsv=getNumberForYear(WUP, year, country)
    rurcsv=(popcsv - urbcsv)

    urbDiff=urbcsv - urbraster
    rurDiff=rurcsv - rurraster

    logging.info("Numbers before adjustment:")
    logDifference(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural)

    # urban:
    if (urbDiff > 0):  # add people
        logging.info("adding urban population")
        try:
            populationProjected=addPopulation(populationProjected, urbDiff,
                                                country, urbanCell, WTP, WUP, countryBoundaries, urbanRural, allIndexes, shape)
        except ValueError as e:
            # TODO need a way to create new urban cells in countries that don't have any
            logging.error("ValueError while adding urban population -- no urban cells present")
            logging.error(e)

    else:   # remove people
        logging.info("removing urban population")
        populationProjected=removePopulation(populationProjected,
                                               np.abs(urbDiff), country,
                                               urbanCell, WTP, WUP, countryBoundaries, urbanRural, allIndexes)

    # and rural:
    if (rurDiff > 0):  # add people
        logging.info("adding rural population")
        populationProjected=addPopulation(populationProjected, rurDiff,
                                            country, ruralCell, WTP, WUP, countryBoundaries, urbanRural, allIndexes, shape)
    else:   # remove people
        logging.info("removing rural population")
        populationProjected=removePopulation(populationProjected,
                                               np.abs(rurDiff), country,
                                               ruralCell, WTP, WUP, countryBoundaries, urbanRural, allIndexes)
    logging.info("Numbers after adjustment:")
    logDifference(populationProjected, year, country, WTP, WUP, countryBoundaries, urbanRural)

    return populationProjected


# returns the values of the top N cells
def getTopNCells(N, populationProjected):
    return bottleneck.partsort(populationProjected, populationProjected.size -N)[-N:]




def addPopulation(populationProjected, pop, country, cellType, WTP, WUP, countryBoundaries, urbanRural, allIndexes, shape):

    # try:
    a= countryBoundaries == int(country)
    b= urbanRural == cellType

    randoms = np.all((a, b), axis =0)
    if np.nansum(randoms) < 0:
        logging.error("Can't add population to "
                      + getCountryByID(country, WTP)
                      + ", country and " + str(cellType) + "conditions not"
                      + "satisfied?")
        logging.error(a)
        logging.error(b)
        return populationProjected

    # This just makes sure we actually have cells for the current country.
    # There are some regions in the dataset (e.g. South Asia) that don't
    # have corresponding cells in the raster dataset.
    if True in randoms:
        # randomIndexes = np.random.choice(allIndexes[a], pop)
        randomIndexes = np.random.choice(allIndexes[randoms], int(pop))

        np.add.at(populationProjected, randomIndexes, 1)


        if(cellType == urbanCell):

            logging.info("Checking if we need to spill over...")

            # before we start, we'll assume that a certain share of the current max is the
            # limit after which we spill over into neighboring cells:

            a = countryBoundaries == int(country)
            b = urbanRural == urbanCell

            urbCountry = populationProjected[np.all((a, b), axis=0)]


            if urbCountry.size > 0:
                if urbCountry.size >= topNcells: # this is the common case
                    topN = getTopNCells(topNcells, urbCountry)
                else: # this catches countries which have urban cells, b ut very few (not sure this is actually an issue, just in case)
                    topN = urbCountry
            else: # if there are no urban cells, take any cells in country:
                urbCountry = populationProjected[a]



            # we'll use the mean of the top n cells of each country as the maximum
            mx = np.nansum(topN) / topN.size
            # ... considering the thinning factor
            limit = mx * thinningFactor

            logging.info("Limit for population per cell after thinning:")
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

    return populationProjected



def removePopulation(populationProjected, pop, country, cellType, WTP, WUP, countryBoundaries, urbanRural, allIndexes):

    # Added the condition that the cell has to have more than 0 population
    # Since we're doing subtract at with 1, this means we should create
    # fewer 'negative' cells...

    a = countryBoundaries == float(country)
    b = populationProjected >= 1.0
    c = urbanRural == cellType
    randoms = np.all((a, b, c), axis=0)

    if (allIndexes[randoms].size > 0):
        randomIndexes = np.random.choice(allIndexes[randoms], int(pop))

        logging.info("Removing 1 person from " + str(len(randomIndexes)) + " cells.")

        np.subtract.at(populationProjected, randomIndexes, 1)


        # as long as we have cells of this type in this country that have a negative population number, shuffle people around (i.e., add back in to make the cell 0, then remove the same number somewhere else)
        while(populationProjected[np.all((a, c, populationProjected < 0), axis=0)].size > 0):
            # select random cells again, based on the number of people we need to remove again:

            # a and c don't change (see above), but b does:
            b = populationProjected >= 1.0
            randoms = np.all((a, b, c), axis=0)

            belowZero = populationProjected < 0
            count = np.abs(np.nansum(populationProjected[belowZero]))

            # print " --- "
            # print populationProjected[np.all((a, c, belowZero), axis=0)]
            # print populationProjected[np.all((a, c, belowZero), axis=0)].size
            # print np.unique(randoms)
            # print count
            # print " "

            try:
                randomIndexes = np.random.choice(allIndexes[randoms], int(count))
                # set cells < 0 to 0
                populationProjected[belowZero] = 0.0
                # and then remove the people we have just added somewhere else:
                logging.info("Removing 1 person from " + str(len(randomIndexes)) + " cells.")
                np.subtract.at(populationProjected, randomIndexes, 1)
                belowZero = populationProjected < 0
            except Exception as e:
                # TODO this is a dirty hack to get around a bug I couldn't fix.
                # Sometimes the script throws an error for some countries when there
                # is only one cell with a negative population number left.
                # The next iteration on this country should fix this, but it basicall means
                # that the total number is a bit off from the UN numbers for this one year
                populationProjected[belowZero] = 0.0
                logging.error("Error skipped: ")
                logging.error(e)
    # if there are no cells matching the criteria, print an error msg:
    else:
        logging.error( " " )
        logging.error( "ERROR: Cannot remove population from country " + getCountryByID(country, WTP) + " ("+country+")")
        logging.error( "Cell type: " + str(cellType) )
        logging.error( " " )




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
        wilsons = getNeighbours(fullCell, shape, 3)
        rI = np.random.choice(wilsons, int(surplus))
        np.add.at(populationProjected, rI, 1)

    return populationProjected



# Returns an array of indexes that correspond to the n x n neighborhood of the index cell
# in a raveled (1D) matrix based on the # shape of the original (2D) matrix.
# Returns only neighbors within shape, exlcuding the input cell
# If n is an even number, will generate the neighborhood for n+1, so that the cell at
# index is always at the center
def getNeighbours(index, shape, n):
    twoDIndex = oneDtoTwoD(index, shape)
    row = twoDIndex[0]
    col = twoDIndex[1]

    neighbors = []

    start = (n/2) * -1
    end = (n/2) + 1

    for r in range(start, end):
        for c in range(start, end):
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

def openTIFFasNParray(file):
    src = gdal.Open(file, gdal.GA_Update)
    band = src.GetRasterBand(1)
    return np.array(band.ReadAsArray())

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
        gdal.GDT_Int32,
        options = [ 'COMPRESS=LZW' ])

    dataset.SetGeoTransform(geotransform)

    dataset.SetProjection(reffile.GetProjection())
    dataset.GetRasterBand(1).SetNoDataValue(-1)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.
