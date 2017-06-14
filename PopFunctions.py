from osgeo import gdal, osr
import os
import datetime
import sys
import operator
import logging
import math
import csv
import numpy as np
from datetime import datetime

# some constants:
ruralCell = 1
urbanCell = 2
MAJ = 'Major area, region, country or area'

multiply = 1  # set to 1000 when running the DESA numbers!

# Factor to simulate densities in the highest density areas going down.
thinningFactor = 0.95

# number of top cells to take into account when calculating the average of the N cells with the highest population per country:
# TODO: Maybe this should be a % of all cells in a country, rather than a
# fixed number? E.g. 3%?
topNcells = 50


def dump_args(func):

    "This decorator dumps out the arguments passed to a function before calling it"

    argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
    fname = func.func_name

    def echo_func(*args, **kwargs):
        print fname, ":", ', '.join(
            '%s=%r' % entry
            for entry in zip(argnames, args) + kwargs.items())
        return func(*args, **kwargs)
    return echo_func


def logSubArraySizes(populationProjected, year, country, WTP, urbanRural):

    logging.info("  ----   ")
    logging.info("Array sizes for " + WTP[str(country)][MAJ])

    logging.info("Population: " +
                 str(populationProjected.size))
    logging.info("Urban: " + str(urbanRural[urbanRural == urbanCell].size))
    logging.info("Rural: " + str(urbanRural[urbanRural == ruralCell].size))
    logging.info("  ----   ")


# DESA population projection CSV
def logDifference(populationProjected, year, country, WTP, WUP, urbanRural):

    """Logs the difference for urban and rural population
    between whats in the populationProjected and the."""

    u = urbanRural == urbanCell
    r = urbanRural == ruralCell

    popraster = np.nansum(populationProjected)
    urbraster = np.nansum(populationProjected[u])
    rurraster = np.nansum(populationProjected[r])

    popcsv = getNumberForYear(WTP, year, country, multiply)
    urbcsv = getNumberForYear(WUP, year, country)
    rurcsv = (popcsv - urbcsv)

    popDiff = "{:,}".format(popcsv - popraster)
    urbDiff = "{:,}".format(urbcsv - urbraster)
    rurDiff = "{:,}".format(rurcsv - rurraster)

    c = WTP[str(country)][MAJ]
    logging.error("Total difference for " + c +
                  " in " + str(year) + ": " + popDiff)
    logging.error("Urban difference for " + c +
                  " in " + str(year) + ": " + urbDiff)
    logging.error("Rural difference for " + c +
                  " in " + str(year) + ": " + rurDiff)
    logging.error("")


# @dump_args
def getNumberForYear(table, year, country, multiply=1):

    """Looks up the population number for a country in a given year from a table.
    Multiplication factor can be used on tables that count in thousands, like the
    world total population (WTP). Defaults to 1 (no multiplication)."""

    # CSVs have weird number formatting with blanks (thanks for nothing, Excel),
    # that's why we need the "replace" bit
    return int(table[country][str(year)].replace(" ", "")) * multiply

# @dump_args
def getCountryByID(country, WTP):

    """Looks up the country name by UN ID."""

    try:
        return WTP[str(country)][MAJ]
    except KeyError:
        print "No country found for ID " + str(country)
        logging.error("No country found for ID " + str(country))
        return 0

# @dump_args
def getUrbanThreshold(country, population, urbanRural, WTP):

    """Calculates the population thresholds for turning a cell from rural to urban.
    Current approach: Urban threshold is the mean between the mean pop for urban cells
    and the mean pop for rural cells."""


    b = urbanRural == urbanCell

    # some countries don't have urban cells, they need special treatment:
    urban = population[b]
    if urban.size == 0:
        # set threshold to 1000 TODO: change?
        urbanMean = 1000
    else:
        urbanMean = np.nanmean(urban)

    return urbanMean


# @dump_args
def urbanize(populationProjected, year, country, WTP, WUP, urbanRural, urbanthreshold):

    """Turns rural into urban cells if national thresholds (see getUrbanThreshold) are exceeded."""

    # check rural cells in this country for population threshold:
    b = urbanRural == ruralCell
    c = populationProjected > urbanthreshold

    # turn these cells urban
    urbanRural[np.all((b, c), axis=0)] = urbanCell

    return urbanRural


# @dump_args
def adjustPopulation(populationProjected, year, country, WTP, WUP, urbanRural, rows, cols):

    """This one just compares the numbers from the raster to the CSV
    and then calls the corresponding functions to add or remove people."""

    # figure out the difference between the populationProjected
    # input raster and what's in the table:

    u = urbanRural == urbanCell
    r = urbanRural == ruralCell

    urbraster = np.nansum(populationProjected[u])
    rurraster = np.nansum(populationProjected[r])

    popcsv = getNumberForYear(WTP, year, country, multiply)
    urbcsv = getNumberForYear(WUP, year, country)

    rurcsv = (popcsv - urbcsv)

    urbDiff = urbcsv - urbraster
    rurDiff = rurcsv - rurraster

    logging.info("Numbers before adjustment:")
    logDifference(populationProjected, year, country, WTP, WUP, urbanRural)

    # urban:
    if (urbDiff > 0):  # add people
        logging.info("adding urban population")
        try:
            populationProjected = addPopulation(populationProjected, urbDiff,
                                                country, urbanCell, WTP, WUP, urbanRural, rows, cols)
        except ValueError as e:
            # TODO need a way to create new urban cells in countries that don't
            # have any
            logging.error(
                "ValueError while adding urban population -- no urban cells present")
            logging.error(e)

    else:   # remove people
        logging.info("removing urban population")
        populationProjected = removePopulation(populationProjected,
                                               np.abs(urbDiff), country,
                                               urbanCell, WTP, WUP, urbanRural, rows, cols)

    # and rural:
    if (rurDiff > 0):  # add people
        logging.info("adding rural population")
        populationProjected = addPopulation(populationProjected, rurDiff,
                                            country, ruralCell, WTP, WUP, urbanRural, rows, cols)
    else:   # remove people
        logging.info("removing rural population")
        populationProjected = removePopulation(populationProjected,
                                               np.abs(rurDiff), country,
                                               ruralCell, WTP, WUP, urbanRural, rows, cols)
    logging.info("Numbers after adjustment:")
    logDifference(populationProjected, year, country, WTP, WUP, urbanRural)

    return populationProjected


# @dump_args
def getTopNCells(N, populationProjected):

    """Returns the highest N values from an array"""

    p = np.partition(-populationProjected, N)
    pp = -p[:N]

    return pp


# @dump_args
def addPopulation(populationProjected, pop, country, cellType, WTP, WUP, urbanRural, rows, cols):

    """Randomly adds population to cells in a country with increasing population."""

    randoms = urbanRural == cellType
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
        randomIndexes = np.random.choice(np.where(randoms)[0], int(pop))

        np.add.at(populationProjected, randomIndexes, 1)

        if(cellType == urbanCell):

            logging.info("Checking if we need to spill over...")

            # before we start, we'll assume that a certain share of the current max is the
            # limit after which we spill over into neighboring cells:

            b = urbanRural == urbanCell
            urbCountry = populationProjected[b]

            if urbCountry.size > 0:
                if urbCountry.size >= topNcells:  # this is the common case
                    topN = getTopNCells(topNcells, urbCountry)
                # this catches countries which have urban cells, b ut very few
                # (not sure this is actually an issue, just in case)
                else:
                    topN = urbCountry
            else:  # if there are no urban cells, take any cells in country:
                urbCountry = populationProjected[a]

            # we'll use the mean of the top n cells of each country as the
            # maximum
            mx = np.nansum(topN) / topN.size
            # ... considering the thinning factor
            limit = mx * thinningFactor

            # logging.info("Limit for population per cell after thinning:")
            # logging.info(limit)

            # Print some stats after the spillover:

            # urb = urbanRural == urbanCell
            # rur = urbanRural == ruralCell

            # logging.info("Rural max:" + str(np.nanmax(populationProjected[rur])))
            # logging.info("Urban min:"  + str(np.nanmin(populationProjected[urb])))
            # logging.info("Urban max:"  + str(np.nanmax(populationProjected[urb])))

            # Repeat the spillover function as long as there are cells above the limit
            # TODO: this may run into an infinite loop!
            while (int(np.nanmax(populationProjected[b])) > int(limit)):

                # currentMax = np.nanmax(populationProjected[b])
                #
                # logging.info("Limit: " + str(limit))
                #
                # logging.info("Current max:" + str(currentMax))

                # logging.info("Are we over the limit? " + str(int(currentMax) > int(limit)))

                # c = populationProjected > limit
                #
                # logging.info("Cells over limit: " + str(populationProjected[np.all((b, c), axis=0)]))
                # logging.info("Indexes: " + str(np.where(np.all((b, c), axis=0))))

                populationProjected = spillover(
                    populationProjected, limit, urbanRural, rows, cols)

            # Print some stats after the spillover:

            # logging.info("Rural max:" + str(np.nanmax(populationProjected[rur])))
            # logging.info("Urban min:"  + str(np.nanmin(populationProjected[urb])))
            # logging.info("Urban max:"  + str(np.nanmax(populationProjected[urb])))

    return populationProjected


# @dump_args
def removePopulation(populationProjected, pop, country, cellType, WTP, WUP, urbanRural, rows, cols):

    """Removes population from a country where the population is declining."""

    # Added the condition that the cell has to have more than 0 population
    # Since we're doing subtract at with 1, this means we should create
    # fewer 'negative' cells...

    b = populationProjected >= 1.0
    c = urbanRural == cellType

    randoms = np.all((b, c), axis=0)

    if (np.where(randoms)[0].size > 0):
        randomIndexes = np.random.choice(np.where(randoms)[0], int(pop))

        logging.info("Removing 1 person from " +
                     str(len(randomIndexes)) + " cells.")

        np.subtract.at(populationProjected, randomIndexes, 1)

        # as long as we have cells of this type in this country that have a
        # negative population number, shuffle people around (i.e., add back in
        # to make the cell 0, then remove the same number somewhere else)
        while(populationProjected[np.all((c, populationProjected < 0), axis=0)].size > 0):
            # select random cells again, based on the number of people we need
            # to remove again:

            # a and c don't change (see above), but b does:
            b = populationProjected >= 1.0
            randoms = np.all((b, c), axis=0)

            belowZero = populationProjected < 0
            count = np.abs(np.nansum(populationProjected[belowZero]))

            # print " --- "
            # print populationProjected[np.all((a, c, belowZero), axis=0)]
            # print populationProjected[np.all((a, c, belowZero), axis=0)].size
            # print np.unique(randoms)
            # print count
            # print " "

            try:
                randomIndexes = np.random.choice(np.where(randoms), int(count))
                # set cells < 0 to 0
                populationProjected[belowZero] = 0.0
                # and then remove the people we have just added somewhere else:
                logging.info("Removing 1 person from " +
                             str(len(randomIndexes)) + " cells.")
                np.subtract.at(populationProjected, randomIndexes, 1)
                belowZero = populationProjected < 0
            except Exception as e:
                # TODO this is a dirty hack to get around a bug I couldn't fix.
                # Sometimes the script throws an error for some countries when there
                # is only one cell with a negative population number left.
                # The next iteration on this country should fix this, but it basicall means
                # that the total number is a bit off from the UN numbers for
                # this one year
                populationProjected[belowZero] = 0.0
                logging.error("Error skipped: ")
                logging.error(e)
    # if there are no cells matching the criteria, print an error msg:
    else:
        logging.error(" ")
        logging.error("ERROR: Cannot remove population from country " +
                      getCountryByID(country, WTP) + " (" + country + ")")
        logging.error("Cell type: " + str(cellType))
        logging.error(" ")

    return populationProjected


# borrowed from
# https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


# @dump_args
def spillover(populationProjected, limit, urbanRural, rows, cols):

    """Removes population above limit and "push" them to the neighboring cells
    (I was thinking about calling this function "gentrify"...)
    Function allows spill over into other cell types, i.e. from urban to rural, to simulate urban growth."""

    u = urbanRural == urbanCell
    l = populationProjected > limit

    overcrowded = np.all((u, l), axis=0)
    # for every overcrowded cell, distribute the surplus population randomly
    # among its neighbors

    cellsoverlimit = len(np.where(overcrowded)[0])
    total = np.nansum(populationProjected[overcrowded])
    totalLimit = limit * cellsoverlimit
    totalSurplus = total - totalLimit

    for fullCell in np.where(overcrowded)[0]:

        logging.info("Spilling over " + str(fullCell))

        # by how much are we over the limit?
        surplus = populationProjected[fullCell] - limit
        # reset those cells to the limit value:
        populationProjected[fullCell] = limit
        # find the indexes for the neighbors:
        neighborIndexes = getNeighbours(rows, cols, rows[fullCell], cols[
                                        fullCell], 3, populationProjected, limit)

        randomIndexes = np.random.choice(neighborIndexes, int(surplus))
        np.add.at(populationProjected, randomIndexes, 1)

    return populationProjected


# # @dump_args
def getNeighbours(rows, cols, row, col, n, populationProjected, limit):
    """Returns an array of indexes that correspond to the n x n neighborhood of the index cell
    at row/colum, while making sure that the return neighbors are listed in the rows/columns indexes.
    If n is an even number, will generate the neighborhood for n+1, so that the cell at
    row/col is always at the center.
    The indexes will also be only of cells that are below the population limit, so that we don't push
    people into neighboring cells that are already overcrowded themselves.
    If no matching neighbors are found, the function recursively calls itself with an enlarged neighborhood (n+2)
    until neighbors are found."""

    nbrows = []
    nbcols = []

    start = (n / 2) * -1
    end = (n / 2) + 1

    cart = cartesian((range(row + start, row + end),
                      range(col + start, col + end)))
    crows = cart[:, 0]
    ccols = cart[:, 1]

    # selecting all cells that are neighbors and below limit:
    isNBrow = np.in1d(rows, crows)
    isNBcol = np.in1d(cols, ccols)
    isUnderLimit = populationProjected < limit

    matches = np.where(np.all((isNBrow, isNBcol, isUnderLimit), axis=0))

    # if no matching cells found, enlarge neighborhood
    if(len(matches) == 0):
        print "No matching cells found for neighbors, extending neiborhood to " + str(n + 2)
        return getNeighbours(rows, cols, row, col, n + 2, populationProjected, limit)
    else:
        return matches




# @dump_args
def transposeDict(listOfDicts, pk):

    """Turns a list of dictionaries into a single one."""

    output = {}
    for dic in listOfDicts:
        output[dic[pk]] = dic
    return output



# @dump_args
def projectLinear(first, second):

    """Calculates the change for each cell between first and second,
    and applies that change to second. Generates an array that is
    essentially a linear continuation of the trend between first and second."""

    out = np.add(second, np.subtract(second, first))
    # replace all values below 0 with 0
    out[out < 0] = 0
    return out

# @dump_args


def openTIFFasNParray(file):
    src = gdal.Open(file, gdal.GA_Update)
    band = src.GetRasterBand(1)
    return np.array(band.ReadAsArray())


# @dump_args
def array_to_raster_noref(array, dst_filename, geotransform, rasterXSize, rasterYSize, projection):

    """Saves a raster as a geotiff to dst_filename.
    This version allows to pass the geotransform and x/y size directly,
    so we don't have to open the reference TIFF every time when repeatedly
    saving TIFFs."""

    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(
        dst_filename,
        rasterXSize,
        rasterYSize,
        1,
        gdal.GDT_Int32,
        options=['COMPRESS=LZW'])

    dataset.SetGeoTransform(geotransform)

    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).SetNoDataValue(-1)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.

# @dump_args
def array_to_raster(array, dst_filename, referencetiff):

    """Saves a raster as a geotiff to dst_filename.
       Projection, extent etc. are copied from referencetiff."""

    reffile = gdal.Open(referencetiff)

    geotransform = reffile.GetGeoTransform()

    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(
        dst_filename,
        reffile.RasterXSize,
        reffile.RasterYSize,
        1,
        gdal.GDT_Int32,
        options=['COMPRESS=LZW'])

    dataset.SetGeoTransform(geotransform)

    dataset.SetProjection(reffile.GetProjection())
    dataset.GetRasterBand(1).SetNoDataValue(-1)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.
