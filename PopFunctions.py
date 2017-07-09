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
    between whats in the populationProjected and the WTP/WUP."""

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
def urbanize(densities, urbanRural, country, year, WUP):

    """Turns rural into urban cells based on increase in urban population;
       e.g., if urban population inreases from 2010 to 2020 by 5%, the
       number of urban cells will also increase by 5%. In this case, the
       corresponding number of rural cells with the highest population density
       will turn urban."""

    oldurbanpop = float(getNumberForYear(WUP, year-10, country))
    newurbanpop = float(getNumberForYear(WUP, year, country))
    increase = newurbanpop/oldurbanpop

    # if there is no increase in urban population, we won't add new urban cells:
    if increase <= 1.0:
        return urbanRural
    else:
        # get the number of urban cells
        numberurbancells = np.where(urbanRural == urbanCell)[0].size

        newurbancells = int(numberurbancells * increase)
        toadd = newurbancells - numberurbancells

        # if "toadd" is 0 at this point, it means that this country has no urban
        # cells at all. We'll give it one, since there is urban population
        # (otherwise there would be no increase):

        if toadd == 0:
            toadd = 1

        # Check the code in "Sandbox/Urbanization demo.ipynb" for an explanation of what's happening here:
        # make copy of the densities array
        densCopy = np.copy(densities)
        # in that copy, set the density in all urban cells to zero:
        densCopy[urbanRural == urbanCell] = 0
        # this means they will get ignored when we pick the cells with the highest densities,
        # i.e., we will automatically get the RURAL cells with the highest densities:
        convert = getIndicesOfTopNCells(toadd, densCopy)
        # then we just turn those urban:
        urbanRural[convert] = urbanCell

        return urbanRural


# @dump_args
def adjustPopulation(populationProjected, year, country, WTP, WUP, urbanRural, rows, cols, areas):

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
                                                country, urbanCell, WTP, WUP, urbanRural, rows, cols, areas)
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
                                               urbanCell, WTP, WUP, urbanRural, rows, cols, areas)

    # and rural:
    if (rurDiff > 0):  # add people
        logging.info("adding rural population")
        populationProjected = addPopulation(populationProjected, rurDiff,
                                            country, ruralCell, WTP, WUP, urbanRural, rows, cols, areas)
    else:   # remove people
        logging.info("removing rural population")
        populationProjected = removePopulation(populationProjected,
                                               np.abs(rurDiff), country,
                                               ruralCell, WTP, WUP, urbanRural, rows, cols, areas)
    logging.info("Numbers after adjustment:")
    logDifference(populationProjected, year, country, WTP, WUP, urbanRural)

    return populationProjected


# @dump_args
def getTopNCells(N, arrrray):

    """Returns the highest N values from an array.
       Returns empty array if N=0."""

    p = np.partition(-arrrray, N)
    pp = -p[:N]

    return pp



def getIndicesOfTopNCells(N, arrrray):

    """Gets the indices of the highest N values in an array.
    Returns empty array if N=0."""

    if N == 0:
        return []
    else:
        return np.argpartition(arrrray, -N)[-N:]


# @dump_args
def addPopulation(populationProjected, pop, country, cellType, WTP, WUP, urbanRural, rows, cols, areas):

    """Randomly adds population to cells in a country with increasing population."""

    randoms = urbanRural == cellType
    if np.nansum(randoms) < 0:
        logging.error("Can't add population to "
                      + getCountryByID(country, WTP)
                      + ", country and " + str(cellType) + "conditions not"
                      + "satisfied?")
        return populationProjected

    # This just makes sure we actually have cells for the current country.
    # There are some regions in the dataset (e.g. South Asia) that don't
    # have corresponding cells in the raster dataset.
    if True in randoms:

        # add the missing number of population to the corresponding cell type (urban or rural):
        probabilities = areas[np.where(randoms)[0]]/(np.nansum(areas[np.where(randoms)[0]]))
        randomIndexes = np.random.choice(np.where(randoms)[0], int(pop), p=probabilities)
        np.add.at(populationProjected, randomIndexes, 1)

        # run the spillover routine if we are dealing with urban cells:
        if(cellType == urbanCell):

            # before we start, we'll assume that a certain share of the current max is the
            # limit after which we spill over into neighboring cells:

            densities = np.divide(populationProjected, areas)

            urbanDensities = densities[urbanRural == urbanCell]

            # Let's get the N densest cells (covering cases with few or no urban calls in a country):
            if urbanDensities.size > 0:
                if urbanDensities.size >= topNcells:  # this is the common case
                    topN = getTopNCells(topNcells, urbanDensities)
                # this catches countries which have urban cells, b ut very few
                # (not sure this is actually an issue, just in case)
                else:
                    topN = urbanDensities
            else:  # if there are no urban cells,
                # take any cells in the densities raster:
                topN = densities


            # we'll use the mean of the top n cells of each country as the limit
            mx = np.nansum(topN) / topN.size

            # ... considering the thinning factor
            limit = mx * thinningFactor

            # Repeat the spillover function as long as there are cells above the limit
            # TODO: this may run into an infinite loop!
            while (np.nanmax(np.divide(populationProjected, areas)) > limit):

                populationProjected = spillover(
                    populationProjected, areas, limit, urbanRural, rows, cols)


    return populationProjected


# @dump_args
def removePopulation(populationProjected, pop, country, cellType, WTP, WUP, urbanRural, rows, cols, areas):

    """Removes population from a country where the population is declining."""

    # Added the condition that the cell has to have more than 0 population
    # Since we're doing subtract at with 1, this means we should create
    # fewer 'negative' cells...

    b = populationProjected >= 1.0
    c = urbanRural == cellType

    randoms = np.all((b, c), axis=0)

    if (np.where(randoms)[0].size > 0):
        probabilities = areas[np.where(randoms)[0]]/(np.nansum(areas[np.where(randoms)[0]]))
        randomIndexes = np.random.choice(np.where(randoms)[0], int(pop), p=probabilities)

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

            try:
                probabilities = areas[np.where(randoms)]/(np.nansum(areas[np.where(randoms)]))
                randomIndexes = np.random.choice(np.where(randoms), int(count), p=probabilities)
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
def spillover(populationProjected, areas, limit, urbanRural, rows, cols):

    """Removes population above limit and "push" them to the neighboring cells
    (I was thinking about calling this function "gentrify"...)
    Function allows spill over into other cell types, i.e. from urban to rural, to simulate urban growth."""

    densities = np.divide(populationProjected, areas)

    u = urbanRural == urbanCell
    l = densities > limit

    overcrowded = np.all((u, l), axis=0)

    # for every overcrowded cell, distribute the surplus population randomly
    # among its neighbors
    for fullCell in np.where(overcrowded)[0]:

        logging.info("Spilling over " + str(fullCell))

        # calculate the maximum population this specific cell can hold
        # without going over the density limit:

        thisPopLimit = np.floor(areas[fullCell] * limit)

        # by how much are we over the limit?
        surplus = populationProjected[fullCell] - thisPopLimit
        # reset those cells to the limit value:
        populationProjected[fullCell] = thisPopLimit
        # find the indexes for the neighbors:
        neighborIndexes = getNeighbours(rows, cols, rows[fullCell], cols[
                                        fullCell], 3, densities, limit)

        probabilities = areas[neighborIndexes]/(np.nansum(areas[neighborIndexes]))
        randomIndexes = np.random.choice(neighborIndexes, int(surplus), p=probabilities)
        np.add.at(populationProjected, randomIndexes, 1)

    return populationProjected


# @dump_args
def getNeighbours(rows, cols, row, col, n, densities, limit):
    """Returns an array of indexes that correspond to the n x n neighborhood of the index cell
    at row/colum, while making sure that the return neighbors are listed in the rows/columns indexes.
    If n is an even number, will generate the neighborhood for n+1, so that the cell at
    row/col is always at the center.
    The indexes will also be only of cells that are below the population density limit, so that we
    don't push people into neighboring cells that are already overcrowded themselves.
    If no matching neighbors are found, the function recursively calls itself with an enlarged
    neighborhood (n+2) until neighbors are found."""

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
    isUnderLimit = densities < limit

    matches = np.where(np.all((isNBrow, isNBcol, isUnderLimit), axis=0))

    # if no matching cells found, enlarge neighborhood
    if(len(matches) == 0):
        logging.info("No matching cells found for neighbors, extending neiborhood to " + str(n + 2))
        return getNeighbours(rows, cols, row, col, n + 2, densities, limit)
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
def array_to_raster_noref(array, dst_filename, geotransform, rasterXSize, rasterYSize, projection, datatype=gdal.GDT_Int32):

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
        datatype,
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
