from osgeo import gdal, osr
import os, datetime, sys, operator, logging, math, csv
import numpy as np
import numpy2geotiff as npgt
from datetime import datetime


# how many times do we want to simulate?
RUNS = 1
# this is a GeoTIFF that we'll use as a reference for our output - same bbox, resolution, CRS, etc.
referencetiff = ""
populationOld = []
populationNew = []
allIndexes = []
countryBoundaries = []
urbanRural = []

def main():
    global populationOld, populationNew, allIndexes, countryBoundaries, urbanRural, referencetiff

    logging.info('Starting...')

    logging.info("Reading reference GeoTIFF")
    # this is a GeoTIFF that we'll use as a reference for our output - same bbox, resolution, CRS, etc.
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
    # logging.info(np.nanmin(populationOld))
    # logging.info(np.nanmax(populationOld))
    # logging.info(np.nanmin(populationNew))
    # logging.info(np.nanmax(populationNew))

    # let's save the 2000 and 2010 tiffs:
    array_to_raster(populationOld.reshape(matrix),
                             os.path.expanduser('~') + "/Dropbox/CISC - Global Population/Asia/Projections/Population-0-2000.tif")
    array_to_raster(populationNew.reshape(matrix),
                             os.path.expanduser('~') + "/Dropbox/CISC - Global Population/Asia/Projections/Population-0-2010.tif")

    # compare numbers between TIFFS and CSVs:

    # logging.info("CountryCode, CountryName, DESA2000, TIFF2000, DESA2010, TIFF2010, DESA2020, TIFF2020, DESA2030, TIFF2030, DESA2040, TIFF2040, DESA2050, TIFF2050")
    #
    # for country in WTP:
    #     tiff2k   = np.sum(populationOld[countryBoundaries==int(country)])
    #     tiff2k10 = np.sum(populationNew[countryBoundaries==int(country)])
    #     logging.info(str(country) + ", " + WTP[str(country)]['Major area, region, country or area'] + ", " + str(int(WTP[str(country)]['2000'])*1000) + ", " + str(int(tiff2k)) + ", " + str(int(WTP[str(country)]['2010'])*1000) + ", " + str(int(tiff2k10)) )

    # check which countries we have:
    # AsianCountries = np.unique(countryBoundaries)
    # for c in AsianCountries:
    #     try:
    #         logging.info(WTP[str(c)]['Major area, region, country or area'])
    #     except Exception:
    #         logging.info("Country for ID "+str(c)+ " not found.")

    # make an array of all indexes; we'll use this later:
    allIndexes = np.arange(countryBoundaries.size)

    logging.info("Growing population...")

    for run in range(RUNS):

        logging.info( "Run no. " + str(run))

        year = 2020
        step = 10
        while year <= 2050:

            populationProjected = projectLinear(populationOld, populationNew)

            # loop through countries:
            # for country in WTP:
            #
            #     wtpcountry = WTP[country]
            #     wupcountry = WUP[country]
            #     # figure out the difference between our linear projection and what's in the table:
            #
            #     popraster = np.sum(populationProjected[countryBoundaries==int(country)])
            #     urbraster = np.sum(populationProjected[np.logical_and(countryBoundaries==int(country), urbanRural==2)])
            #     rurraster = popraster-urbraster
            #
            #     popcsv = int(wtpcountry[str(year)])*1000
            #     urbcsv = int(wupcountry[str(year)])*1000
            #     rurcsv = popcsv-urbcsv
            #
            #     urbDiff = urbcsv - urbraster
            #     rurDiff = rurcsv - rurraster
            #
            #     populationOld       = populationNew
            #     populationProjected = changePopulation(populationProjected, country, urbDiff, 2)
            #     populationNew       = changePopulation(populationProjected, country, rurDiff, 1)
            #
            #     # Save to GeoTIFF
            #     logging.info('Saving GeoTIFF.')
            #     # transform back to 2D array with the original dimensions:

            # prepare everything for the next iteration

            populationOld = populationNew
            populationNew = populationProjected

            # logging.info(year)
            # logging.info(np.nanmin(populationNew))
            # logging.info(np.nanmax(populationNew))


            # compare numbers in raster to DESA projections:
            # for country in WTP:
            #     tiffval   = np.sum(populationNew[countryBoundaries==int(country)])
            #     logging.info(str(int(WTP[str(country)][str(year)])*1000) + ", " + str(int(tiffval)))
            #


            # save tiff
            array_to_raster(populationNew.reshape(matrix),
                                     os.path.expanduser('~') + "/Dropbox/CISC - Global Population/Asia/Projections/Population-"+str(run)+"-"+str(year)+".tif")
            year = year + step

    logging.info('Done.')




########################################################
# Some convenience functions
########################################################

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


# country: int countryCode
# addPop: number of people to remove or add
# cellType: 1 for rural, 2 for urban
def changePopulation(populationProjected, country, pop, cellType):
    global allIndexes, countryBoundaries, urbanRural
    if pop >= 0:
        # selecting cells of correct type in country with people in them:
        try:
            randomIndexes = np.random.choice(allIndexes[np.logical_and(countryBoundaries==country, urbanRural==cellType)], pop)
            np.add.at(populationProjected, randomIndexes, 1)
        except Exception, e:
            logging.exception("Error processing country " + str(country))
            logging.exception(e)
    else:  # remove people, make sure we do not go below 0 in each cell!
        try:
            randomIndexes = np.random.choice(allIndexes[np.logical_and(countryBoundaries==country, np.logical_and(populationProjected>0, urbanRural==cellType))], math.fabs(pop))

            np.subtract.at(populationProjected, randomIndexes, 1)

            # add a little loop to add people back to the cells that have dropped below 0,
            # then randomly remove them somewhere else:
            while(populationProjected[populationProjected<0].size > 0):
                # logging.info('Cells below 0: ' +str(population[population<0].size))
                # logging.info('Number of people to add and remove somewhere else: ' + str(math.fabs(np.sum(population[population<0]))))
                # select random cells again, based on the number of people we need to remove again:
                randomIndexes = np.random.choice(allIndexes[np.logical_and(countryBoundaries==country, np.logical_and(populationProjected>0, urbanRural==cellType))], math.fabs(np.sum(populationProjected[populationProjected<0])))
                # set cells < 0 to 0
                populationProjected[populationProjected<0] = 0;
                # and then remove the people we have just added somewhere else:
                if randomIndexes.size > 0:
                    np.subtract.at(populationProjected, randomIndexes, 1)
                else:
                    logging.info("Tried to remove more people than possible; all cells of type "+cellType+" have already been set to 0.")
        except Exception, e:
            logging.exception("Error processing country " + str(country))
            logging.exception(e)

    return populationProjected


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
                        format='%(asctime)s %(levelname)-8s %(message)s')
    try:
        main()
    except Exception, e:
        logging.exception(e)
