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

    logging.info('; CountryCode; Country; Raster total; CSV total; Raster urban; CSV urban; Raster rural; CSV rural; Error')

    filedir = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
    # this is the root folder of this project:
    dir = os.path.abspath(os.path.join(filedir, os.pardir))

    # logging.info("Reading reference GeoTIFF")
    # this is a GeoTIFF that we'll use as a reference for our output - same bbox, resolution, CRS, etc.
    referencetiff = os.path.join(dir, "Data/NumpyLayers/Population2010_clipped.tif")

    # logging.info('Reading CSVs')

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
    population = np.load(os.path.join(dir, "Data/NumpyLayers/Population2000.npy")).ravel()

    # replace no data values with 0:
    population[population == -9223372036854775808] = 0

    logging.info(urbanRural.shape)
    logging.info(countryBoundaries.shape)
    logging.info(population.shape)

    # and an array of all indexes; we'll use this later:
    allIndexes = np.arange(countryBoundaries.size)


    # logging.info("Growing population...")

    for run in range(RUNS):

        #logging.info( "Run no. " + str(run))

        # First eliminate discrepancy between the numbers for 2000
        # in our raster and what we have in the WUP/WTP spreadsheets for 2010
        for countryCode in WTP:
            if int(countryCode) < 900:

                country = WTP[countryCode]
                wupcountry = WUP[countryCode]

                # logging.info(countryCode + ": "+country["Major area, region, country or area"])

                pop2000raster = np.sum(population[countryBoundaries==int(countryCode)])

                if pop2000raster == 0:
                    logging.error("Country not found in raster file: " + countryCode + ": "+country["Major area, region, country or area"])
                else:
                    urb2000raster = np.sum(population[np.logical_and(countryBoundaries==int(countryCode), urbanRural==2)])
                    rur2000raster = pop2000raster-urb2000raster

                    pop2010csv = int(country["2010"])*1000
                    urb2010csv = int(wupcountry["2010"])*1000
                    rur2010csv = pop2010csv-urb2010csv

                    urbDiff = urb2010csv - urb2000raster
                    rurDiff = rur2010csv - rur2000raster

                    logging.info("; " + countryCode + "; "+country["Major area, region, country or area"]+"; "+str(pop2000raster)+"; "+str(pop2010csv)+"; "+str(urb2000raster)+"; "+str(urb2010csv)+"; "+str(rur2000raster)+"; "+str(rur2010csv)+"; no")

                    # logging.info("Total Pop 2000 raster: " + str(pop2000raster))
                    # logging.info("Urban Pop 2000 raster: " + str(urb2000raster))
                    # logging.info("Rural Pop 2000 raster: " + str(rur2000raster))
                    # logging.info("Total Pop 2010 WTP: " + str(pop2010csv))
                    # logging.info("Urban Pop 2010 WUP: " + str(urb2010csv))
                    # logging.info("Rural Pop 2010 WTP-WUP: " + str(rur2010csv))
                    # logging.info("Urban Difference: " + str(urbDiff))
                    # logging.info("Rural Difference: " + str(rurDiff))

                    # make sure none of these values try to go below 0:
                    if (urb2000raster + urbDiff) < 0:
                        # logging.info("Resetting urban difference to avoid going below 0.")
                        urbDiff = -1 * urb2000raster

                    if (rur2000raster + rurDiff) < 0:
                        # logging.info("Resetting rural difference to avoid going below 0.")
                        rurDiff = -1 * rur2000raster

                    changePopulation(int(countryCode), rurDiff, 1)
                    changePopulation(int(countryCode), urbDiff, 2)

        # logging.info('Saving GeoTIFF.')
        # transform back to 2D array with the original dimensions:
        array_to_raster(population.reshape(matrix), os.path.join(dir, "Data/NumpyLayers/Population-"+str(run)+"-2010.tif"))


    #
    #     year = 2015
    #     logging.info( str(year) )
    #     while year <= 2050:
    #
    #         # loop through countries:
    #
    #             # figure out how many people urban and rural to add/remove compared to 5 years ago:
    #             urbanChange =  (WUP.loc[country, str(year)]) - (WUP.loc[country, str(year-5)])*1000
    #
    #
    #         # save the array, reshaped back to its original 2D extents
    #         # logging.info('Saving array.')
    #         # os.chdir(os.path.join(dir, "Data/NumpyLayers"))
    #         # np.save("germany500k", population.reshape(matrix))
    #
    #         # Save to GeoTIFF
    #         logging.info('Saving GeoTIFF.')
    #         # transform back to 2D array with the original dimensions:
    #         array_to_raster(population.reshape(matrix),
    #                              os.path.join(dir, "Data/NumpyLayers/Population-"+str(run)+"-"+str(year)+".tif"))
    #
    # logging.info('Done.')




########################################################
# Some convenience functions
########################################################


# Turns a list of dictionaries into a single one:
def transposeDict(listOfDicts, pk):
    output = {}
    for dic in listOfDicts:
        output[dic[pk]] = dic
    return output


# country: int countryCode
# addPop: number of people to remove or add
# cellType: 1 for rural, 2 for urban
def changePopulation(country, pop, cellType):
    global population, allIndexes, countryBoundaries, urbanRural
    if pop >= 0:
        # selecting cells of correct type in country with people in them:
        try:
            randomIndexes = np.random.choice(allIndexes[np.logical_and(countryBoundaries==country, urbanRural==cellType)], pop)
            np.add.at(population, randomIndexes, 1)
        except Exception, e:
            logging.exception("Error processing country " + str(country))
            logging.exception(e)
    else:  # remove people, make sure we do not go below 0 in each cell!
        try:
            randomIndexes = np.random.choice(allIndexes[np.logical_and(countryBoundaries==country, np.logical_and(population>0, urbanRural==cellType))], math.fabs(pop))

            np.subtract.at(population, randomIndexes, 1)

            # add a little loop to add people back to the cells that have dropped below 0,
            # then randomly remove them somewhere else:
            while(population[population<0].size > 0):
                # logging.info('Cells below 0: ' +str(population[population<0].size))
                # logging.info('Number of people to add and remove somewhere else: ' + str(math.fabs(np.sum(population[population<0]))))
                # select random cells again, based on the number of people we need to remove again:
                randomIndexes = np.random.choice(allIndexes[np.logical_and(countryBoundaries==276, np.logical_and(population>0, urbanRural==cellType))], math.fabs(np.sum(population[population<0])))
                # set cells < 0 to 0
                population[population<0] = 0;
                # and then remove the people we have just added somewhere else:
                if randomIndexes.size > 0:
                    np.subtract.at(population, randomIndexes, 1)
                else:
                    logging.info("Tried to remove more people than possible; all cells of type "+cellType+" have already been set to 0.")
        except Exception, e:
            logging.exception("Error processing country " + str(country))
            logging.exception(e)





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
