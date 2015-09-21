import os, datetime, sys, operator, logging, math
import numpy as np
import pandas as pd
import numpy2geotiff as npgt
import random

def main():
    logging.info('Starting...')

    filedir = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
    # this is the root folder of this project:
    dir = os.path.abspath(os.path.join(filedir, os.pardir))

    logging.info('Reading CSVs')

    # in this dataset: 1=rural, 2=urban
    WUP = pd.read_csv(os.path.join(dir, "Data/DESA/WUP2014Urban.csv"))
    indexed_WUP = WUP.set_index("Country Code")

    WTP = pd.read_csv(os.path.join(dir, "Data/DESA/WTP2014.csv"))
    indexed_WTP = WTP.set_index("Country Code")

    logging.info('Reading Numpy arrays')

    urbanRural = np.load(os.path.join(dir, "Data/NumpyLayers/UrbanRural.npy"))

    # save the shape of these arrays for later, so that we
    # can properly reshape them after flattening:
    matrix = urbanRural.shape

    # we flatten all arrays to 1D, so we don't have to deal with 2D arrays:
    urbanRural = urbanRural.ravel()
    countryBoundaries = np.load(os.path.join(dir, "Data/NumpyLayers/NationOutlines.npy")).ravel()
    population = np.load(os.path.join(dir, "Data/NumpyLayers/Population2000.npy")).ravel()

    # and an array of all indexes; we'll use this later:
    allIndexes = np.arange(countryBoundaries.size)

    logging.info('Total German pop: '+str(np.sum(population[countryBoundaries==276])))
    logging.info('Rural German pop: '+str(np.sum(population[np.logical_and(countryBoundaries==276, urbanRural==1)])))

    logging.info('Removing 500 k people from rural Germany')

    # selecting rural cells in Germany with people in them:
    randomIndexes = np.random.choice(allIndexes[np.logical_and(countryBoundaries==276, np.logical_and(population>0, urbanRural==1))], 5000000)

    np.subtract.at(population, randomIndexes, 1)

    logging.info('Total German pop: '+str(np.sum(population[countryBoundaries==276])))
    logging.info('Rural German pop: '+str(np.sum(population[np.logical_and(countryBoundaries==276, urbanRural==1)])))

    # add a little loop to add people back to the cells that have dropped below 0,
    # the randomly remove them somewhere else:
    while(population[population<0].size > 0):
        logging.info('Cells below 0: ' +str(population[population<0].size))
        logging.info('Number of people to add and remove somewhere else: ' + str(math.fabs(np.sum(population[population<0]))))
        # select random cells again, based on the number of people we need to remove again:
        randomIndexes = np.random.choice(allIndexes[np.logical_and(countryBoundaries==276, np.logical_and(population>0, urbanRural==1))], math.fabs(np.sum(population[population<0])))
        # set cells < 0 to 0
        population[population<0] = 0;
        np.subtract.at(population, randomIndexes, 1)

    logging.info('Total German pop: '+str(np.sum(population[countryBoundaries==276])))
    logging.info('Rural German pop: '+str(np.sum(population[np.logical_and(countryBoundaries==276, urbanRural==1)])))


    # save the array, reshaped back its original 2D extents
    # logging.info('Saving array.')
    # os.chdir(os.path.join(dir, "Data/NumpyLayers"))
    # np.save("germany500k", population.reshape(matrix))

    # TODO: saving to TIFF does not work yet
    logging.info('Saving TIFF.')
    # transform back to 2D array with the original dimensions:
    npgt.array_to_raster(population.reshape(matrix),
                         os.path.join(dir, "Data/NumpyLayers/germany500k.tif"),
                         os.path.join(dir, "Data/NumpyLayers/Population2000.tif"))

    logging.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        filename='output.log',
                        filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    try:
        main()
    except Exception, e:
        logging.exception(e)
