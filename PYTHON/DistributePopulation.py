import os, datetime, sys, operator, logging
import MASTER as m
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
    pop2000 = np.load(os.path.join(dir, "Data/NumpyLayers/Population2000.npy")).ravel()

    # and an array of all indexes; we'll use this later:
    allIndexes = np.arange(countryBoundaries.size)

    logging.info('Total German pop: '+str(np.sum(pop2000[countryBoundaries==276])))

    logging.info('Adding 500 k people to Germany')

    randomIndexes = np.random.choice(allIndexes[countryBoundaries==276], 500000)


    np.add.at(pop2000, randomIndexes, 1)

    logging.info('Total German pop: '+str(np.sum(pop2000[countryBoundaries==276])))

    #logging.info('Saving array.')

    #np.save(os.path.join(dir, "Data/NumpyLayers/germany500k", pop2000))

    # TODO: saving does not work yet
    # logging.info('Saving TIFF.')
    # transform back to 2D array with the original dimensions:
    # npgt.array_to_raster(pop2000.reshape(matrix),
    #                     os.path.join(dir, "Data/NumpyLayers/germany500k.tif"),
    #                     os.path.join(dir, "Data/NumpyLayers/Population2000.tif"))

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
