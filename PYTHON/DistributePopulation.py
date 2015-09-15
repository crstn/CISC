import os, datetime, sys, operator, logging
import MASTER as m
import numpy as np
import pandas as pd

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

    urbanRural = np.load(os.path.join(dir, "Data/NumpyLayers/glurextents-int.npy"))
    countryBoundaries = np.load(os.path.join(dir, "Data/NumpyLayers/gluntlbnds-clipped-int.npy"))
    pop2000 = np.load(os.path.join(dir, "Data/NumpyLayers/glup00ag-clipped.npy"))


    logging.info('Population array size: '+str(pop2000.shape))
    logging.info(str(np.unique(countryBoundaries)))
    logging.info(str(pop2000[countryBoundaries==276]))
    logging.info('Total German pop:      '+str(np.sum(pop2000[countryBoundaries==276])))

    logging.info('Adding 500 k people to Germany')

    #idx = np.random.choice(np.where((urbanRural==2) & (countryBoundaries==356))[0], 500000000)
    idx = np.random.choice(np.where(countryBoundaries==276)[0], 500000)
    np.add.at(pop2000, idx, 1)

    logging.info('Population array size: '+str(pop2000.shape))
    logging.info('Total German pop:      '+str(np.sum(pop2000[np.where(countryBoundaries==276)[0]])))

    logging.info('Saving array.')

    np.save(os.path.join(dir, "Data/NumpyLayers/india500mio", pop2000))

    logging.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        #filename='output.log',
                        #filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    try:
        main()
    except Exception, e:
        logging.exception(e)
