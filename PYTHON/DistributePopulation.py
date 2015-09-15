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

    logging.info('Adding one million people to India (356)')

    idx = np.random.choice(np.where(urbanRural==2 & countryBoundaries==356)[0], 1000000)
    np.add.at(pop2000, idx, 1)

    logging.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(filename='output.log',
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    try:
        main()
    except Exception, e:
        logging.exception(e)
