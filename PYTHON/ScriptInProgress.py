import os, datetime, sys, operator, logging
import MASTER as m
import numpy as np
import pandas as pd

def main():
    logging.info('Starting...')

    filedir = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
    # this is the root folder of this project:
    dir = os.path.abspath(os.path.join(filedir, os.pardir))

    WUP = pd.read_csv(os.path.join(dir, "Data/DESA/WUP2014Urban.csv"))
    indexed_WUP = WUP.set_index("Country Code")

    WTP = pd.read_csv(os.path.join(dir, "Data/DESA/WTP2014.csv"))
    indexed_WTP = WTP.set_index("Country Code")

    path = os.path.join(dir, "LIVE")

    GLURPath = path+"/NumPy_GLUR"
    PopPath = path+"/NumPy_Pop"


    countryList = []

    # if we are profiling, only grab the country code from the command line argument, this will be the only country in our list:
    if len(sys.argv) == 5:

        logging.info("Running only country " + sys.argv[4])
        countryList.append(sys.argv[4])

    # else we are doing a full run, so check the directory:
    else :

        logging.info ("Running ALL countries in " + GLURPath)
        # let's get all GLUR files, ordered by size. This will make sure that the large files are evenly distributed across
        # multiple processes, and we don't accidentally put all large countries in one process
        dirListUnsorted = [ [files, os.path.getsize(path+"/NumPy_GLUR/"+files)] for files in os.listdir(path+"/NumPy_GLUR") ]
        dirList = sorted(dirListUnsorted, key=operator.itemgetter(1))

        for fileItem in dirList:
            i = fileItem[0] # fetch file name
            x = i[5:]       # remove prefix ...
            y = x[:-4]      # ... and extension
            countryList.append(y)

    # logging.info("List of country codes:"
    # logging.info(countryList)

    numthreads = int(sys.argv[1])
    thisthread = int(sys.argv[2])
    modelruns  = int(sys.argv[3])

    currentthread = 1

    for i in countryList:
        if currentthread == thisthread:

            glur_array = np.load(GLURPath+"/GLUR_"+i+".npy")
            pop_array = np.load(PopPath+"/Pop00_"+i+".npy")

            urban_cell_list = []
            rural_cell_list = []

    #        logging.info("Creating lists of urban and rural cells."
            it = np.nditer(glur_array, flags=['multi_index'])
            while not it.finished:
                for x in it:
                    if x == 2:
                        urban_cell_list.append(it.multi_index)
                    elif x == 1:
                        rural_cell_list.append(it.multi_index)
                it.iternext()

    #        logging.info("Lists of urban and rural cells compiled.")
    #        logging.info("There are " + str(len(urban_cell_list)) + "urban cells in the array."
    #
    #        logging.info("There are " + str(len(rural_cell_list)) + "rural cells in the array."

            m.main(path, i, urban_cell_list, rural_cell_list, pop_array, indexed_WUP, indexed_WTP, modelruns)


        currentthread = currentthread + 1
        if currentthread > numthreads:
            currentthread = 1

    logging.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(filename='output.log',
                        filemode='w',
                        level=logging.ERROR,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    try:
        main()
    except Exception, e:
        logging.exception(e)