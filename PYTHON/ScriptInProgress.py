import os, datetime, sys
import MASTER
import numpy as np
import pandas as pd

print 'Starting at: '
print datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")

dir = os.path.dirname(__file__)
WUP = pd.read_csv(os.path.join(dir, "../Data/DESA/WUP2014Urban.csv"))
indexed_WUP = WUP.set_index("Country Code")

WTP = pd.read_csv(os.path.join(dir, "../Data/DESA/WTP2014.csv"))
indexed_WTP = WTP.set_index("Country Code")

path = os.path.join(dir, "../LIVE")
countryList = []
dirList = os.listdir(path+"/NumPy_GLUR")
for i in dirList:
    x = i[5:]
    y = x[:-4]
    countryList.append(y)

# print "List of country codes:"
# print countryList

GLURPath = path+"/NumPy_GLUR"
PopPath = path+"/NumPy_Pop"

numthreads = int(sys.argv[1])
thisthread = int(sys.argv[2])
modelruns  = int(sys.argv[3])

currentthread = 1

for i in countryList:
    if currentthread == thisthread:
#        print i
        glur_array = np.load(GLURPath+"/GLUR_"+i+".npy")
        pop_array = np.load(PopPath+"/Pop00_"+i+".npy")

        urban_cell_list = []
        rural_cell_list = []

#        print "Creating lists of urban and rural cells."
        it = np.nditer(glur_array, flags=['multi_index'])
        while not it.finished:
            for x in it:
                if x == 2:
                    urban_cell_list.append(it.multi_index)
                elif x == 1:
                    rural_cell_list.append(it.multi_index)
            it.iternext()

#        print "Lists of urban and rural cells compiled at:"
#        print datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
#        print "There are ", len(urban_cell_list), "urban cells in the array."
#        print
#        print "There are ", len(rural_cell_list), "rural cells in the array."

    MASTER.main(path, i, urban_cell_list, rural_cell_list, pop_array, modelruns)

    currentthread = currentthread + 1
    if currentthread > numthreads:
        currentthread = 1

print 'Done at:'
print datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
