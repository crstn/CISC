
import numpy as np

version = np.version.version
print version

# values for analysis
country = "usa"
ruralCell = 1
urbanCell = 2
# current population
populationProjected = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 10] 
# population to add
pop = 1000 
# urban -rural classification
urbanRural = [1, 1, 1, 1, 1, 2, 2, 2, 1, 1] 
# country classifcation
countryList = ["usa", "gbk", "ind", "ind", "usa", "chn", "ice", "arg", "usa", "usa"] 

# make a list of all indices for arrays
countryBoundaries = np.array(countryList)
allIndexes = np.arange(countryBoundaries.size)

# turn all important lists into arrays for numpy analyses
arr_1 = np.array(countryList)
arr_2 = np.array(urbanRural)
arr_4 = np.array(populationProjected)
arr_5 = np.array(allIndexes)

# Carsten's function
def removePopulation(populationProjected, pop, country, cellType):
    try:
        randoms = np.all((arr_1 == country, arr_2 == cellType, populationProjected > 0),axis = 0)

        randomIndexes = np.random.choice(arr_5[randoms], pop)
        np.subtract.at(populationProjected, randomIndexes, 1) # What happens if you take out way too many?

        while(populationProjected[populationProjected < 0].size >0):
            a = arr_1 == country
            b = populationProjected > 0
            c = arr_2 == cellType

            randoms = np.all((a,b,c), axis = 0)

            less = populationProjected < 0
            count = np.abs(np.sum(populationProjected[less]))

            randomIndexes = np.random.choice(arr_5[randoms], count)
            populationProjected[less] = 0

            if randomIndexes.size > 0:
                np.subtract.at(populationProjected, randomIndexes, 1)
            else:
                print "Tried to remove more people than possible \n"
    except Exception, e:
        print "Could not remove population"

    return populationProjected 



print "Original list: ", arr_4
print "The size of the original list is: ", np.sum(arr_4)
print
print "We remove %d from the original list." % (pop)
print
y = removePopulation(arr_4, pop, country, ruralCell)
print "Final list: ", y
print "The size of the final population is: ", np.sum(arr_4)
#
#print


