import numpy as np

# values for analysis
country = "usa"
ruralCell = 1
urbanCell = 2
# current population
populationProjected = [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10],[10, 10, 10, 10, 10, 10, 10, 10, 10, 10],[10, 10, 10, 10, 10, 10, 10, 10, 10, 10],[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]

# population to add
pop = 1500

# max. population per cell before we start spillover
limitPop = 100

# urban -rural classification
urbanRural = [[2, 1, 1, 1, 2, 2, 2, 2, 2, 2], [2, 1, 2, 2, 2, 1, 1, 1, 1, 2], [2, 1, 2, 1, 2, 1, 1, 1, 2, 2], [1, 1, 2, 2, 2, 2, 1, 1, 1, 1]]

# country classifcation
countryList = [["usa", "gbk", "ind", "ind", "usa", "chn", "ice", "arg", "usa", "usa"], ["usa", "usa", "ind", "ind", "usa", "chn", "ice", "arg", "usa", "usa"], ["usa", "usa", "usa", "ind", "usa", "chn", "ice", "arg", "usa", "usa"], ["usa", "usa", "ind", "ind", "usa", "chn", "ice", "usa", "usa", "usa"]]

# make a list of all indices for arrays
countryBoundaries = np.array(countryList)
allIndexes = np.arange(countryBoundaries.size)

# turn all important lists into arrays for numpy analyses
arr_1 = np.array(countryList)

shape = arr_1.shape

countries = arr_1.ravel()
urbanRural = np.array(urbanRural).ravel()
popProj = np.array(populationProjected).ravel()
allIndexes = np.array(allIndexes)

def addPopulation(populationProjected, pop, country, cellType, limit):
    randoms = np.all((countries == country, urbanRural == cellType), axis = 0)
    if np.sum(randoms) < 0:
        return populationProjected

    randomIndexes = np.random.choice(allIndexes[randoms], pop)
    np.add.at(populationProjected, randomIndexes, 1)

    print populationProjected.reshape(shape)
    # repeat the spillover function as long as there are cells above the limit
    # TODO: this may run into an infinite loop!
    counter = 0
    while np.amax(populationProjected) > limit:
        print np.amax(populationProjected)
        populationProjected = spillover(populationProjected, country, limit)

    return populationProjected

# Spillover: go into a loop where we remove population above limit and "push" them to the neighboring cells
# (I was thinking about calling this function "gentrify"...)
# IMPORTANT: we only select by country and whether the cell is overcrowded,
# so that we can spill over into other cell types, i.e. from urban to rural
def spillover(populationProjected, country, limit):

    print "spilling over..."

    overcrowded = np.all((countries == country, populationProjected > limit), axis = 0)
    # for every overcrowded cell, distribute the surplus population randomly among its neighbors
    for fullCell in allIndexes[overcrowded]:
        surplus = populationProjected[fullCell] - limit # by how much are we over the limit?
        # reset those cells to the limit value:
        populationProjected[fullCell] = limit
        # and move the extra people to the neighbors:
        wilsons = getNeighbours(fullCell, shape)
        rI = np.random.choice(wilsons, surplus)
        np.add.at(populationProjected, rI, 1)

    print populationProjected.reshape(shape)
    return populationProjected

# Returns an array of indexes that correspond to the 3x3 neighborhood of the index cell
# in a raveled (1D) matrix based on the # shape of the original (2D) matrix.
# Returns only neighbors within shape, exlcuding the input cell
def getNeighbours(index, shape):
    twoDIndex = oneDtoTwoD(index, shape)
    row = twoDIndex[0]
    col = twoDIndex[1]

    neighbors = []

    for r in range(-1, 2):
        for c in range(-1, 2):
            rn = row + r
            cn = col + c
            if r != 0 or c !=0: # don't add the original cell
                if 0 <= rn < shape[0] and 0 <= cn < shape[1]: # don't add neighbors that are outside of the shape!
                    neighbors.append(twoDtoOneD(rn, cn, shape))

    return neighbors


# Computes the "raveled" index from a 2D index. Shape is a tuple (rows, columns).
# WARNING: does NOT check whether row and col are outside of shape!
def twoDtoOneD(row, col, shape):
    return (row * shape[1]) + col



# Computes the 2D index as a tuple (row, column) from its "raveled" index.
# Shape is a tuple (rows, columns).
def oneDtoTwoD(index, shape):
    return int(index/shape[1]), int(index%shape[1])


print countries.reshape(shape)
print urbanRural.reshape(shape)
print "This is the original population distribution: "
print popProj.reshape(shape)
print "This is the sum of the original population: ", np.sum(populationProjected)

x = addPopulation(popProj, pop, country, urbanCell, limitPop)

print "This is the new population distribution: "
print x.reshape(shape)
print "This is the sum of the new population: ", np.sum(x)
