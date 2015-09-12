import numpy as np

kind = np.array(['urban','urban','rural','urban','urban','rural','rural'])
pop  = np.array([1,4,10,15,4,8,12])

# add 10 to 4 random elements
# pop[np.random.randint(7, size=4)] += 10

# add 10 to 4 random elements that are urban
randindexes = np.random.randint(7, size=4)
print randindexes

pop[(kind == 'urban') & randindexes] += 10


# if we make the number high enough so that indexes have to appear twice
# in the randomly generated numbers, the operation is only executed ONCE
# on these indexes -> not what we want
# pop[np.random.randint(7, size=20)] += 10

# add 10 to all urban cells
# pop[kind == 'urban'] += 10

# add 10 to the 2nd and 4th element
# pop[[1,3]] += 10

# take the square root of all rural cells
# pop = np.sqrt(pop[kind == 'rural'])

print pop

# adds X to N random elements in the input array
# def addTenToX(x):
#     return x+10
#
# addTenToX = np.vectorize(addTenToX)
#
# print addTenToX(np.random.choice(input, 12))
