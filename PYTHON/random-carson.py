import numpy as np

kind = np.array(['urban','urban','rural','urban','urban','rural','rural'])
pop  = np.array([1,4,10,15,4,8,12])

print pop
# First, we randomly choose 4 indexes where kind is urban
# (the [0] is because where returns a tuple):
idx = np.random.choice(np.where(kind=='urban')[0], 4)
# In my test, index was: array([4, 1, 0, 4])

print idx

np.add.at(pop, idx, 10)
# In my test, pop is now: array([11, 14, 10, 15, 24,  8, 12])

print pop
