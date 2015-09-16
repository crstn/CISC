import numpy as np

# create a 2D, 3x3 array of cells that are either urban or rural
kind = np.array([['urban','urban','rural'],
                 ['urban','urban','rural'],
                 ['rural','urban','rural']])

# create a 2D, 3x3 array of populations in these cells
pop  = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

#print np.sum(pop)

# create list of countries for these cells
country = np.array([['ISA', 'ISA', 'COD'],
                    ['COD', 'COD', 'ALG'],
                    ['ISA', 'COD', 'ALG']])

#  we randomly choose 5 indices were kind is urban and country is COD
matches = np.where(np.logical_and(kind=='urban', country == 'COD'))
idx = np.array([np.random.choice(matches[0], 5),np.random.choice(matches[1], 5)])

print idx

# add 1 to the 5 random urban elements
np.add.at(pop, idx, 1)

print (pop)

print np.sum(pop)
