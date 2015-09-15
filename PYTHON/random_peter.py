import numpy as np

# create list of indices for cells that are either urban or rural
kind    = np.array(['urban','urban','rural','urban','urban','rural','rural'])

# create list of populations in these cells
pop     = np.array([1,      4,      10,      15,     4,      8,     12])

# create list of countries for these cells
country = np.array(['ISA', 'ISA',  'COD',   'COD',   'COD', 'ALG',  'ALG'])

#  we randomly choose 2 indices were kind is urban and country is COD
idx = np.random.choice(np.where(np.logical_and(kind=='urban', country == 'COD'))[0], 5)

# add 10 to the 2 random urban elements
np.add.at(pop, idx, 10)

print (pop)
