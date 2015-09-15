import numpy as np
import os

os.chdir('/Users/carsten/Dropbox/Code/CISC/Data/NumpyLayers')

# just to make sure they actually all have the same size now:

urbanrural = np.load("UrbanRural.npy")

print urbanrural.shape
print np.min(urbanrural)
print np.max(urbanrural)

nationoutlines = np.load("NationOutlines.npy")

print nationoutlines.shape
print np.min(nationoutlines)
print np.max(nationoutlines)

population = np.load("Population2000.npy")

print population.shape
print np.min(population)
print np.max(population)
